#!/usr/bin/env python3
"""Build an evidence-backed pretest review; no model or test outcome access."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re

import pandas as pd

from solvent_select.freeze import queue_evidence
from solvent_select.study import load_registry
from solvent_select.views import sha256_file,write_json


def main():
    registry=load_registry();output=Path(registry['pretest_review'])
    if output.exists():raise FileExistsError('Preserve existing review')
    sources={__file__:sha256_file(__file__)}
    def pin(path):
        path=Path(path);sources[str(path)]=sha256_file(path);return json.loads(path.read_text())
    def verified(path,status):
        record=pin(path)
        if record.get('status')!=status:raise ValueError('Unpassed required evidence: '+str(path))
        for source,digest in record.get('evidence_sha256',{}).items():
            if sha256_file(source)!=digest:raise ValueError('Audit input changed: '+source)
            sources[source]=digest
        return record
    primary=verified('artifacts/audit/sourceqa/reconstruction.json','PASSED_ACTUAL_RECONSTRUCTION')
    for row in primary['comparisons']:
        if not row['identical'] or sha256_file(row['artifact'])!=row['expected_sha256']:raise ValueError('Primary reconstruction changed')
        sources[row['artifact']]=row['expected_sha256']
    strict=verified('artifacts/audit/strictqa/reconstruction.json','PASSED_ACTUAL_STRICTQA_RECONSTRUCTION')
    for row in strict['files']:
        if not row['identical'] or sha256_file(row['file'])!=row['expected_sha256']:raise ValueError('Strict reconstruction changed')
        sources[row['file']]=row['expected_sha256']
    resolution=verified('artifacts/audit/strictqa/partition_audit_resolution.json','RESOLVED_AND_REEVALUATED')
    if resolution['erratum_sha256']!=sha256_file('artifacts/audit/sourceqa/partition_audit_erratum.json'):
        raise ValueError('Erratum resolution refers to a different finding')
    baselines=verified('artifacts/audit/dualqa/completed_baseline_audit_full.json','PASSED_COMPLETED_BASELINE_AUDIT')
    if len(baselines['models'])!=14 or baselines['pending_models']:raise ValueError('Incomplete fitted-property provenance')
    isolation=verified('artifacts/audit/dualqa/query_label_isolation_v1.json','PASSED_CURRENT_METADATA_QUERY_AND_PROMPT_LABEL_ISOLATION')
    if isolation['target_files_read'] or isolation['test_predictions_generated']:raise ValueError('Wrong isolation audit scope')
    grading=verified('artifacts/audit/dualqa/independent_validation_scores_v8.json','PASSED_ALL_REGISTERED_VALIDATION_GRADING')
    if len(grading['runs'])!=62 or grading['pending_runs']:raise ValueError('Incomplete independent grading')
    repeat=pin('artifacts/audit/dualqa/independent_validation_scores_v8_reproduced.json')
    if repeat!=grading:raise ValueError('Grade reconstruction differs')
    inputs=verified('artifacts/audit/dualqa/completed_lm_input_audit_v9.json','PASSED_ALL_REGISTERED_LM_INPUTS')
    if len(inputs['completed_inferences'])!=38 or inputs['pending_inferences']:raise ValueError('Incomplete actual-input audit')
    train_queries=0;card_occurrences=0;adapter_checks=[]
    test_sets=[];development_sets=[]
    for protocol,config in registry['protocols'].items():
        manifest=pd.read_csv(config['split'],dtype={'solute_id':str})
        test_sets.append(set(manifest.loc[manifest.split=='test','solute_id']))
        development_sets.append(set(manifest.loc[manifest.split!='test','solute_id']))
        for family in ['sft','rag']:
            for seed in [17,42,2026]:
                name=config['run_prefix']+protocol+f'_qwen_{family}_{seed}';run=Path(config['runs_root'])/name
                audit=verified(Path('artifacts/audit/dualqa/training_evidence_completed')/name/'audit.json','PASSED_SAVED_TRAINING_EVIDENCE_AUDIT')
                metadata=pin(run/'run.json')
                if metadata['status']!='completed' or sha256_file(run/'run.json')!=audit['training_metadata_sha256_at_audit']:
                    raise ValueError('Training audit metadata differs')
                selected=Path(metadata['best_checkpoint'])/'adapter_model.safetensors';adapter=run/'adapter/adapter_model.safetensors'
                if sha256_file(selected)!=sha256_file(adapter):raise ValueError('Inference adapter differs from selected checkpoint')
                for p in [selected,adapter,run/'adapter/adapter_config.json']:sources[str(p)]=sha256_file(p)
                for key in ['train','val']:
                    row=audit[key]
                    if row['self_or_same_group_retrievals'] or row['non_training_records']:raise ValueError('Saved evidence isolation failure')
                    train_queries+=row['queries_checked'];card_occurrences+=row['cards_checked']
                adapter_checks.append({'run':str(run),'selected_checkpoint':metadata['best_checkpoint'],'adapter_sha256':sha256_file(adapter)})
    if test_sets[0]!=test_sets[1] or len(test_sets[0])!=144 or any(test_sets[0]&s for s in development_sets):
        raise ValueError('Global final-test isolation failed')
    controls=verified('artifacts/audit/dualqa/independent_consistency_full.json','PASSED_ALL_REGISTERED_CONSISTENCY_COMPARISONS')
    if len(controls['comparisons'])!=64:raise ValueError('Missing registered consistency comparison')
    if pin('artifacts/audit/dualqa/independent_consistency_full_reproduced.json')!=controls:raise ValueError('Consistency reconstruction differs')
    for kind,count in [('classical',4),('chemprop',6),('evidence',1)]:
        replay=pin(Path('artifacts/audit/dualqa/final_control_validation_replay')/kind/'summary.json')
        if replay['status']!='PASSED_REAL_VALIDATION_REPLAY' or len(replay['runs'])!=count or replay['protocol']!='scaffold':
            raise ValueError('Strict final-control validation replay failed')
        for row in replay['runs']:
            for path,digest in row['reference_sha256'].items():
                if sha256_file(path)!=digest:raise ValueError('Control reference changed')
                sources[path]=digest
    receipts=[]
    for stage in [*registry['queues'],{'plan':registry['consistency_plan'],'state':registry['consistency_state']}]:
        receipt=queue_evidence(stage['plan'],stage['state'])
        if receipt['pending'] or receipt['errors']:raise ValueError('Incomplete queue receipts')
        receipts.append(receipt)
        pin(stage['plan']);pin(stage['state'])
    decision=verified(registry['validation_decision'],'VALIDATION_DECISION_RECORDED')
    if decision['test_used'] or decision['selected_methods']!='ALL_REGISTERED' or decision['completed_methods']!=62:
        raise ValueError('Invalid validation-only method decision')
    log=Path('logs/pytest_full_pretest_restored_20261028.log');text=log.read_text()
    if not re.search(r'245 passed, 2 skipped',text):raise ValueError('Full restored-environment checks did not pass')
    sources[str(log)]=sha256_file(log)
    cp_log=Path('logs/tests_pretest_chemprop_actual_env.log')
    if not re.search(r'2 passed, 8 deselected',cp_log.read_text()):raise ValueError('Actual Chemprop checks did not pass')
    sources[str(cp_log)]=sha256_file(cp_log)
    preservation=pin('artifacts/audit/dualqa/continuation_state.json')
    for path,digest in preservation['drafts_preserved_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('Deferred manuscript was modified')
    checks={name:True for name in ['data_reconstruction','split_leakage','label_isolation','retrieval_provenance',
                                  'metric_hand_checks','all_validation_receipts','consistency_validation','manuscript_claim_scope']}
    write_json(output,{'status':'PASSED_PREFREEZE_REVIEW','created_utc':datetime.now(timezone.utc).isoformat(),
        'checks':checks,'registered_validation_methods':62,'consistent_comparison_inputs_checked':64,
        'training_and_teacher_validation_query_occurrences':train_queries,'training_card_occurrences':card_occurrences,
        'adapter_checks':adapter_checks,'software_checks':'245passed2skipped in restored main environment; the two Chemprop-specific tests passed in their actual separate environment.',
        'queue_receipts':receipts,'strict_erratum_resolution':'Verified companion record; historical validation exposure remains disclosed.',
        'scope':'Project-agent code/provenance review with separate independent arithmetic implementations; not independent human peer review. Test targets have not been deserialized and test predictions have not been generated.',
        'manuscript_claim_scope':'Writing deferred by user. Nine original draft hashes match the preserved snapshot; drafts remain incomplete and are not claimed as final scientific results. No publication-readiness judgment.',
        'test_target_values_read':False,'test_predictions_generated':False,'retained_methods':'ALL_REGISTERED',
        'validation_joint_benefit_supported':{p:row['joint_criteria_supported'] for p,row in decision['contrasts'].items()},
        'evidence_sha256':sources})


if __name__=='__main__':main()

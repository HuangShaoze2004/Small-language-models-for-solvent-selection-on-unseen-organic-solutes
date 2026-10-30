#!/usr/bin/env python3
"""Close the strict-curation erratum only with current data and rerun evidence.

Does not erase historical validation exposure, alter data, or release a test.
"""
import json
from pathlib import Path
import re
import tarfile

import pandas as pd

from solvent_select.freeze import queue_evidence
from solvent_select.study import load_registry,property_names,evidence_name
from solvent_select.views import sha256_file,write_json


def source_scope(metadata,roles,selected,flagged):
    assigned=metadata.assign(role=metadata.solute_id.map(roles))
    if assigned.role.isna().any():raise ValueError('Missing original strict role')
    train=set(assigned.loc[assigned.role=='train','source_doi'])
    forbidden=set(assigned.loc[assigned.role!='train','source_doi'])
    eligible=train-forbidden
    if not selected<=eligible or not flagged<=selected:raise ValueError('Source selection or quarantine crosses a held-out role')
    return eligible


def main():
    output=Path('artifacts/audit/strictqa/partition_audit_resolution.json')
    if output.exists():raise FileExistsError('Preserve the original resolution record')
    registry=load_registry();config=registry['protocols']['scaffold']
    policy_path=Path('configs/strict_source_audit_v1.json');policy=json.loads(policy_path.read_text())
    quarantine_path=Path('configs/strict_quarantine_v1.json');quarantine=json.loads(quarantine_path.read_text())
    erratum=Path('artifacts/audit/sourceqa/partition_audit_erratum.json')
    sources={str(p):sha256_file(p) for p in [policy_path,quarantine_path,erratum,Path(__file__)]}
    def pin(path):
        path=Path(path);sources[str(path)]=sha256_file(path);return json.loads(path.read_text())
    if sha256_file(policy['measurement_path'])!=policy['measurement_sha256'] or sha256_file(policy['split_path'])!=policy['split_sha256']:
        raise ValueError('Original strict inputs changed')
    original=pd.read_parquet(policy['measurement_path'],columns=['record_id','solute_id','source_doi'])
    old_roles=pd.read_csv(policy['split_path'],dtype={'solute_id':str}).set_index('solute_id').split.to_dict()
    selection=pin('artifacts/audit/strictqa/training_sources/selection.json')
    download=pin('artifacts/audit/strictqa/training_sources/download_manifest.json')
    selected=set(selection['source_dois']);flagged=set(quarantine['source_dois'])
    eligible=source_scope(original,old_roles,selected,flagged)
    if selection['policy_sha256']!=sha256_file(policy_path) or selection['selection_before_numeric_comparison'] is not True:
        raise ValueError('Source selection lacks its original policy registration')
    if set(download['files'])!=selected:raise ValueError('Selected source files differ')
    archive=Path('data/raw/source_audit/ThermoML.v2020-09-30.tgz')
    if sha256_file(archive)!=selection['archive_sha256'] or download['archive_sha256']!=selection['archive_sha256']:
        raise ValueError('Archive changed')
    available=set()
    with tarfile.open(archive,'r:gz') as stream:
        for member in stream:
            match=re.search(r'(10\.\d{4,9}/.+)\.json$',member.name)
            if member.isfile() and match:available.add(match[1].lower())
    if selected!=eligible&available:raise ValueError('Strict-training-exclusive archive source selection is incomplete')
    sources[str(archive)]=selection['archive_sha256']
    for record in download['files'].values():
        if sha256_file(record['path'])!=record['sha256']:raise ValueError('Selected source JSON changed')
        sources[record['path']]=record['sha256']
    reconstruction=pin('artifacts/audit/strictqa/final_source_audit_reconstruction_check.json')
    if reconstruction['status']!='PASSED_ACTUAL_STRICT_TRAINING_SOURCE_AUDIT_RECONSTRUCTION':raise ValueError('Strict numeric audit not reconstructed')
    for row in reconstruction['files']:
        if not row['byte_identical'] or sha256_file(row['original'])!=row['sha256'] or sha256_file(row['reproduced'])!=row['sha256']:
            raise ValueError('Strict numeric reconstruction differs')
        sources[row['original']]=row['sha256'];sources[row['reproduced']]=row['sha256']
    summary=pd.read_csv(quarantine['source_summary'])
    if set(summary.loc[summary.flagged_records>0,'source_doi'])!=flagged or sha256_file(quarantine['source_summary'])!=quarantine['source_summary_sha256']:
        raise ValueError('Quarantine differs from the recorded strict numeric audit')
    removed=original[original.source_doi.isin(flagged)]
    if set(removed.solute_id.map(old_roles))!={'train'}:raise ValueError('Quarantine touched a held-out record')
    current=pd.read_parquet(config['measurements'],columns=['record_id','solute_id','leakage_group','scaffold_group'])
    if set(current.record_id)!=set(original.record_id)-set(removed.record_id):raise ValueError('Incorrect strict record membership')
    roles=pd.read_csv(config['split'],dtype={'solute_id':str}).set_index('solute_id').split.to_dict()
    if any(old_roles.get(sid)!=role for sid,role in roles.items()):raise ValueError('Role changed')
    for partition in ['val','test']:
        if {s for s,r in old_roles.items() if r==partition}!={s for s,r in roles.items() if r==partition}:raise ValueError('Held-out identity removed')
    assigned=current.assign(role=current.solute_id.map(roles))
    if assigned.role.isna().any() or any(assigned.groupby(key).role.nunique().max()!=1 for key in ['leakage_group','scaffold_group']):
        raise ValueError('Group overlap in current strict roles')
    data=pin(config['audit']);manifest=pin('artifacts/audit/strictqa/data_manifest.json')
    if data['validation_or_test_labels_used_for_quarantine'] or data['numeric_labels_replaced']:raise ValueError('Quarantine policy violated')
    if len(removed)!=796 or len(flagged)!=16 or len(roles)!=969:raise ValueError('Unexpected fixed strict membership counts')
    for path,digest in manifest['outputs'].items():
        if sha256_file(path)!=digest:raise ValueError('Curated scientific artifact changed')
        sources[path]=digest
    for path,digest in data['heldout_query_target_bytes_preserved'].items():
        if sha256_file(path)!=digest:raise ValueError('Original held-out bytes changed')
        sources[path]=digest
    rebuilt=pin('artifacts/audit/strictqa/reconstruction.json')
    if rebuilt['status']!='PASSED_ACTUAL_STRICTQA_RECONSTRUCTION' or not all(r['identical'] and sha256_file(r['file'])==r['expected_sha256'] for r in rebuilt['files']):
        raise ValueError('Curated reconstruction evidence failed')
    baseline=pin('artifacts/audit/dualqa/completed_baseline_audit_full.json')
    if len(baseline['models'])!=14 or len(baseline['direct_evidence'])!=10 or baseline['pending_models']:raise ValueError('Baseline provenance incomplete')
    grade=pin('artifacts/audit/dualqa/independent_validation_scores_v8.json')
    if len(grade['runs'])!=62 or grade['pending_runs'] or any(r['status']!='PASSED' for r in grade['runs']):raise ValueError('Independent grading incomplete')
    inputs=pin('artifacts/audit/dualqa/completed_lm_input_audit_v9.json')
    if len(inputs['completed_inferences'])!=38 or inputs['pending_inferences']:raise ValueError('Actual LM-input verification incomplete')
    consistency=pin('artifacts/audit/dualqa/independent_consistency_full.json')
    if len(consistency['comparisons'])!=64 or consistency['status']!='PASSED_ALL_REGISTERED_CONSISTENCY_COMPARISONS':raise ValueError('Consistency audit incomplete')
    stage=next(x for x in registry['queues'] if x['role']=='scaffold_models');receipt=queue_evidence(stage['plan'],stage['state'])
    if receipt['errors'] or receipt['pending'] or receipt['tasks']!=38:raise ValueError('Strict queue not fully verified')
    for path in [stage['plan'],stage['state']]:pin(path)
    for name in property_names('scaffold',config):pin(Path(config['runs_root'])/name/'metadata.json')
    for family in ['sft','rag']:
        for seed in [17,42,2026]:
            name=f'strictqa_scaffold_qwen_{family}_{seed}';evidence=pin(Path('artifacts/audit/dualqa/training_evidence_completed')/name/'audit.json')
            if evidence['training_status_at_audit']!='completed' or evidence['protocol']!='scaffold':raise ValueError('Unverified strict training')
            if evidence['train']['non_training_records'] or evidence['train']['self_or_same_group_retrievals']:raise ValueError('Training evidence leakage')
            pin(Path(config['runs_root'])/name/'run.json')
    for kind,count in [('classical',4),('chemprop',6),('evidence',1)]:
        replay=pin(Path('artifacts/audit/dualqa/final_control_validation_replay')/kind/'summary.json')
        if replay['status']!='PASSED_REAL_VALIDATION_REPLAY' or replay['protocol']!='scaffold' or len(replay['runs'])!=count:
            raise ValueError('Strict final-control helpers not validated')
    write_json(output,{'status':'RESOLVED_AND_REEVALUATED','erratum_sha256':sha256_file(erratum),
        'strict_original_training_exclusive_sources':len(selected),'quarantined_training_dois':len(flagged),'quarantined_training_records':len(removed),
        'strict_roles':{p:sum(r==p for r in roles.values()) for p in ['train','val','test']},'strict_registered_queue':receipt,
        'checks':{'source_selection_train_exclusive_and_complete':True,'numeric_source_audit_reconstructed':True,
                  'quarantine_training_only':True,'heldout_bytes_and_roles_preserved':True,'all_affected_models_refitted_and_validated':True,
                  'all_validation_scores_independently_regraded':True,'actual_training_and_inference_boundaries_verified':True,
                  'consistency_and_final_control_helpers_validated':True},'test_target_values_read':False,'test_predictions_generated':False,
        'historical_limitation':'This resolves the fold-specific curation implementation and required reruns. It cannot undo the analyst\'s historical strict-validation exposure. Old sourceqa strict results remain diagnostics; the two development protocols are not independent replications. The shared global final test remains unscored.',
        'evidence_sha256':sources})


if __name__=='__main__':main()

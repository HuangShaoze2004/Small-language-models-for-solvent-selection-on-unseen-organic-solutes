"""Read-only pre-test audit and immutable prediction-configuration freeze.

Incomplete validation, missing decisions, altered receipts or unknown test
cohorts refuse freezing. This command never loads test target values, generates
predictions or calculates a test score.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from types import SimpleNamespace

from .llm import load_config,prediction_fingerprint,read_jsonl,validate_query_manifest
from .views import sha256_file,write_json
from .study import DEFAULT_REGISTRY,load_registry,queue_pairs,property_names,evidence_name,verify_adapter_boundary

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_QUEUES=[('configs/development_queue_sourceqa_primary.json','artifacts/development_queue_sourceqa_primary/state.json'),
                ('configs/development_queue_identity_sourceqa_v2.json','artifacts/development_queue_identity_sourceqa_v2/state.json'),
                ('configs/development_queue_scaffold_strictqa_v2.json','artifacts/development_queue_scaffold_strictqa_v2/state.json')]


def layouts():
    if Path(DEFAULT_REGISTRY).exists():return load_registry()['protocols']
    # Compatibility for isolated unit fixtures and explicitly supplied old plans;
    # actual freezing below always requires a registered dual-protocol registry.
    return {p:{'run_prefix':'sourceqa_','runs_root':'artifacts/runs_sourceqa',
               'consistency_runs_root':'artifacts/consistency/sourceqa'} for p in ['identity','scaffold']}


def queue_evidence(plan_path,state_path):
    if not Path(plan_path).exists():
        return {'plan':str(plan_path),'state':str(state_path),'plan_sha256':None,'state_sha256':None,
                'tasks':0,'pending':[{'task':'unprepared_queue','status':'missing_plan'}],'errors':['Missing queue plan: '+str(plan_path)]}
    plan=json.loads(Path(plan_path).read_text())
    state=json.loads(Path(state_path).read_text()) if Path(state_path).exists() else {'tasks':{},'status':'not_started'}
    errors=[]
    if Path(state_path).exists() and state.get('plan_sha256')!=sha256_file(plan_path):errors.append('Queue state/plan SHA256 mismatch')
    for path,expected in plan['input_sha256'].items():
        if not Path(path).exists() or sha256_file(path)!=expected:errors.append('Pinned input changed: '+path)
    pending=[]
    for task in plan['tasks']:
        record=state.get('tasks',{}).get(task['id'],{})
        if record.get('status')!='completed':
            pending.append({'task':task['id'],'status':record.get('status','not_started')});continue
        if record.get('exit_code')!=0:errors.append('Completed task has nonzero exit: '+task['id'])
        for output in task['outputs']:
            path=output['path']
            if not Path(path).exists() or record.get('output_sha256',{}).get(path)!=sha256_file(path):
                errors.append('Completed output missing/modified: '+path);continue
            if 'status' in output and json.loads(Path(path).read_text()).get('status')!=output['status']:
                errors.append('Child not complete: '+path)
    if state.get('status')!='completed' and not pending:
        errors.append('Queue has not reached completed status')
    return {'plan':str(plan_path),'state':str(state_path),'plan_sha256':sha256_file(plan_path),
            'state_sha256':sha256_file(state_path) if Path(state_path).exists() else None,'tasks':len(plan['tasks']),'pending':pending,'errors':errors}


def verified_record(path,status):
    path=Path(path)
    if not path.exists():return None,'Missing '+str(path)
    data=json.loads(path.read_text())
    if data.get('status')!=status:return None,'Unexpected status in '+str(path)
    for source,expected in data.get('evidence_sha256',{}).items():
        if not Path(source).exists() or sha256_file(source)!=expected:return None,'Changed evidence: '+source
    if not data.get('evidence_sha256'):return None,'No auditable evidence in '+str(path)
    return data,None


def consistency_paths(queues=DEFAULT_QUEUES):
    """All planned core comparisons, rather than a favorable result subset."""
    paths=[]
    layout=layouts()
    for protocol,config in layout.items():
        parent=Path(config['consistency_runs_root'])/protocol
        names=property_names(protocol,config)
        names += [f"{config['run_prefix']}{protocol}_chemprop_{seed}_canonical_control" for seed in [17,42,2026]]
        names += [evidence_name(protocol,config,mode) for mode in ['text','structure','condition','no_temperature','no_solvent']]
        names += [evidence_name(protocol,config,'text')+'_canonical_control']
        paths += [parent/name/'evaluation/metrics.json' for name in names]
    for plan_path,_ in queues:
        for task in json.loads(Path(plan_path).read_text())['tasks']:
            cmd=task['command']
            if ('solvent_select.llm' in cmd and 'predict' in cmd
                    and option(cmd,'--retrieval-mode') in ['none','text','structure','condition']):
                paths.append(Path(layout[option(cmd,'--protocol')]['consistency_runs_root'])/option(cmd,'--protocol')/
                             (task['id'].removesuffix('_val')+'_consistency_val')/'evaluation/metrics.json')
    return paths


def preflight(queues=None,decision=None,review=None):
    registry=load_registry() if Path(DEFAULT_REGISTRY).exists() else None
    if queues is None:queues=queue_pairs(load_registry())
    decision=decision or (registry['validation_decision'] if registry else 'artifacts/reports/sourceqa/validation_decision.json')
    review=review or (registry['pretest_review'] if registry else 'artifacts/audit/sourceqa/pretest_review.json')
    audits=[queue_evidence(*pair) for pair in queues]
    gaps=[issue for q in audits for issue in q['errors']]
    erratum=Path('artifacts/audit/sourceqa/partition_audit_erratum.json')
    if erratum.exists():
        resolution,error=verified_record('artifacts/audit/strictqa/partition_audit_resolution.json','RESOLVED_AND_REEVALUATED')
        if error or resolution.get('erratum_sha256')!=sha256_file(erratum):
            gaps.append('Strict source-curation boundary remediation is not completed; see '+str(erratum))
    pending=sum(len(q['pending']) for q in audits)
    decision_data,error=verified_record(decision,'VALIDATION_DECISION_RECORDED')
    if error:gaps.append(error)
    elif decision_data.get('test_used') is not False or decision_data.get('selected_methods')!='ALL_REGISTERED':
        gaps.append('Decision must use validation only and retain all registered methods')
    review_data,error=verified_record(review,'PASSED_PREFREEZE_REVIEW')
    if error:gaps.append(error)
    elif not all(review_data.get('checks',{}).get(key) is True for key in
                 ['data_reconstruction','split_leakage','label_isolation','retrieval_provenance',
                  'metric_hand_checks','all_validation_receipts','consistency_validation','manuscript_claim_scope']):
        gaps.append('Required pre-freeze review checks are missing or failed')
    missing_consistency=[]
    for path in consistency_paths(queues):
        if not path.exists():
            missing_consistency.append(str(path));continue
        metrics=json.loads(path.read_text())
        if metrics.get('partition')!='val' or metrics.get('queries')!=1024:
            gaps.append('Incomplete/wrong-partition consistency result: '+str(path));continue
        for source,expected in metrics.get('inputs',{}).items():
            if not Path(source).exists() or sha256_file(source)!=expected:
                gaps.append('Consistency evidence changed: '+source)
    for protocol,config in layouts().items():
        for name in property_names(protocol,config):
            path=Path(config['runs_root'])/name/'metadata.json'
            if not path.exists() or json.loads(path.read_text()).get('status')!='completed':
                gaps.append('Incomplete property model: '+name)
        if registry:
            for mode in ['text','structure','condition','no_temperature','no_solvent']:
                name=evidence_name(protocol,config,mode)
                path=Path(config['runs_root'])/name/'predictions.jsonl.run.json'
                if not path.exists() or json.loads(path.read_text()).get('status')!='completed':
                    gaps.append('Incomplete direct evidence validation: '+name)
    return {'status':'PASSED_PREFREEZE' if not gaps and not pending and not missing_consistency else 'INCOMPLETE',
            'created_utc':datetime.now(timezone.utc).isoformat(),'queue_audits':audits,'pending_tasks':pending,
            'missing_consistency':missing_consistency,
            'gaps':gaps,'decision_path':str(decision),'review_path':str(review),
            'test_target_values_read':False,'test_predictions_generated':False,'code_sha256':sha256_file(__file__)}


def option(command,name,default=None):
    return command[command.index(name)+1] if name in command else default


def test_prediction_spec(command,protocol,folder,consistency=False):
    inner=command[command.index('--')+1:] if '--' in command else command.copy()
    config=load_config(option(inner,'--config'),seed=int(option(inner,'--seed','17')),
                       retrieval_mode=option(inner,'--retrieval-mode'))
    manifest=option(inner,'--manifest',f'data/splits/sourceqa/{protocol}.csv')
    tag=Path(manifest).parent.name
    consistency_root=Path('data/consistency')/(tag if tag!='splits' else '')
    validation_queries=option(inner,'--queries',str(Path('data/views')/(tag if tag!='splits' else '')/protocol/'val_queries.jsonl'))
    queries=(str(consistency_root/f'{protocol}_test/transformed_queries.jsonl') if consistency
             else str(Path(validation_queries).with_name('test_queries.jsonl')))
    args=SimpleNamespace(prediction_kind='evidence' if 'evidence' in inner else 'llm',queries=queries,adapter=option(inner,'--adapter'),
                         manifest=manifest,measurements=option(inner,'--measurements'),limit=None)
    fingerprint,digest=prediction_fingerprint(args,config)
    return {'fingerprint':fingerprint,'sha256':digest,'validation_command':command,
            'queries':queries,'adapter':args.adapter,'retrieval_mode':config.retrieval_mode,
            'freeze_manifest':str(folder/f'{protocol}.json'),'consistency':consistency,
            'prediction_kind':args.prediction_kind,'validation_run_id':option(command,'--run-id')}


def freeze(output,analysis_plan,queues=None,decision=None,review=None):
    audit=preflight(queues,decision,review)
    if audit['status']!='PASSED_PREFREEZE':
        raise ValueError('Pre-test audit incomplete; no FROZEN manifest written')
    output=Path(output)
    if output.exists():raise FileExistsError('A freeze is immutable; never overwrite it')
    analysis=json.loads(Path(analysis_plan).read_text())
    if analysis.get('status')!='REGISTERED_BEFORE_TEST':raise ValueError('Final analysis plan not registered')
    registry=load_registry(analysis['registry_path'])
    if sha256_file(analysis['registry_path'])!=analysis['registry_sha256']:raise ValueError('Study registry changed')
    queues=queues or queue_pairs(registry)
    decision=decision or registry['validation_decision'];review=review or registry['pretest_review']
    specifications={protocol:[] for protocol in ['identity','scaffold']}
    for plan_path,_ in queues:
        for task in json.loads(Path(plan_path).read_text())['tasks']:
            cmd=task['command']
            if 'solvent_select.llm' not in cmd or 'predict' not in cmd:continue
            protocol=option(cmd,'--protocol')
            specifications[protocol].append(test_prediction_spec(cmd,protocol,output))
            mode=option(cmd,'--retrieval-mode')
            if mode in ['none','text','structure','condition']:
                specifications[protocol].append(test_prediction_spec(cmd,protocol,output,True))
    # Direct evidence is a required fixed workflow using the identical cards,
    # tokenizer and budget. It must be explicitly frozen too.
    for protocol,config in registry['protocols'].items():
        for mode in ['text','structure','condition','no_temperature','no_solvent']:
            name=evidence_name(protocol,config,mode)
            path=Path(config['runs_root'])/name/'predictions.jsonl.run.json'
            metadata=json.loads(path.read_text())
            if metadata.get('status')!='completed':raise ValueError('Incomplete direct evidence validation')
            cmd=['.venv/bin/python','-m','solvent_select.llm','evidence','--config','configs/llm_formal_sft.yaml',
                 '--seed','17','--retrieval-mode',mode,'--manifest',config['split'],
                 '--measurements',config['measurements'],'--queries',str(Path(config['views'])/'val_queries.jsonl')]
            spec=test_prediction_spec(cmd,protocol,output)
            spec['validation_run_id']=name
            expected=metadata['fingerprint']
            if any(spec['fingerprint'][key]!=expected[key] for key in
                   ['config','manifest_sha256','measurements_sha256','implementation_sha256','prediction_kind']):
                raise ValueError('Direct evidence configuration differs from validation')
            specifications[protocol].append(spec)
            transformed=test_prediction_spec(cmd,protocol,output,True);transformed['validation_run_id']=name
            specifications[protocol].append(transformed)
    manifests={}
    for protocol,config in registry['protocols'].items():
        query=Path(config['views'])/'test_queries.jsonl'
        manifest=Path(config['split'])
        rows=read_jsonl(query);validate_query_manifest(rows,manifest)
        if {q['split'] for q in rows}!={'test'}:raise ValueError('Final query cohort is not test')
        consistency=Path(config['consistency_data_root'])/f'{protocol}_test/manifest.json'
        assets={}
        property_models=[]
        controls=[]
        for name in analysis['property_methods'][protocol]:
            path=Path(config['runs_root'])/name/'metadata.json'
            meta=json.loads(path.read_text())
            if meta.get('status')!='completed':raise ValueError('Incomplete frozen property model')
            if meta['split_sha256']!=sha256_file(manifest):raise ValueError('Property split differs from release')
            assets[str(path)]=sha256_file(path)
            run=path.parent
            weights=Path(meta['best_checkpoint']) if 'best_checkpoint' in meta else run/'model.joblib'
            assets[str(weights)]=sha256_file(weights)
            for key,hash_key in [('measurements','measurement_sha256'),('splits','split_sha256')]:
                if sha256_file(meta[key])!=meta[hash_key]:raise ValueError('Trained property input changed')
                assets[meta[key]]=meta[hash_key]
            if (run/'scalers.json').exists():assets[str(run/'scalers.json')]=sha256_file(run/'scalers.json')
            property_models.append({'name':name,'run':str(run),'method':meta['method'],'seed':meta['seed']})
            controls.append({'id':name,'kind':'property','model_run':str(run),'canonicalize_query':False})
            if meta['method']=='chemprop':
                controls.append({'id':name+'_canonical_control','kind':'property','model_run':str(run),'canonicalize_query':True})
        controls.append({'id':f"{config['run_prefix']}{protocol}_evidence_text_canonical_control",'kind':'evidence',
                         'canonicalize_query':True,'retrieval_mode':'text','config':'configs/llm_formal_sft.yaml',
                         'measurements':config['measurements']})
        adapter_assets={}
        for spec in specifications[protocol]:
            if not spec['adapter']:continue
            adapter=Path(spec['adapter']);training=adapter.parent/'run.json'
            meta=json.loads(training.read_text())
            if meta.get('status')!='completed':raise ValueError('Adapter training not complete')
            verify_adapter_boundary(meta,config,spec['fingerprint']['config'])
            spec['adaptation']='sft' if meta['config']['retrieval_mode']=='none' else 'rag_sft'
            for asset in [training,adapter/'adapter_config.json',adapter/'adapter_model.safetensors']:
                adapter_assets[str(asset)]=sha256_file(asset)
            adapter_assets.update(meta['inputs'])
        code={str(p):sha256_file(p) for p in sorted(Path('src/solvent_select').glob('*.py'))}
        manifests[protocol]={'status':'FROZEN','created_utc':datetime.now(timezone.utc).isoformat(),
             'protocol_path':str(analysis_plan),'protocol_sha256':sha256_file(analysis_plan),
             'registry_path':analysis['registry_path'],'registry_sha256':analysis['registry_sha256'],
             'measurement_path':config['measurements'],'runs_root':config['runs_root'],
             'consistency_runs_root':config['consistency_runs_root'],'adapter_assets':adapter_assets,
             'query_path':str(query),'query_sha256':sha256_file(query),'query_count':len(rows),
             'target_path':str(Path(config['views'])/'test_targets.jsonl'),
             'target_sha256':sha256_file(Path(config['views'])/'test_targets.jsonl'),
             'split_sha256':sha256_file(manifest),'split_path':str(manifest),
             'consistency_manifest_path':str(consistency),
             'consistency_manifest_sha256':sha256_file(consistency),
             'prediction_fingerprints':[s['sha256'] for s in specifications[protocol]],
             'prediction_specifications':specifications[protocol],
             'property_assets':assets,'code_sha256':code,
             'property_models':property_models,'cpu_consistency_controls':controls,
             'error_analysis_code_sha256':sha256_file('src/solvent_select/error_analysis.py'),
             'error_analysis_config_sha256':sha256_file('configs/error_analysis.yaml'),
             'decision_sha256':sha256_file(decision),'pretest_review_sha256':sha256_file(review),
             'decision_path':str(decision),'pretest_review_path':str(review),
             'environment_locks':{p:sha256_file(p) for p in ['requirements.lock','requirements-chemprop.lock']},
             'scripts_sha256':{str(p):sha256_file(p) for p in sorted(Path('scripts').glob('*.py'))},
             'foundation_model_snapshot_path':'artifacts/audit/model_snapshot.json',
             'foundation_model_snapshot_sha256':sha256_file('artifacts/audit/model_snapshot.json'),
             'validation_consistency_sha256':{str(p):sha256_file(p) for p in consistency_paths(queues)},
             'queue_receipts':audit['queue_audits'],
             'test_target_values_read':False,'note':'Target bytes hashed only; no target values loaded. Test results cannot select methods or hyperparameters.'}
    if manifests['identity']['query_sha256']!=manifests['scaffold']['query_sha256']:
        raise ValueError('Protocols must share the identical global test cohort')
    output.mkdir(parents=True)
    for protocol,value in manifests.items():write_json(output/f'{protocol}.json',value)
    write_json(output/'preflight.json',audit)
    return manifests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','freeze'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--analysis-plan',default='configs/final_analysis_plan_dualqa_v2.json')
    parser.add_argument('--decision')
    parser.add_argument('--review')
    args=parser.parse_args()
    if args.command=='preflight':
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        result=preflight(decision=args.decision,review=args.review);write_json(args.output,result)
        print(json.dumps({'status':result['status'],'pending_tasks':result['pending_tasks'],'gaps':result['gaps']},indent=2))
    else:
        result=freeze(args.output,args.analysis_plan,decision=args.decision,review=args.review)
        print(json.dumps({'status':'FROZEN','protocols':list(result)},indent=2))


if __name__=='__main__':main()

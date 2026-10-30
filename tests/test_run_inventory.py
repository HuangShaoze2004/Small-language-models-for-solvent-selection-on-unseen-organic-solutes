import importlib.util
import json
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location('run_inventory', Path(__file__).parents[1] / 'scripts/build_run_inventory.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def test_signal_saved_training_is_not_completed_just_because_exit_zero():
    wrapper = {'status': 'completed', 'exit_code': 0}
    assert inventory.effective_status(wrapper, {'status': 'interrupted'}) == 'interrupted'
    assert inventory.effective_status(wrapper, {'status': 'failed'}) == 'failed'
    assert inventory.effective_status(wrapper, {'status': 'completed'}) == 'completed'
    assert inventory.effective_status(wrapper, {'status': 'preparing'}) == 'UNVERIFIED_CHILD_COMPLETION'
    assert inventory.effective_status({'status': 'interrupted', 'exit_code': -15}, {'status': 'predicting'}) == 'interrupted'


def test_standalone_evidence_and_consistency_are_in_master_inventory(tmp_path):
    evidence=tmp_path/'artifacts/runs_sourceqa/sourceqa_identity_evidence_condition_val'
    evidence.mkdir(parents=True)
    (evidence/'predictions.jsonl.run.json').write_text(json.dumps({'status':'completed',
        'fingerprint':{'prediction_kind':'evidence','config':{'retrieval_mode':'condition','seed':17}}}))
    control=tmp_path/'artifacts/consistency/sourceqa/scaffold/sourceqa_scaffold_chemprop_17'
    control.mkdir(parents=True)
    (control/'run.json').write_text(json.dumps({'status':'completed','test_labels_read':False}))
    (control/'evaluation').mkdir()
    (control/'evaluation/metrics.json').write_text(json.dumps({'partition':'val','queries':1024,
        'by_variant_task':{'equivalent_smiles/pair':{'exact_agreement':1.}}}))
    output=tmp_path/'inventory.csv'
    result=inventory.build(tmp_path,output)
    frame=pd.read_csv(output)
    assert result['runs']==2
    assert set(frame.status)=={'completed'}
    assert set(frame.protocol)=={'identity','scaffold'}
    assert frame.wrapper_evidence.isna().all()
    row=frame[frame.kind=='validation_input_consistency'].iloc[0]
    assert json.loads(row.validation_consistency_metrics)['evaluation']['queries']==1024


def test_repaired_strict_models_have_protocol_and_distinct_training_namespace(tmp_path):
    paths = {'identity': {'run_prefix':'sourceqa_', 'runs_root':'artifacts/runs_sourceqa',
                         'consistency_runs_root':'artifacts/consistency/sourceqa','data_tag':'sourceqa'},
             'scaffold': {'run_prefix':'strictqa_', 'runs_root':'artifacts/runs_strictqa',
                          'consistency_runs_root':'artifacts/consistency/strictqa','data_tag':'strictqa'}}
    registry = tmp_path/'configs/study_registry_dualqa_v2.json'
    registry.parent.mkdir()
    registry.write_text(json.dumps({'protocols':paths}))
    for root,name in [('runs_strictqa','strictqa_scaffold_nearest'),
                      ('runs_sourceqa','sourceqa_scaffold_nearest')]:
        folder=tmp_path/'artifacts'/root/name
        folder.mkdir(parents=True)
        (folder/'metadata.json').write_text(json.dumps({'status':'completed','method':'nearest','seed':17,
            'splits':'data/splits/'+('strictqa' if root=='runs_strictqa' else 'sourceqa')+'/scaffold.csv'}))
    output=tmp_path/'inventory.csv'
    inventory.build(tmp_path,output)
    frame=pd.read_csv(output).set_index('run_id')
    assert set(frame.protocol)=={'scaffold'}
    assert frame.loc['strictqa_scaffold_nearest','data_tag']=='strictqa'
    assert frame.loc['strictqa_scaffold_nearest','study_scope']=='REGISTERED_DATA_NAMESPACE'
    assert frame.loc['sourceqa_scaffold_nearest','data_tag']=='sourceqa'
    assert frame.loc['sourceqa_scaffold_nearest','study_scope']=='HISTORICAL_OR_UNCLASSIFIED'


def test_cpu_timing_records_are_distinct_experiments_with_their_actual_scope(tmp_path):
    folder=tmp_path/'artifacts/latency/property_v1/strictqa_scaffold_nearest'
    folder.mkdir(parents=True)
    record={'status':'COMPLETED_VALIDATION_CPU_TIMING','protocol':'scaffold',
            'run':'artifacts/runs_strictqa/strictqa_scaffold_nearest','method':'nearest','seed':17,
            'accelerator':'cpu','cpu_threads':4,'public_questions':10,'unique_candidate_calls':12,
            'candidate_appearances':20,'model_loading_seconds':.2,'timings':[{'repetition':0,'seconds':.1}],
            'implementation':{'feature_scope':'combined'}}
    (folder/'run.json').write_text(json.dumps(record))
    output=tmp_path/'inventory.csv'
    manifest=inventory.build(tmp_path,output)
    assert manifest['runs']==1
    row=pd.read_csv(output).iloc[0]
    assert row.kind=='validation_cpu_latency'
    assert row.protocol=='scaffold'
    assert row.status=='completed'
    assert json.loads(row.validation_cpu_timing)['public_questions']==10
    assert pd.isna(row.train_records)

"""Actual registered public-input hashes plus synthetic task-graph checks.

These tests never read test target values, execute a model, or create a release.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from solvent_select.freeze import option,queue_evidence,test_prediction_spec as prediction_spec
from solvent_select.final_plan import reusable_zero_shot
from solvent_select.pipeline import live_queue
from solvent_select.study import load_registry,property_names
from solvent_select.study import verify_adapter_boundary
from solvent_select.views import sha256_file
from test_final_plan import releases
from solvent_select.final_plan import assemble

ROOT=Path(__file__).parents[1]


def test_registered_strict_tasks_use_repaired_data_and_no_test_command():
    registry=load_registry(ROOT/'configs/study_registry_dualqa_v2.json')
    assert registry['protocols']['identity']['data_tag']=='sourceqa'
    assert registry['protocols']['scaffold']['data_tag']=='strictqa'
    path=ROOT/'configs/development_queue_scaffold_strictqa_v2.json'
    plan=json.loads(path.read_text())
    assert len(plan['tasks'])==38
    for task in plan['tasks']:
        assert task['id'].startswith('strictqa_scaffold_')
        assert '--allow-test' not in task['command']
        assert not any('test_queries' in token or 'test_targets' in token for token in task['command'])
        assert not any('data/views/sourceqa/scaffold' in token or 'data/splits/sourceqa/scaffold' in token for token in task['command'])
    assert len(property_names('scaffold',registry['protocols']['scaffold']))==7


def test_actual_zero_shot_inputs_share_bytes_but_preserve_fold_manifest_hashes(tmp_path,monkeypatch):
    monkeypatch.chdir(ROOT)
    if not Path('data/views/strictqa/scaffold/test_queries.jsonl').exists():
        pytest.skip('Run the public-data reconstruction before the real-input hash audit')
    pairs=[]
    for protocol,path in [('identity','configs/development_queue_identity_sourceqa_v2.json'),
                          ('scaffold','configs/development_queue_scaffold_strictqa_v2.json')]:
        plan=json.loads(Path(path).read_text())
        task=next(t for t in plan['tasks'] if t['id'].endswith('_qwen_base_17_none_val'))
        spec=prediction_spec(task['command'],protocol,tmp_path)
        manifest='data/splits/'+('sourceqa' if protocol=='identity' else 'strictqa')+'/'+protocol+'.csv'
        assert spec['fingerprint']['manifest_sha256']==sha256_file(manifest)
        assert spec['queries']==f'data/views/{"sourceqa" if protocol=="identity" else "strictqa"}/{protocol}/test_queries.jsonl'
        transformed=prediction_spec(task['command'],protocol,tmp_path,True)
        assert ('strictqa' in transformed['queries'])==(protocol=='scaffold')
        pairs.append(spec)
    assert pairs[0]['fingerprint']['manifest_sha256']!=pairs[1]['fingerprint']['manifest_sha256']
    assert reusable_zero_shot(*pairs)


def test_final_graph_handles_distinct_banks_and_model_names():
    data=releases()
    data['scaffold']=json.loads(json.dumps(data['scaffold']).replace('sourceqa','strictqa'))
    data['scaffold'].update(runs_root='artifacts/runs_strictqa',measurement_path='data/processed/strictqa/measurements_scaffold.parquet',
                            consistency_runs_root='artifacts/consistency/strictqa')
    tasks=assemble(data)
    assert len(tasks)==294
    for task in tasks:
        if task['protocol']!='scaffold':continue
        joined=' '.join(task.get('command',[]))
        assert 'data/views/sourceqa/scaffold' not in joined
        assert 'data/splits/sourceqa/scaffold' not in joined
        assert 'sourceqa_scaffold_qwen' not in joined
    comparison=next(t for t in tasks if t['id']=='scaffold_primary_comparison')
    assert all('artifacts/runs_strictqa' in x for x in comparison['command'] if 'evaluation_test_view' in x)


def test_pipeline_recognizes_exact_live_queue_argv_without_restarting(tmp_path,monkeypatch):
    import time
    import solvent_select.pipeline as pipeline
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    command=[sys.executable,'-c',"import time; from pathlib import Path; Path('UNIT_READY').write_text('ready'); time.sleep(30)",'scripts/run_development_queue.py','--plan','UNIT_plan.json']
    child=subprocess.Popen(command,cwd=tmp_path)
    try:
        deadline=time.monotonic()+3
        while not (tmp_path/'UNIT_READY').exists() and time.monotonic()<deadline:time.sleep(.01)
        assert (tmp_path/'UNIT_READY').exists()
        assert live_queue(child.pid,'UNIT_plan.json')
        assert not live_queue(child.pid,'other_plan.json')
    finally:
        child.terminate();child.wait(timeout=5)


def test_missing_queue_state_is_pending_not_a_false_completion(tmp_path):
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'input_sha256':{},'tasks':[{'id':'unit','command':[],'outputs':[]}]}))
    audit=queue_evidence(plan,tmp_path/'not_started.json')
    assert audit['pending']==[{'task':'unit','status':'not_started'}]
    assert audit['state_sha256'] is None and not audit['errors']


def test_actual_completed_adapter_has_exact_primary_training_boundary(monkeypatch):
    monkeypatch.chdir(ROOT)
    if not Path('artifacts/runs_sourceqa/sourceqa_identity_qwen_sft_17/run.json').exists():
        pytest.skip('Actual completed adapter audit runs after training; never replace it with synthetic results')
    config=load_registry()['protocols']['identity']
    metadata=json.loads(Path('artifacts/runs_sourceqa/sourceqa_identity_qwen_sft_17/run.json').read_text())
    expected=verify_adapter_boundary(metadata,config,metadata['config'])
    assert len(expected)==5
    wrong=load_registry()['protocols']['scaffold']
    with pytest.raises(ValueError,match='protocol boundary'):
        verify_adapter_boundary(metadata,wrong,metadata['config'])

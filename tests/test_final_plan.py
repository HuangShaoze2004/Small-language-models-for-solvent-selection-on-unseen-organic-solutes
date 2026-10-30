"""Synthetic task-graph fixtures; never scientific predictions or test scores."""
import copy
import json
from pathlib import Path

import pytest

from solvent_select.final_plan import assemble, prepare, reusable_zero_shot
import solvent_select.final_runner as runner
from solvent_select.views import sha256_file


def releases():
    """Use registered method identities, with intentionally fake unit-test hashes."""
    root=Path(__file__).parents[1]
    plan=json.loads((root/'configs/development_queue_sourceqa.json').read_text())
    analysis=json.loads((root/'configs/final_analysis_plan.json').read_text())
    result={}
    for protocol in ['identity','scaffold']:
        specs=[]
        for task in plan['tasks']:
            cmd=task['command']
            if 'solvent_select.llm' not in cmd or 'predict' not in cmd or cmd[cmd.index('--protocol')+1]!=protocol:
                continue
            mode=cmd[cmd.index('--retrieval-mode')+1]
            seed=int(cmd[cmd.index('--seed')+1])
            adapter=cmd[cmd.index('--adapter')+1] if '--adapter' in cmd else None
            for consistency in [False,True] if mode in ['none','text','structure','condition'] else [False]:
                fingerprint={'prediction_kind':'llm','config':{'seed':seed,'retrieval_mode':mode},
                             'adapter_sha256':adapter,'manifest_sha256':protocol,
                             'query_sha256':'transformed' if consistency else 'full'}
                specs.append({'validation_command':cmd,'adapter':adapter,'consistency':consistency,
                              'prediction_kind':'llm','retrieval_mode':mode,'fingerprint':fingerprint,
                              'sha256':'UNIT-'+task['id']+str(consistency),'queries':'transformed_queries.jsonl' if consistency else 'test_queries.jsonl'})
        for mode in analysis['direct_evidence_modes']:
            for consistency in [False,True]:
                cmd=['python','-m','solvent_select.llm','evidence','--retrieval-mode',mode]
                specs.append({'validation_command':cmd,'adapter':None,'consistency':consistency,
                              'prediction_kind':'evidence','retrieval_mode':mode,
                              'fingerprint':{'prediction_kind':'evidence','config':{'seed':17,'retrieval_mode':mode}},
                              'sha256':'UNIT-'+mode+str(consistency),'queries':'test_queries.jsonl'})
        models=[]
        controls=[]
        for name in analysis['property_methods'][protocol]:
            method='chemprop' if '_chemprop_' in name else 'lightgbm' if '_lgbm_' in name else 'nearest'
            run='artifacts/runs_sourceqa/'+name
            models.append({'name':name,'run':run,'method':method,'seed':17})
            controls.append({'id':name,'kind':'property','model_run':run,'canonicalize_query':False})
            if method=='chemprop':
                controls.append({'id':name+'_canonical_control','kind':'property','model_run':run,'canonicalize_query':True})
        controls.append({'id':f'sourceqa_{protocol}_evidence_text_canonical_control','kind':'evidence','canonicalize_query':True})
        result[protocol]={'_path':f'UNIT/{protocol}.json','query_path':f'data/views/sourceqa/{protocol}/test_queries.jsonl',
                          'target_path':f'data/views/sourceqa/{protocol}/test_targets.jsonl',
                          'split_path':f'data/splits/sourceqa/{protocol}.csv',
                          'consistency_manifest_path':f'data/consistency/sourceqa/{protocol}_test/manifest.json',
                          'property_models':models,'cpu_consistency_controls':controls,'prediction_specifications':specs}
    return result


def test_all_registered_methods_controls_and_error_analyses_survive_plan_building():
    tasks=assemble(releases())
    assert len(tasks)==294
    assert len({t['id'] for t in tasks})==294
    assert sum(t['action']=='reuse_zero_shot' for t in tasks)==2
    assert sum('property_prediction' in t for t in tasks)==14
    assert sum(t['id'].endswith('_error_analysis') for t in tasks)==62
    assert sum('solvent_select.final_controls' in t.get('command',[]) for t in tasks)==22
    assert sum(t['id'].endswith('_primary_comparison') for t in tasks)==2
    for task in tasks:
        if 'prediction_specification' in task:
            assert '--targets' not in task.get('command',[])
            assert task['prediction_specification']['sha256'].startswith('UNIT-')
        if task['action']=='command' and 'solvent_select.llm' in task['command']:
            assert '--allow-test' in task['command'] and '--freeze-manifest' in task['command']


def test_reuse_allows_only_identical_unadapted_zero_shot_inputs():
    a={'fingerprint':{'prediction_kind':'llm','adapter_sha256':None,'config':{'retrieval_mode':'none'},'query_sha256':'same','manifest_sha256':'a'}}
    b=copy.deepcopy(a);b['fingerprint']['manifest_sha256']='b'
    assert reusable_zero_shot(a,b)
    b['fingerprint']['query_sha256']='different'
    assert not reusable_zero_shot(a,b)
    b=copy.deepcopy(a);b['fingerprint']['adapter_sha256']='trained'
    assert not reusable_zero_shot(a,b)
    b=copy.deepcopy(a);b['fingerprint']['config']['retrieval_mode']='condition'
    assert not reusable_zero_shot(a,b)


def test_preparation_refuses_candidate_before_creating_any_final_plan(tmp_path):
    candidate=tmp_path/'candidate.json';candidate.write_text('{"status":"CANDIDATE_NOT_FROZEN"}')
    output=tmp_path/'final_plan.json'
    with pytest.raises(ValueError,match='actual FROZEN'):
        prepare([candidate,candidate],output)
    assert not output.exists()


def test_executor_rebuilds_exact_graph_and_rejects_edited_or_removed_methods(tmp_path,monkeypatch):
    data=releases();hashes={}
    for protocol,release in data.items():
        path=tmp_path/(protocol+'.json');release['_path']=str(path)
        path.write_text(json.dumps(release));hashes[str(path)]=sha256_file(path)
    # This test covers task-graph locking, not release verification; it never
    # creates a scientific FROZEN manifest or opens actual held-out data.
    monkeypatch.setattr(runner,'verify_release',lambda path:{'protocol':Path(path).stem})
    plan={'status':'FROZEN_FINAL_EXECUTION_PLAN','partition':'test','release_sha256':hashes,
          'tasks':assemble(data),'code_sha256':sha256_file(Path(runner.__file__).with_name('final_plan.py'))}
    runner.verify_plan(plan)
    altered=copy.deepcopy(plan);altered['tasks'].pop()
    with pytest.raises(ValueError,match='task graph differs'):
        runner.verify_plan(altered)
    altered=copy.deepcopy(plan);altered['tasks'][0]['command']+=['--unregistered-option']
    with pytest.raises(ValueError,match='task graph differs'):
        runner.verify_plan(altered)

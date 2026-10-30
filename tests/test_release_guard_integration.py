"""Private synthetic release-schema fixtures; no model execution or real targets.

All fixture files live under pytest's temporary directory. The fake weight bytes
and intentionally non-JSON target bytes are never scientific result artifacts.
"""
import copy
import json
from pathlib import Path

import pytest

from solvent_select.freeze import test_prediction_spec as prediction_spec
from solvent_select.llm import MODEL_ID,MODEL_REVISION
from solvent_select.release_guard import verify_release
from solvent_select.views import sha256_file


def fixture_release(tmp_path,monkeypatch,protocol='identity',with_registry=False):
    monkeypatch.chdir(tmp_path)
    tag='strictqa' if protocol=='scaffold' else 'sourceqa'
    def put(path,content):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content if isinstance(content,str) else json.dumps(content))
        return str(path)
    manifest=put(f'data/splits/{tag}/{protocol}.csv','solute_id,split\ns1,test\n')
    query={'query_id':'UNIT_q','solute_id':'s1','split':'test','task':'pair'}
    queries=put(f'data/views/{tag}/{protocol}/test_queries.jsonl',json.dumps(query)+'\n')
    targets=put('UNIT_opaque_targets.jsonl','THIS IS NOT JSON: GUARD MUST ONLY HASH IT')
    transformed=put(f'data/consistency/{tag}/{protocol}_test/transformed_queries.jsonl',json.dumps({**query,'query_id':'UNIT_t'})+'\n')
    base=put(f'data/consistency/{tag}/{protocol}_test/base_queries.jsonl',json.dumps(query)+'\n')
    mapping=put(f'data/consistency/{tag}/{protocol}_test/mapping.jsonl','{}\n')
    consistency=put(f'data/consistency/{tag}/{protocol}_test/manifest.json',{
        'split':'test','split_sha256':sha256_file(manifest),
        'files':{Path(p).name:sha256_file(p) for p in [base,transformed,mapping]}})
    config=put('UNIT_config.yaml','retrieval_mode: none\n')
    method=f'{tag}_{protocol}_qwen_base_17_none_val'
    command=['python','wrapper','--run-id',method,'--',
             'python','-m','solvent_select.llm','predict','--config',config,'--retrieval-mode','none',
             '--manifest',manifest,'--queries',queries]
    folder=tmp_path/'UNIT_release';folder.mkdir()
    specs=[prediction_spec(command,protocol,folder,c) for c in [False,True]]
    analysis=put('UNIT_analysis.json',{'status':'REGISTERED_BEFORE_TEST',
        'llm_validation_tasks_to_carry_to_test':[method],
        'property_methods':{protocol:['UNIT_property']},'direct_evidence_modes':[]})
    files={field:put('UNIT_'+field+'.json',{}) for field in ['decision','pretest_review']}
    fake_weights=put('UNIT_weights/model.bin','UNIT TEST BYTES, NOT MODEL WEIGHTS')
    snapshot=put('UNIT_snapshot.json',{'model':MODEL_ID,'revision':MODEL_REVISION,
        'path':str(tmp_path/'UNIT_weights'),'files':[{'name':'model.bin','sha256':sha256_file(fake_weights)}]})
    artifact=put('UNIT_artifact.txt','UNIT fixture')
    error_code=put('src/solvent_select/error_analysis.py','# UNIT fixture')
    error_config=put('configs/error_analysis.yaml','unit_fixture: true\n')
    record={'status':'FROZEN','protocol_path':analysis,'protocol_sha256':sha256_file(analysis),
        'query_path':queries,'query_sha256':sha256_file(queries),'query_count':1,
        'target_path':targets,'target_sha256':sha256_file(targets),
        'split_path':manifest,'split_sha256':sha256_file(manifest),
        'consistency_manifest_path':consistency,'consistency_manifest_sha256':sha256_file(consistency),
        'foundation_model_snapshot_path':snapshot,'foundation_model_snapshot_sha256':sha256_file(snapshot),
        'prediction_specifications':specs,'prediction_fingerprints':[s['sha256'] for s in specs],
        'property_models':[{'name':'UNIT_property','run':'UNIT_property','method':'nearest'}],
        'cpu_consistency_controls':[{'id':'UNIT_property','kind':'property','model_run':'UNIT_property','canonicalize_query':False},
            {'id':'UNIT_bm25','kind':'evidence','retrieval_mode':'text','canonicalize_query':True}],
        'queue_receipts':[],'error_analysis_code_sha256':sha256_file(error_code),'error_analysis_config_sha256':sha256_file(error_config)}
    for field,path in files.items():record[field+'_path']=path;record[field+'_sha256']=sha256_file(path)
    for field in ['property_assets','code_sha256','scripts_sha256','environment_locks','validation_consistency_sha256']:
        record[field]={artifact:sha256_file(artifact)}
    if with_registry:
        registry=put('UNIT_registry.json',{'status':'REGISTERED_VALIDATION_ONLY_DUAL_QA','protocols':{
            'identity':{'data_tag':'sourceqa','run_prefix':'sourceqa_'},
            'scaffold':{'data_tag':'strictqa','run_prefix':'strictqa_'}}})
        registered=json.loads(Path(registry).read_text())
        registered['protocols'][protocol].update(split=manifest,measurements='UNIT_measurement_metadata',runs_root='UNIT_runs')
        Path(registry).write_text(json.dumps(registered))
        record.update(registry_path=registry,registry_sha256=sha256_file(registry),
                      measurement_path='UNIT_measurement_metadata',runs_root='UNIT_runs',
                      adapter_assets={artifact:sha256_file(artifact)})
    path=folder/f'{protocol}.json';path.write_text(json.dumps(record))
    return path,record


def test_full_guard_verifies_files_and_fingerprints_without_parsing_target_bytes(tmp_path,monkeypatch):
    path,_=fixture_release(tmp_path,monkeypatch)
    result=verify_release(path)
    assert result['status']=='VERIFIED_FROZEN_RELEASE'
    assert result['test_target_values_read'] is False and result['models_executed']==0


@pytest.mark.parametrize('change',['omit_transform','omit_model','change_config','change_weight','change_error_config'])
def test_full_guard_detects_changes_and_omissions(tmp_path,monkeypatch,change):
    path,record=fixture_release(tmp_path,monkeypatch)
    if change=='omit_transform':
        record['prediction_specifications'].pop();record['prediction_fingerprints'].pop()
    elif change=='omit_model':record['property_models']=[]
    elif change=='change_config':Path('UNIT_config.yaml').write_text('retrieval_mode: none\nmax_new_tokens: 10\n')
    elif change=='change_weight':Path('UNIT_weights/model.bin').write_text('changed UNIT bytes')
    else:Path('configs/error_analysis.yaml').write_text('changed: true\n')
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):verify_release(path)


@pytest.mark.parametrize('protocol',['identity','scaffold'])
def test_full_guard_uses_the_registered_protocol_namespace(tmp_path,monkeypatch,protocol):
    path,_=fixture_release(tmp_path,monkeypatch,protocol=protocol,with_registry=True)
    result=verify_release(path)
    assert result['protocol']==protocol and result['test_target_values_read'] is False


def test_strict_guard_rejects_a_legacy_primary_namespace_in_the_analysis(tmp_path,monkeypatch):
    path,record=fixture_release(tmp_path,monkeypatch,protocol='scaffold',with_registry=True)
    analysis=Path(record['protocol_path']);value=json.loads(analysis.read_text())
    value['llm_validation_tasks_to_carry_to_test']=['sourceqa_scaffold_qwen_base_17_none_val']
    analysis.write_text(json.dumps(value));record['protocol_sha256']=sha256_file(analysis)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='LM methods differ'):verify_release(path)

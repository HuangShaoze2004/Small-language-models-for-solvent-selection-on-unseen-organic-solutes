"""Synthetic taint and candidate-block fixtures, never model-quality results."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('query_isolation',Path(__file__).parents[1]/'scripts/audit_query_label_isolation.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def metadata(n=17):
    return pd.DataFrame([{'record_id':f'r{i}','solute_id':'s','solute_smiles':'CC(=O)O',
        'solvent_id':f'v{i}','solvent_smiles':'C'*(i+1),'source_doi':'10.1234/example',
        'temperature_k':298.15,'leakage_group':'g','logx':-float(i)} for i in range(n)])


def test_metadata_questions_are_independent_of_targets_and_row_order():
    frame=metadata()
    expected=module.expected_questions(frame,{'s':'train'})
    # 17 candidates form balanced6/6/5blocks:40pairs and3rankings.
    assert len(expected)==43
    changed=frame.sample(frac=1,random_state=17).copy()
    changed['logx']=1000.
    changed['choice']='FORBIDDEN_LABEL'
    changed['ranking']='FORBIDDEN_RANK'
    assert module.expected_questions(changed,{'s':'train'})==expected


def test_query_identity_does_not_encode_partition_or_correct_answer():
    first=module.expected_questions(metadata(2),{'s':'train'})
    second=module.expected_questions(metadata(2),{'s':'val'})
    assert set(first)==set(second)
    assert len(first)==1
    assert next(iter(first.values()))[0]=='train'
    assert next(iter(second.values()))[0]=='val'


def query():
    return {'query_id':'q','solute_id':'s','task':'pair','solute_smiles':'CC(=O)O',
            'temperature_k':298.15,'split':'val','candidates':[
                {'id':'A','smiles':'O','record_id':'a'},{'id':'B','smiles':'CCO','record_id':'b'}]}


def test_prompt_sentinel_audit_passes_actual_allowlist_and_detects_a_leaking_serializer(monkeypatch):
    module.assert_prompt_isolation(query())
    original=module.build_messages
    def leaking(value):
        result=original(value)
        if 'choice' in value:result.append({'role':'user','content':value['choice']})
        return result
    monkeypatch.setattr(module,'build_messages',leaking)
    with pytest.raises(ValueError,match='forbidden target/source field'):
        module.assert_prompt_isolation(query())

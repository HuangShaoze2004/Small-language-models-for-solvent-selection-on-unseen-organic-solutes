"""Synthetic diagnostics fixtures; no chemical performance measurements."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('input_differences',Path(__file__).parents[1]/'scripts/analyze_retrieval_input_differences.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_payload_identity_excludes_non_input_scores_and_preserves_order():
    first={'evidence':[{'card_id':'c','record_ids':['a','b'],'score':.3}]}
    score_change={'evidence':[{'card_id':'c','record_ids':['a','b'],'score':.9}]}
    reorder={'evidence':[{'card_id':'c','record_ids':['b','a'],'score':.3}]}
    assert module.payload_identity(first)==module.payload_identity(score_change)
    assert module.payload_identity(first)!=module.payload_identity(reorder)


def test_conditional_macro_balances_conditions_then_solutes():
    frame=pd.DataFrame([{'solute_id':sid,'condition_id':cid,'value':value}
                        for sid,cid,value in [('s','c',0),('s','c',1),('s','d',1),('t','c',0)]])
    assert module.balanced(frame,['value'])=={'value':.375}
    assert module.balanced(frame.iloc[:0],['value'])=={}


def test_alignment_refuses_incomplete_or_duplicate_cohorts(tmp_path):
    path=tmp_path/'predictions.jsonl';queries=[{'query_id':'a'},{'query_id':'b'}]
    path.write_text('{"query_id":"a"}\n{"query_id":"a"}\n')
    with pytest.raises(ValueError,match='Incomplete or duplicate'):
        module.aligned(path,queries)
    path.write_text('{"query_id":"a"}\n{"query_id":"b"}\n')
    assert set(module.aligned(path,queries))=={'a','b'}

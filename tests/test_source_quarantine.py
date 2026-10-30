import json
import pandas as pd
import pytest

from solvent_select.source_quarantine import filter_tasks, retained_manifest


def fixture(tmp_path):
    q={'query_id':'q','task':'pair','candidates':[{'record_id':'a'},{'record_id':'b'}]}
    t={'query_id':'q','choice':'A','values':{'A':-1.,'B':-2.}}
    sourceq,sourcet=tmp_path/'q.jsonl',tmp_path/'t.jsonl'
    sourceq.write_text(json.dumps(q)+'\n');sourcet.write_text(json.dumps(t)+'\n')
    return sourceq,sourcet,tmp_path/'qo.jsonl',tmp_path/'to.jsonl'


def test_retained_tasks_and_targets_are_byte_identical(tmp_path):
    paths=fixture(tmp_path)
    result=filter_tasks(*paths,{'unrelated'})
    assert result['retained_by_task']=={'pair':1}
    assert paths[0].read_bytes()==paths[2].read_bytes()
    assert paths[1].read_bytes()==paths[3].read_bytes()


def test_whole_source_removed_without_new_combinations(tmp_path):
    paths=fixture(tmp_path)
    result=filter_tasks(*paths,{'a','b'})
    assert result['excluded_queries']==1
    assert paths[2].read_bytes()==paths[3].read_bytes()==b''
    with pytest.raises(ValueError,match='candidate subset'):
        filter_tasks(*paths,{'a'})


def test_dropping_training_identity_never_reshuffles_other_roles():
    m=pd.DataFrame({'solute_id':['a','b','c','d'],'split':['train','train','val','test']})
    result=retained_manifest(m,{'b','c','d'})
    assert dict(zip(result.solute_id,result.split))=={'b':'train','c':'val','d':'test'}

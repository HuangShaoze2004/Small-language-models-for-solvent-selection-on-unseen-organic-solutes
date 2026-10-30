import pytest
from solvent_select.evaluate import evaluate_queries, paired_bootstrap

def pair(qid='q', solute='s', condition='c'):
    return dict(query_id=qid, solute_id=solute, condition_id=condition, task='pair',
                candidates=[dict(id='A', smiles='CO'),dict(id='B', smiles='CCO')])

def test_missing_prediction_is_counted_and_receives_worst_regret():
    m, detail, sol = evaluate_queries([pair()], [dict(query_id='q',choice='A',values={'A':-1.,'B':-3.})], [])
    assert m['by_task']['pair']['pair_accuracy'] == 0
    assert m['by_task']['pair']['regret'] == 2
    assert m['by_task']['pair']['covered'] == 0

def test_hierarchical_macro_not_row_micro():
    qs = [pair('a1'), pair('a2'), pair('b', condition='d'),pair('c', solute='t')]
    gold = [dict(query_id=q['query_id'],choice='A',values={'A':-1.,'B':-3.}) for q in qs]
    pred = [dict(query_id=q['query_id'],choice='A' if q['query_id'] in {'a1','a2'} else 'B') for q in qs]
    metrics, _, _ = evaluate_queries(qs,gold,pred)
    assert metrics['by_task']['pair']['pair_accuracy'] == .25

def test_ranking_near_best_counts_hit_but_has_nonzero_regret():
    q = dict(query_id='q',solute_id='s',condition_id='c',task='rank',candidates=[dict(id=k,smiles=k) for k in ['X','Y','Z']])
    m,_,_=evaluate_queries([q],[dict(query_id='q',values={'X':-1,'Y':-1.08,'Z':-1.16})],[dict(query_id='q',ranking=['Y','X','Z'])])
    assert m['by_task']['rank']['top1'] == 1
    assert m['by_task']['rank']['regret'] == pytest.approx(.08)

def test_invalid_duplicate_ranking_not_silently_repaired():
    q = dict(query_id='q',solute_id='s',condition_id='c',task='rank',candidates=[dict(id=k,smiles=k) for k in ['X','Y','Z']])
    m,_,_=evaluate_queries([q],[dict(query_id='q',values={'X':-1,'Y':-2,'Z':-3})],[dict(query_id='q',ranking=['X','X','Z'])])
    assert m['by_task']['rank']['covered'] == 0
    assert m['by_task']['rank']['regret'] == 2

def test_bootstrap_pairs_solutes_and_rejects_misalignment():
    _,_,a=evaluate_queries([pair()],[dict(query_id='q',choice='A',values={'A':-1,'B':-2})],[dict(query_id='q',choice='A')])
    result=paired_bootstrap(a,a,'pair_accuracy','pair')
    assert result['ci95'] == [0.,0.]
    b=a.copy(); b['solute_id']='different'
    with pytest.raises(ValueError): paired_bootstrap(a,b,'pair_accuracy','pair')

def test_unhashable_model_output_is_invalid_not_a_crash():
    m,_,_=evaluate_queries([pair()],[dict(query_id='q',choice='A',values={'A':-1,'B':-2})],[dict(query_id='q',choice=['A'])])
    assert m['by_task']['pair']['covered'] == 0

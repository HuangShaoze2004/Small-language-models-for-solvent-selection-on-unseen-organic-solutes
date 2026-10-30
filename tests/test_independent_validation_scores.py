"""Hand-computed synthetic fixtures for the independent grader, never efficacy data."""
import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('independent_scores',Path(__file__).parents[1]/'scripts/audit_validation_scores.py')
grader=importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)


def query(task='pair',sid='s',condition='c'):
    ids=['A','B'] if task=='pair' else ['A','B','C']
    return {'task':task,'solute_id':sid,'condition_id':condition,
            'candidates':[{'id':cid,'smiles':smiles} for cid,smiles in zip(ids,['O','OCC','CC'])]}


def test_tau_b_counts_truth_ties_and_keeps_invalid_penalty():
    q=query('rank');truth={'A':1.,'B':1.,'C':0.}
    first=grader.utility(q,truth,{'ranking':['A','B','C']})
    reverse=grader.utility(q,truth,{'ranking':['C','B','A']})
    assert first['rank_tau']==pytest.approx(2/math.sqrt(6))
    assert first['top1']==1 and first['regret']==0
    assert reverse['rank_tau']==pytest.approx(-2/math.sqrt(6))
    assert reverse['top1']==0 and reverse['regret']==1
    invalid=grader.utility(q,truth,{'ranking':['A','A','B']})
    assert invalid['rank_tau']==-1 and invalid['top1']==0
    assert invalid['covered']==0 and invalid['regret']==1
    constant=grader.utility(q,{'A':1.,'B':1.,'C':1.},{'ranking':['A','B','C']})
    assert math.isnan(constant['rank_tau']) and constant['top1']==1


def test_near_tie_boundary_and_chemical_selection_under_option_swap():
    assert grader.choice({'A':.1,'B':0})=='TIE'
    assert grader.choice({'A':.10001,'B':0})=='A'
    assert grader.choice({'A':0,'B':.10001})=='B'
    q=query();original=grader.utility(q,{'A':0.,'B':.08},{'choice':'TIE'})
    swapped=query();swapped['candidates']=[{'id':'A','smiles':'CCO'},{'id':'B','smiles':'O'}]
    actual=grader.utility(swapped,{'A':.08,'B':0.},{'choice':'TIE'})
    assert original['pair_accuracy']==actual['pair_accuracy']==1
    assert original['regret']==actual['regret']==0
    invalid=grader.utility(q,{'A':0.,'B':1.},None)
    assert invalid['covered']==invalid['pair_accuracy']==0 and invalid['regret']==1
    assert math.isnan(original['nontie_pair_accuracy'])


def test_physical_condition_and_solute_weighting_not_question_average():
    queries=[query(condition='c1'),query(condition='c1'),query(condition='c2'),query(sid='other')]
    rows=[{'pair_accuracy':v} for v in [0.,1.,1.,0.]]
    solutes,macro=grader.aggregate(queries,rows)
    assert solutes[('s','pair','pair_accuracy')]==.75
    assert macro[('pair','pair_accuracy')]==.375


@pytest.mark.parametrize('text', ['{"choice":"A","reason":"x"}','{"choice":[]}',
                                   '{"choice":"a"}','["A"]','not json'])
def test_raw_schema_rejects_extra_fields_and_malformed_choice(text):
    assert grader.decode(text,query()) is None


def test_raw_schema_accepts_fence_and_requires_full_unique_ranking():
    assert grader.decode('```json\n{"choice":"A"}\n```',query())=={'choice':'A'}
    assert grader.decode('{"ranking":["C","B","A"]}',query('rank'))=={'ranking':['C','B','A']}
    for text in ['{"ranking":["A","A","B"]}','{"ranking":["A","B"]}',
                 '{"ranking":["A","B",1]}','{"ranking":["A","B","X"]}']:
        assert grader.decode(text,query('rank')) is None


def test_solute_cohort_cannot_silently_omit_duplicate_or_add_groups():
    queries=[query()];computed=[grader.utility(queries[0],{'A':1.,'B':0.},{'choice':'A'})]
    correct=pd.DataFrame([dict(solute_id='s',task='pair',**computed[0])]).set_index(['solute_id','task'])
    assert grader.check_solute_scores(queries,computed,correct)[('pair','pair_accuracy')]==1
    for corrupted in [correct.iloc[:0],pd.concat([correct,correct]),correct.rename(index={'s':'wrong'},level='solute_id')]:
        with pytest.raises(ValueError,match='Wrong stored solute score cohort'):
            grader.check_solute_scores(queries,computed,corrupted)

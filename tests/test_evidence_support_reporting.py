"""Hand-computed evidence support fixtures; no held-out chemistry results."""
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('resource_report',Path(__file__).parents[1]/'scripts/summarize_validation_resources.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_valid_ranking_with_one_supported_pair_does_not_have_complete_evidence():
    query={'query_id':'q','solute_id':'s','condition_id':'c','task':'rank',
           'candidates':[{'id':c} for c in ['A','B','C']]}
    prediction={'prediction':{'ranking':['A','B','C']},'pair_coverage':1/3,
                'pairwise_evidence':[{'a':'A','b':'B','support':2},{'a':'A','b':'C','support':0},
                                     {'a':'B','b':'C','support':0}]}
    row=module.support_row(query,prediction)
    assert row['valid_output']==1
    assert row['supported_pair_fraction']==1/3
    assert row['all_candidate_pairs_supported']==0
    assert row['no_candidate_pair_supported']==0


def test_unsupported_tie_is_valid_output_with_zero_evidence_support():
    query={'query_id':'q','solute_id':'s','condition_id':'c','task':'pair','candidates':[{'id':'A'},{'id':'B'}]}
    prediction={'prediction':{'choice':'TIE'},'pair_coverage':0.,
                'pairwise_evidence':[{'a':'A','b':'B','support':0}]}
    row=module.support_row(query,prediction)
    assert row['valid_output']==1
    assert row['supported_pair_fraction']==0
    assert row['unsupported_fallback_tie']==1
    prediction['pair_coverage']=1.
    with pytest.raises(ValueError,match='coverage disagrees'):
        module.support_row(query,prediction)


def test_malformed_choice_remains_in_validity_denominator():
    query={'query_id':'q','solute_id':'s','condition_id':'c','task':'pair','candidates':[{'id':'A'},{'id':'B'}]}
    prediction={'prediction':{'choice':['TIE']},'pair_coverage':0.,
                'pairwise_evidence':[{'a':'A','b':'B','support':0}]}
    row=module.support_row(query,prediction)
    assert row['valid_output']==0
    assert row['no_candidate_pair_supported']==1
    assert row['unsupported_fallback_tie']==0

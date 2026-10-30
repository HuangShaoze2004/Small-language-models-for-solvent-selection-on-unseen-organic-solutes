"""Equivalent solvent notation must not change a chemical matching operation."""
import copy

import pytest

from solvent_select.retrieval import canonical_smiles, direct_evidence_prediction
from test_retrieval import card, index, query


@pytest.mark.parametrize('mode', ['structure', 'condition', 'no_temperature', 'no_solvent'])
def test_equivalent_solvent_keeps_retrieval_scores_and_support(mode):
    original = query()
    alternate = copy.deepcopy(original)
    alternate['candidates'][1]['smiles'] = 'OCC'
    retrieval = index([card('a'), card('b', smiles='CCCC', group='other')])
    assert retrieval._scores(original, mode).tolist() == retrieval._scores(alternate, mode).tolist()
    first = retrieval.retrieve(original, mode)
    second = retrieval.retrieve(alternate, mode)
    assert first == second
    assert direct_evidence_prediction(original, first) == direct_evidence_prediction(alternate, second)
    assert direct_evidence_prediction(alternate, second)['pair_coverage'] == 1


def test_matching_preserves_stereo_charge_isotopes_and_salts():
    assert canonical_smiles('OCC') == canonical_smiles('CCO')
    assert canonical_smiles('C[C@H](O)Cl') != canonical_smiles('C[C@@H](O)Cl')
    assert canonical_smiles('CC(=O)[O-].[Na+]') != canonical_smiles('CC(=O)O')
    assert canonical_smiles('[13CH3]CO') != canonical_smiles('CCO')


def test_coverage_refresh_uses_identity_after_prompt_trimming():
    from solvent_select.llm import _refresh_evidence_coverage
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text)
    alternate = query()
    alternate['candidates'][1]['smiles'] = 'OCC'
    evidence = index([card()]).retrieve(alternate)
    _refresh_evidence_coverage(evidence, alternate, Tokenizer())
    assert evidence.candidate_coverage == {'water': 1, 'ethanol': 1}


@pytest.mark.parametrize('value', [[], {}, ['A'], None, 1])
def test_malformed_pair_choice_counts_invalid_without_crashing(value):
    import json
    from solvent_select.llm import parse_prediction
    prediction, error = parse_prediction(json.dumps({'choice': value}), query())
    assert prediction is None and error == 'invalid_choice'

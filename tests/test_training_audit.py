"""Synthetic isolation fixtures; not scientific training/evaluation results."""
import copy

import pandas as pd
import pytest

from solvent_select.training_audit import audit_rows


def fixture():
    query = {'query_id': 'q', 'solute_id': 'query', 'leakage_group': 'query_parent'}
    train = pd.DataFrame([{'record_id': 'train_record', 'solute_id': 'other_solute',
                           'leakage_group': 'other_parent', 'source_doi': 'fixed_training_source'}])
    row = {'query_id': 'q', 'retrieval_mode': 'condition', 'input_tokens': 500, 'target_tokens': 7,
           'evidence_tokens': 200, 'evidence': [{'solute_id': 'other_solute', 'leakage_group': 'other_parent',
                    'source_doi': 'fixed_training_source', 'record_ids': ['train_record']}]}
    config = {'retrieval_mode': 'condition', 'max_sequence_length': 3072, 'max_evidence_tokens': 1536}
    return query, train, row, config


def test_actual_record_provenance_is_required_not_only_the_cards_declared_identity():
    query, train, row, config = fixture()
    assert audit_rows([query], [row], train, config)['cards_checked'] == 1
    changed = copy.deepcopy(row)
    changed['evidence'][0]['source_doi'] = 'a_false_declared_source'
    with pytest.raises(ValueError, match='provenance disagrees'):
        audit_rows([query], [changed], train, config)


@pytest.mark.parametrize('field,value', [('solute_id', 'query'), ('leakage_group', 'query_parent')])
def test_self_identity_and_different_identity_in_same_group_are_both_forbidden(field, value):
    query, train, row, config = fixture()
    row['evidence'][0][field] = value
    with pytest.raises(ValueError, match='own solute or isolation group'):
        audit_rows([query], [row], train, config)


def test_a_card_cannot_reference_a_validation_measurement_under_a_training_identity():
    query, train, row, config = fixture()
    row['evidence'][0]['record_ids'] = ['heldout_record']
    with pytest.raises(ValueError, match='non-training measurement'):
        audit_rows([query], [row], train, config)


def test_saved_query_ids_and_order_must_match_the_sampled_training_inputs():
    query, train, row, config = fixture()
    row['query_id'] = 'unrelated'
    with pytest.raises(ValueError, match='query identities and order'):
        audit_rows([query], [row], train, config)


def test_mode_and_sequence_budget_cannot_be_overridden_in_saved_evidence():
    query, train, row, config = fixture()
    changed = copy.deepcopy(row)
    changed['retrieval_mode'] = 'text'
    with pytest.raises(ValueError, match='fixed within-solute cycling'):
        audit_rows([query], [changed], train, config)
    row['input_tokens'] = 3071
    with pytest.raises(ValueError, match='fixed context budget'):
        audit_rows([query], [row], train, config)

import copy
import json
from pathlib import Path

import pytest

from solvent_select.consistency import alternate_smiles, build, normalized, score, transformed
from solvent_select.retrieval import canonical_smiles


def query(task='pair', **updates):
    q = {'query_id': 'q-' + task, 'solute_id': 'solute', 'leakage_group': 'group',
         'condition_id': 'condition', 'task': task, 'split': 'val', 'temperature_k': 298.15,
         'solute_smiles': 'C[C@H](O)Cl', 'candidates': [
             {'id': 'A', 'smiles': 'CCO', 'record_id': 'r1'},
             {'id': 'B', 'smiles': 'O', 'record_id': 'r2'}]}
    q.update(updates)
    return q


def test_pair_swap_maps_chemical_choice_and_preserves_records():
    q = query()
    swapped, mapping = transformed(q, 'candidate_order')
    assert swapped['candidates'][0] == {'id': 'A', 'smiles': 'O', 'record_id': 'r2'}
    assert normalized(swapped, {'choice': 'B'}, mapping['candidate_to_original']) == normalized(q, {'choice': 'A'})
    assert normalized(swapped, {'choice': 'TIE'}, mapping['candidate_to_original']) == ('TIE',)
    assert q['candidates'][0]['smiles'] == 'CCO'


def test_rank_reorder_preserves_ids_and_full_ranking():
    q = query('rank')
    swapped, mapping = transformed(q, 'candidate_order')
    assert [c['id'] for c in swapped['candidates']] == ['B', 'A']
    assert normalized(swapped, {'ranking': ['B', 'A']}, mapping['candidate_to_original']) == ('B', 'A')


@pytest.mark.parametrize('smiles', ['C[C@H](O)Cl', 'CC(=O)[O-].[Na+]', '[13CH3]CO', 'c1ccccc1O'])
def test_alternate_smiles_is_reproducible_and_preserves_identity(smiles):
    a, b = alternate_smiles(smiles, 'fixed'), alternate_smiles(smiles, 'fixed')
    assert a == b and canonical_smiles(a) == canonical_smiles(smiles)


def fixture(tmp_path, partition='val'):
    queries = tmp_path / 'queries.jsonl'
    queries.write_text(''.join(json.dumps(query(task, split=partition)) + '\n' for task in ['pair', 'rank']))
    manifest = tmp_path / 'splits.csv'
    manifest.write_text('solute_id,split\nsolute,' + partition + '\n')
    folder = tmp_path / 'view'
    build(queries, manifest, folder, limit=2)
    return folder


def test_no_targets_required_and_invalids_count_as_disagreement(tmp_path):
    folder = fixture(tmp_path)
    base_predictions = tmp_path / 'base.jsonl'
    base_predictions.write_text(json.dumps({'query_id': 'q-pair', 'choice': 'A'}) + '\n' +
                                json.dumps({'query_id': 'q-rank', 'ranking': ['A', 'B']}) + '\n')
    predictions = []
    for line in (folder / 'mapping.jsonl').read_text().splitlines():
        m = json.loads(line)
        is_pair = m['original_query_id'] == 'q-pair'
        p = {'choice': 'B' if m['variant'] == 'candidate_order' else 'A'} if is_pair else {'ranking': ['A', 'B']}
        predictions.append({'query_id': m['query_id'], **p})
    transformed_predictions = tmp_path / 'transformed.jsonl'
    transformed_predictions.write_text(''.join(json.dumps(p) + '\n' for p in predictions))
    result = score(folder, base_predictions, transformed_predictions, tmp_path / 'scores')
    assert all(r['exact_agreement'] == 1 for r in result['by_variant_task'].values())
    predictions[0]['choice'] = []
    transformed_predictions.write_text(''.join(json.dumps(p) + '\n' for p in predictions))
    result = score(folder, base_predictions, transformed_predictions, tmp_path / 'scores2')
    assert result['by_variant_task']['candidate_order/pair']['both_valid'] == 0
    assert result['by_variant_task']['candidate_order/pair']['exact_agreement'] == 0


def test_test_guard_precedes_prediction_loading(tmp_path):
    folder = fixture(tmp_path, 'test')
    with pytest.raises(ValueError, match='frozen manifest'):
        score(folder, tmp_path / 'MISSING1', tmp_path / 'MISSING2', tmp_path / 'scores')


def test_modified_registered_queries_rejected(tmp_path):
    folder = fixture(tmp_path)
    path = folder / 'transformed_queries.jsonl'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='input changed'):
        score(folder, tmp_path / 'MISSING1', tmp_path / 'MISSING2', tmp_path / 'scores')

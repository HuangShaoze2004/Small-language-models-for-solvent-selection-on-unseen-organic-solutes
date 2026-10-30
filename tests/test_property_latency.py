"""Hand-built candidate-interface fixtures, not chemistry experiment results."""
import copy
import json
from pathlib import Path

import pytest

from solvent_select.property_latency import benchmark, candidate_requests, decisions_equal
from solvent_select.views import sha256_file


def test_deduplication_preserves_full_identity_temperature_and_original_queries():
    first = {'query_id': 'pair', 'solute_id': 's', 'solute_smiles': 'C[C@H](O)C(=O)O',
             'temperature_k': 298.15, 'task': 'pair', 'candidates': [
                 {'id': 'A', 'record_id': 'measured_source_1', 'smiles': 'O'},
                 {'id': 'B', 'record_id': 'measured_source_2', 'smiles': 'CCO'}]}
    second = copy.deepcopy(first)
    second['query_id'] = 'another_source'
    second['candidates'][0]['record_id'] = 'different_source_same_public_features'
    hotter = copy.deepcopy(first)
    hotter['query_id'] = 'hotter'
    hotter['temperature_k'] = 308.15
    stereoisomer = copy.deepcopy(first)
    stereoisomer['query_id'] = 'stereoisomer'
    stereoisomer['solute_smiles'] = 'C[C@@H](O)C(=O)O'
    queries = [first, second, hotter, stereoisomer]
    before = copy.deepcopy(queries)
    converted, frame = candidate_requests(queries)
    assert queries == before
    assert len(frame) == 6  # same-source metadata is absent from the property interface
    assert converted[0]['candidates'][0]['record_id'] == converted[1]['candidates'][0]['record_id']
    assert converted[0]['candidates'][0]['record_id'] != converted[2]['candidates'][0]['record_id']
    assert converted[0]['candidates'][0]['record_id'] != converted[3]['candidates'][0]['record_id']


def test_decision_comparison_reports_roundoff_changes_only_if_answers_change():
    a = [{'query_id': 'pair', 'choice': 'A', 'predicted_values': {'A': 1., 'B': .1}},
         {'query_id': 'rank', 'ranking': ['A', 'B']}]
    b = [{'query_id': 'rank', 'ranking': ['A', 'B']},
         {'query_id': 'pair', 'choice': 'A', 'predicted_values': {'A': 1.000001, 'B': .1}}]
    assert decisions_equal(a, b) == 0
    b[0]['ranking'] = ['B', 'A']
    assert decisions_equal(a, b) == 1


def test_test_partition_is_rejected_before_saved_predictions_or_model_are_opened(tmp_path):
    root=tmp_path/'runs'
    run=root/'model'
    run.mkdir(parents=True)
    data=tmp_path/'measurements.parquet'
    data.write_text('unread metadata placeholder')
    split=tmp_path/'identity.csv'
    split.write_text('solute_id,split\ns,test\n')
    views=tmp_path/'views'
    views.mkdir()
    (views/'val_queries.jsonl').write_text(json.dumps({'query_id':'q','solute_id':'s','split':'test'})+'\n')
    (run/'metadata.json').write_text(json.dumps({'status':'completed','measurements':str(data),
        'measurement_sha256':sha256_file(data),'splits':str(split),'split_sha256':sha256_file(split)}))
    spec=tmp_path/'spec.json'
    spec.write_text('{}')
    registry=tmp_path/'registry.json'
    registry.write_text('{}')
    config={'runs_root':str(root),'measurements':str(data),'split':str(split),'views':str(views)}
    with pytest.raises(ValueError,match='restricted to registered validation'):
        benchmark('identity',config,'model',tmp_path/'output',{},spec,registry)
    assert not (tmp_path/'output').exists()

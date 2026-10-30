"""Hand-calculated final-audit fixtures, without production evaluation calls."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('final_audit', Path(__file__).parents[1] / 'scripts/audit_final_scores.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(tmp_path):
    def write(name, value, jsonl=False):
        path = tmp_path / name
        path.write_text(json.dumps(value) + ('\n' if jsonl else ''))
        return path
    split = tmp_path / 'split.csv'
    pd.DataFrame([{'solute_id': 's', 'split': 'test'}]).to_csv(split, index=False)
    measured = tmp_path / 'measured.parquet'
    pd.DataFrame([{'solute_id': 's', 'record_id': 'a', 'logx': -1.},
                  {'solute_id': 's', 'record_id': 'b', 'logx': -2.}]).to_parquet(measured)
    query = {'query_id': 'q', 'solute_id': 's', 'condition_id': 'c', 'split': 'test', 'task': 'pair',
             'candidates': [{'id': 'A', 'record_id': 'a', 'smiles': 'C'},
                            {'id': 'B', 'record_id': 'b', 'smiles': 'O'}]}
    q = write('queries.jsonl', query, True)
    t = write('targets.jsonl', {'query_id': 'q', 'values': {'A': -1., 'B': -2.}, 'choice': 'A'}, True)
    pred = tmp_path / 'predictions.parquet'
    pd.DataFrame([{'record_id': 'a', 'pred_logx': -1.}, {'record_id': 'b', 'pred_logx': -2.}]).to_parquet(pred)
    release = write('release.json', {'query_path': str(q), 'target_path': str(t),
        'split_path': str(split), 'measurement_path': str(measured)})
    out = tmp_path / 'evaluation'
    out.mkdir()
    scores = {'pair_accuracy': 1., 'nontie_pair_accuracy': 1., 'top1': None,
              'rank_tau': None, 'regret': 0., 'covered': 1.}
    pd.DataFrame([{'query_id': 'q', 'solute_id': 's', 'condition_id': 'c', 'task': 'pair', **scores}]).to_parquet(out / 'per_query.parquet')
    pd.DataFrame([{'solute_id': 's', 'task': 'pair', **scores}]).to_csv(out / 'per_solute.csv', index=False)
    (out / 'metrics.json').write_text(json.dumps({'inputs': {str(p): audit.grader.sha256_file(p) for p in [q, t, pred]},
        'by_task': {'pair': {'queries': 1, 'solutes': 1, **scores}}}))
    command = ['python', '--queries', str(q), '--targets', str(t), '--predictions', str(pred),
               '--output', str(out), '--freeze-manifest', str(release), '--property-predictions']
    task = {'command': command, 'depends_on': ['prediction'], 'protocol': 'identity'}
    state = {'tasks': {'prediction': {'status': 'completed', 'output_sha256': {str(pred): audit.grader.sha256_file(pred)}}}}
    return task, state, out, split


def test_hand_computed_test_pair_passes(tmp_path):
    task, state, _, _ = fixture(tmp_path)
    result = audit.audit_method(task, state, {})
    assert result['status'] == 'PASSED'
    assert result['independent_metrics']['pair']['pair_accuracy'] == 1.
    assert result['independent_metrics']['pair']['regret'] == 0.


def test_non_test_identity_is_rejected(tmp_path):
    task, state, _, split = fixture(tmp_path)
    pd.DataFrame([{'solute_id': 's', 'split': 'train'}]).to_csv(split, index=False)
    with pytest.raises(ValueError, match='test membership'):
        audit.audit_method(task, state, {})


def test_incorrect_scientific_score_is_rejected(tmp_path):
    task, state, out, _ = fixture(tmp_path)
    p = out / 'per_query.parquet'
    frame = pd.read_parquet(p)
    frame['regret'] = .2
    frame.to_parquet(p)
    with pytest.raises(ValueError, match='per-query score mismatch: regret'):
        audit.audit_method(task, state, {})

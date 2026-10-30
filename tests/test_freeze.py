"""Synthetic safety fixtures only: no scientific or held-out model results."""
import json
from pathlib import Path

import pytest

import solvent_select.freeze as freezing
from solvent_select.views import sha256_file


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def queue(tmp_path, completed=True):
    source = tmp_path / 'input.txt'
    source.write_text('fixed public input')
    output = write(tmp_path / 'result.json', {'status': 'completed'})
    plan = write(tmp_path / 'plan.json', {'input_sha256': {str(source): sha256_file(source)},
                 'tasks': [{'id': 'one', 'command': [], 'outputs': [{'path': str(output), 'status': 'completed'}]}]})
    record = {'status': 'completed' if completed else 'running', 'exit_code': 0,
              'output_sha256': {str(output): sha256_file(output)}}
    state = write(tmp_path / 'state.json', {'plan_sha256': sha256_file(plan),
                  'status': 'completed' if completed else 'running', 'tasks': {'one': record}})
    return plan, state, source, output


def test_queue_rejects_changed_pins_and_changed_completed_outputs(tmp_path):
    plan, state, source, output = queue(tmp_path)
    assert freezing.queue_evidence(plan, state)['errors'] == []
    source.write_text('changed')
    output.write_text('{}')
    result = freezing.queue_evidence(plan, state)
    assert len(result['errors']) == 2
    assert any('Pinned input changed' in s for s in result['errors'])
    assert any('Completed output missing/modified' in s for s in result['errors'])


def test_exit_zero_does_not_complete_running_task(tmp_path):
    plan, state, _, _ = queue(tmp_path, completed=False)
    result = freezing.queue_evidence(plan, state)
    assert result['pending'] == [{'task': 'one', 'status': 'running'}]


def test_record_requires_hash_evidence_and_detects_alteration(tmp_path):
    evidence = tmp_path / 'evidence'
    evidence.write_text('a real file')
    record = write(tmp_path / 'review.json', {'status': 'PASSED'})
    assert freezing.verified_record(record, 'PASSED')[0] is None
    write(record, {'status': 'PASSED', 'evidence_sha256': {str(evidence): sha256_file(evidence)}})
    assert freezing.verified_record(record, 'PASSED')[1] is None
    evidence.write_text('edited')
    assert 'Changed evidence' in freezing.verified_record(record, 'PASSED')[1]


def test_incomplete_validation_cannot_write_frozen_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan, state, _, _ = queue(tmp_path, completed=False)
    monkeypatch.setattr(freezing, 'consistency_paths', lambda queues: [])
    result = freezing.preflight([(plan, state)], 'missing_decision.json', 'missing_review.json')
    assert result['status'] == 'INCOMPLETE'
    assert result['pending_tasks'] == 1
    with pytest.raises(ValueError, match='Pre-test audit incomplete'):
        freezing.freeze(tmp_path / 'frozen', 'missing_analysis.json', [(plan, state)],
                        'missing_decision.json', 'missing_review.json')
    assert not (tmp_path / 'frozen').exists()


def test_consistency_results_must_precede_freeze_even_with_complete_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan, state, _, _ = queue(tmp_path)
    monkeypatch.setattr(freezing, 'consistency_paths', lambda queues: [Path('missing_consistency.json')])
    evidence = tmp_path / 'evidence'
    evidence.write_text('verified evidence fixture')
    evidence_sha256 = {str(evidence): sha256_file(evidence)}
    decision = write('decision.json', {'status': 'VALIDATION_DECISION_RECORDED', 'test_used': False,
                     'selected_methods': 'ALL_REGISTERED', 'evidence_sha256': evidence_sha256})
    review = write('review.json', {'status': 'PASSED_PREFREEZE_REVIEW', 'evidence_sha256': evidence_sha256,
                   'checks': {key: True for key in ['data_reconstruction', 'split_leakage', 'label_isolation',
                              'retrieval_provenance', 'metric_hand_checks', 'all_validation_receipts',
                              'consistency_validation', 'manuscript_claim_scope']}})
    for protocol in ['identity', 'scaffold']:
        names = [f'sourceqa_{protocol}_nearest'] + [f'sourceqa_{protocol}_{kind}_{seed}'
                 for kind in ['lgbm', 'chemprop'] for seed in [17, 42, 2026]]
        for name in names:
            write(Path('artifacts/runs_sourceqa') / name / 'metadata.json', {'status': 'completed'})
    result = freezing.preflight([(plan, state)], decision, review)
    assert result['gaps'] == []
    assert result['missing_consistency'] == ['missing_consistency.json']
    assert result['status'] == 'INCOMPLETE'

import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location('development_queue', Path(__file__).parents[1] / 'scripts/run_development_queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def plan(tmp_path, monkeypatch, status='completed', code=0):
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    output = tmp_path / 'output.json'
    command = [sys.executable, '-c', f'import json; from pathlib import Path; Path({str(output)!r}).write_text(json.dumps({{"status": {status!r}}})); raise SystemExit({code})']
    config = {'partition': 'val', 'test_forbidden': True, 'input_sha256': {}, 'tasks': [
        {'id': 'one', 'command': command, 'outputs': [{'path': str(output), 'status': 'completed'}]}]}
    source = tmp_path / 'plan.json'
    source.write_text(json.dumps(config))
    return source, tmp_path / 'state.json', output


def test_completed_is_reused_and_modified_artifact_rejected(tmp_path, monkeypatch):
    source, state, output = plan(tmp_path, monkeypatch)
    assert queue.execute(source, state) == 0
    assert queue.execute(source, state) == 0
    output.write_text('{"status":"completed","tampered":true}')
    with pytest.raises(ValueError, match='artifacts changed'):
        queue.execute(source, state)


def test_exit_zero_does_not_make_interrupted_training_complete(tmp_path, monkeypatch):
    source, state, output = plan(tmp_path, monkeypatch, status='interrupted')
    with pytest.raises(ValueError, match='status=interrupted'):
        queue.execute(source, state)
    assert json.loads(state.read_text())['tasks']['one']['status'] == 'failed'


def test_nonzero_stops_before_advancing_and_requires_explicit_resume(tmp_path, monkeypatch):
    source, state, output = plan(tmp_path, monkeypatch, code=2)
    assert queue.execute(source, state) == 1
    with pytest.raises(ValueError, match='Explicit --resume-task'):
        queue.execute(source, state)


def test_pinned_input_change_rejected_before_execution(tmp_path, monkeypatch):
    source, state, output = plan(tmp_path, monkeypatch)
    config = json.loads(source.read_text())
    config['input_sha256'][str(source)] = 'wrong'
    source.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Pinned input changed'):
        queue.execute(source, state)
    assert not output.exists()


def test_live_child_refused(tmp_path, monkeypatch):
    source, state, output = plan(tmp_path, monkeypatch)
    monkeypatch.setattr(queue, 'active_task_processes', lambda task: [123])
    with pytest.raises(RuntimeError, match='Existing task processes'):
        queue.execute(source, state)
    assert not output.exists()

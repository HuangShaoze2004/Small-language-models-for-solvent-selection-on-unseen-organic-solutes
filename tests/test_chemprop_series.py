"""CPU-only lifecycle mocks: never launch a model or touch real study runs."""
import importlib.util
import json
import os
from pathlib import Path
import signal

import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_chemprop_series.py"
SPEC = importlib.util.spec_from_file_location("chemprop_series_test_module", SCRIPT)
series = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(series)


@pytest.fixture
def mock_root(tmp_path):
    contents = {"data/processed/measurements.parquet": b"mock measurements, not scientific data",
                "data/splits/identity.csv": b"solute_id,split\ns1,train\ns2,val\n",
                "src/solvent_select/chemprop_runner.py": b"mock implementation",
                "data/processed/identity/val_queries.jsonl": b"mock validation queries",
                "data/targets/identity/val.jsonl": b"mock validation targets"}
    for name, value in contents.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    config = tmp_path / "configs/baselines_chemprop.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({"hidden_dim": 300, "max_epochs": 50}))
    return tmp_path


def write_model(root, seed, status="completed", *, predictions=True, checkpoint=True):
    expected = series.expected_inputs(root)
    folder = series.run_folder(root, seed)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {"method": "chemprop", "seed": seed, "purpose": "formal", "target": "log10_mole_fraction",
                "config": expected["config"], "measurement_sha256": expected["sha256"]["measurements"],
                "split_sha256": expected["sha256"]["splits"], "code_sha256": expected["sha256"]["code"],
                "execution_limits": {"train_batches": None, "val_batches": None}, "status": status,
                "measurements": str(root / expected["paths"]["measurements"]),
                "splits": str(root / expected["paths"]["splits"])}
    series.atomic_json(folder / "metadata.json", metadata)
    series.atomic_json(folder / "experiment.json", {"kind": "chemprop", "protocol": "identity", "seed": seed,
                       "status": status, "pid": 999999, "wrapper_pid": 999998})
    if predictions:
        (folder / "predictions_val.parquet").write_bytes(f"mock predictions {seed}".encode())
    if checkpoint:
        (folder / "checkpoints").mkdir(exist_ok=True)
        (folder / "checkpoints/last.ckpt").write_bytes(b"mock complete checkpoint")


def write_evaluation(root, seed):
    expected = series.expected_inputs(root)
    folder = series.run_folder(root, seed)
    evaluation = folder / "evaluation"
    evaluation.mkdir(exist_ok=True)
    inputs = {expected["paths"]["queries"]: expected["sha256"]["queries"],
              expected["paths"]["targets"]: expected["sha256"]["targets"],
              str(folder / "predictions_val.parquet"): series.digest(folder / "predictions_val.parquet")}
    series.atomic_json(evaluation / "metrics.json", {"inputs": inputs, "mock_only": True})
    (evaluation / "per_query.parquet").write_bytes(b"mock per query")
    (evaluation / "per_solute.csv").write_text("mock per solute\n")


class FakeProcesses:
    def __init__(self, root, *, false_success=False, terminate=False):
        self.root, self.calls = root, []
        self.false_success, self.terminate = false_success, terminate

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        owner = self
        train = "scripts/run_experiment.py" in command
        seed = int(series.option(command, "--seed")) if train else int(Path(series.option(command, "--output")).parent.name.rsplit("_", 1)[1])

        class FakeProcess:
            pid = 1234567
            returncode = None
            stop_signal = None

            def poll(self):
                return self.returncode

            def send_signal(self, signum):
                self.stop_signal = signum

            def wait(self, timeout=None):
                if train:
                    if owner.terminate:
                        signal.raise_signal(signal.SIGTERM)
                    status = "interrupted" if self.stop_signal or owner.false_success else "completed"
                    write_model(owner.root, seed, status, predictions=status == "completed")
                    if owner.false_success:
                        exp = series.read_json(series.run_folder(owner.root, seed) / "experiment.json")
                        exp["status"] = "completed"
                        series.atomic_json(series.run_folder(owner.root, seed) / "experiment.json", exp)
                else:
                    write_evaluation(owner.root, seed)
                self.returncode = 0
                return self.returncode
        return FakeProcess()


def no_processes(*args):
    return []


def test_completed_series_is_reused_without_subprocesses(mock_root):
    for seed in series.SEEDS:
        write_model(mock_root, seed)
        write_evaluation(mock_root, seed)
    fake = FakeProcesses(mock_root)
    assert series.Series(mock_root, popen=fake, inspector=no_processes).execute() == 0
    assert fake.calls == []
    state = series.read_json(mock_root / "artifacts/chemprop_series/series_state.json")
    assert state["status"] == "completed"
    assert [r["training_action"] for r in state["results"]] == ["reuse"] * 3


def test_completed_first_seed_only_gets_missing_validation_then_next_seeds_train(mock_root):
    write_model(mock_root, 17)
    fake = FakeProcesses(mock_root)
    assert series.Series(mock_root, popen=fake, inspector=no_processes).execute() == 0
    training = [c for c in fake.calls if "scripts/run_experiment.py" in c]
    assert [series.option(c, "--seed") for c in training] == ["42", "2026"]
    assert "solvent_select.evaluate" in fake.calls[0]
    for command in fake.calls:
        if "solvent_select.evaluate" in command:
            assert Path(series.option(command, "--queries")).name == "val_queries.jsonl"
            assert Path(series.option(command, "--targets")).name == "val.jsonl"


def test_live_process_blocks_even_explicit_resume(mock_root):
    write_model(mock_root, 17, "running", predictions=False)
    fake = FakeProcesses(mock_root)
    inspector = lambda *args: [{"pid": 999, "command": ["python", "live trainer"]}]
    assert series.Series(mock_root, 17, popen=fake, inspector=inspector).execute() == 2
    assert fake.calls == []
    state = series.read_json(mock_root / "artifacts/chemprop_series/series_state.json")
    assert state["status"] == "active_run_detected"
    assert state["recovery"]["resume_command"] is None


def test_zero_exit_without_completed_model_stops_before_next_seed(mock_root):
    fake = FakeProcesses(mock_root, false_success=True)
    assert series.Series(mock_root, popen=fake, inspector=no_processes).execute() == 1
    assert len(fake.calls) == 1
    state = series.read_json(mock_root / "artifacts/chemprop_series/series_state.json")
    assert state["status"] == "failed" and state["current_seed"] == 17
    assert "not completed" in state["error"]


def test_failed_run_requires_explicit_matching_resume(mock_root):
    write_model(mock_root, 17, "failed", predictions=False)
    fake = FakeProcesses(mock_root)
    assert series.Series(mock_root, popen=fake, inspector=no_processes).execute() == 3
    assert fake.calls == []
    assert series.Series(mock_root, 17, popen=fake, inspector=no_processes).execute() == 0
    first = fake.calls[0]
    assert first.count("--resume") == 2  # wrapper and trainer
    assert [series.option(c, "--seed") for c in fake.calls if "scripts/run_experiment.py" in c] == ["17", "42", "2026"]


def test_resume_requires_checkpoint_and_identical_parameters(mock_root):
    write_model(mock_root, 17, "interrupted", predictions=False, checkpoint=False)
    fake = FakeProcesses(mock_root)
    assert series.Series(mock_root, 17, popen=fake, inspector=no_processes).execute() == 3
    assert fake.calls == []
    write_model(mock_root, 17, "interrupted", predictions=False)
    path = series.run_folder(mock_root, 17) / "metadata.json"
    metadata = series.read_json(path)
    metadata["config"]["hidden_dim"] = 123
    series.atomic_json(path, metadata)
    assert series.Series(mock_root, 17, popen=fake, inspector=no_processes).execute() == 3
    assert fake.calls == []


def test_sigterm_forwarded_and_no_later_seed_starts(mock_root):
    fake = FakeProcesses(mock_root, terminate=True)
    assert series.Series(mock_root, popen=fake, inspector=no_processes).execute() == 143
    assert len(fake.calls) == 1
    state = series.read_json(mock_root / "artifacts/chemprop_series/series_state.json")
    assert state["status"] == "interrupted" and state["stop_signal"] == signal.SIGTERM
    assert state["current_seed"] == 17


def test_second_queue_cannot_acquire_lock(mock_root):
    fake = FakeProcesses(mock_root)
    with series.queue_lock(mock_root / "artifacts/chemprop_series.lock"):
        with pytest.raises(series.QueueStop, match="holds the queue lock"):
            series.Series(mock_root, popen=fake, inspector=no_processes).execute()
    assert fake.calls == []


def test_command_matching_ignores_reused_pid_other_project_and_shell_substrings(mock_root):
    argv = ["python", "scripts/run_experiment.py", "--run-id", "identity_chemprop_17"]
    assert series.command_matches_run(argv, mock_root, 17, mock_root)
    assert not series.command_matches_run(argv, mock_root, 42, mock_root)
    assert not series.command_matches_run(argv, mock_root, 17, Path("/another/project"))
    assert not series.command_matches_run(["bash", "-c", " ".join(argv)], mock_root, 17, mock_root)


def test_existing_stale_evaluation_is_not_silently_overwritten(mock_root):
    write_model(mock_root, 17)
    write_evaluation(mock_root, 17)
    prediction = series.run_folder(mock_root, 17) / "predictions_val.parquet"
    prediction.write_bytes(b"changed prediction file")
    fake = FakeProcesses(mock_root)
    assert series.Series(mock_root, popen=fake, inspector=no_processes).execute() == 3
    assert fake.calls == []

#!/usr/bin/env python3
"""Run identity Chemprop seeds 17/42/2026 serially, then evaluate validation only.

Completed matching runs are reused. Failed/interrupted/stale runs require an
explicit ``--resume SEED`` and a matching complete last.ckpt. A live process is
never duplicated. This queue owns only chemprop_series.lock; the model runner
independently owns the GPU lock, so nested GPU-lock deadlocks are avoided.

    .venv/bin/python scripts/run_chemprop_series.py
    .venv/bin/python scripts/run_chemprop_series.py --resume 42
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 42, 2026)
RECOVERABLE = {"failed", "interrupted", "running", "starting"}


class QueueStop(RuntimeError):
    def __init__(self, message, *, code=3, detail=None):
        super().__init__(message)
        self.code = code
        self.detail = detail or {}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextmanager
def queue_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            raise QueueStop("Another Chemprop series holds the queue lock; attach to that process.",
                            code=2, detail={"lock_path": str(path), "owner": handle.read()}) from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_utc": now(), "command": sys.argv}) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_id(seed):
    return f"identity_chemprop_{seed}"


def run_folder(root, seed):
    return root / "artifacts/runs" / run_id(seed)


def option(argv, name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def command_matches_run(argv, root, seed, cwd=None):
    """Check tokenized process command lines, not mere substring/PID matches."""
    scripts = [Path(token) for token in argv if Path(token).name == "run_experiment.py"]
    if scripts:
        script = scripts[0] if scripts[0].is_absolute() else (cwd or root) / scripts[0]
        return option(argv, "--run-id") == run_id(seed) and script.resolve() == (root / "scripts/run_experiment.py").resolve()
    if option(argv, "-m") != "solvent_select.chemprop_runner":
        return False
    output = option(argv, "--output") or option(argv, "--run")
    if output is None:
        return False
    path = Path(output)
    if not path.is_absolute():
        path = (cwd or root) / path
    return path.resolve() == run_folder(root, seed).resolve()


def find_active_processes(root, seed, experiment=None, metadata=None):
    """Resolve recorded PIDs and scan for matching live wrappers/model runners.

    A reused PID with a different command is not considered the old run.
    Unreadable recorded live processes fail closed.
    """
    recorded = {int(entry[key]) for entry in [experiment or {}, metadata or {}]
                for key in ["pid", "wrapper_pid"] if isinstance(entry.get(key), int)}
    found = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        pid = int(directory.name)
        if pid == os.getpid():
            continue
        try:
            argv = [value.decode(errors="replace") for value in (directory / "cmdline").read_bytes().split(b"\0") if value]
            if not argv:  # exited/zombie process
                continue
            try:
                cwd = (directory / "cwd").resolve(strict=True)
            except OSError:
                cwd = root
            if command_matches_run(argv, root, seed, cwd):
                found.append({"pid": pid, "command": argv, "recorded_pid": pid in recorded})
        except FileNotFoundError:
            continue
        except PermissionError:
            if pid in recorded:
                found.append({"pid": pid, "command": None, "recorded_pid": True,
                              "reason": "Recorded live PID is unreadable; cannot safely start a duplicate"})
    return found


def expected_inputs(root):
    paths = {"measurements": "data/processed/measurements.parquet", "splits": "data/splits/identity.csv",
             "config": "configs/baselines_chemprop.yaml", "code": "src/solvent_select/chemprop_runner.py",
             "queries": "data/processed/identity/val_queries.jsonl", "targets": "data/targets/identity/val.jsonl"}
    for path in paths.values():
        if not (root / path).is_file():
            raise QueueStop(f"Required fixed input is missing: {path}")
    return {"paths": paths, "sha256": {key: digest(root / path) for key, path in paths.items()},
            "config": yaml.safe_load((root / paths["config"]).read_text())}


def validate_metadata(root, seed, metadata, expected):
    fields = {"method": "chemprop", "seed": seed, "purpose": "formal", "target": "log10_mole_fraction",
              "config": expected["config"], "measurement_sha256": expected["sha256"]["measurements"],
              "split_sha256": expected["sha256"]["splits"], "code_sha256": expected["sha256"]["code"],
              "execution_limits": {"train_batches": None, "val_batches": None}}
    for key, value in fields.items():
        if metadata.get(key) != value:
            raise QueueStop(f"Seed {seed} metadata does not match fixed parameters: {key}",
                            detail={"field": key, "run_id": run_id(seed)})
    for key in ["measurements", "splits"]:
        if Path(metadata.get(key, "")).resolve() != (root / expected["paths"][key]).resolve():
            raise QueueStop(f"Seed {seed} points to a different {key} file")


def validate_evaluation(root, seed, expected):
    folder = run_folder(root, seed)
    required = [folder / "evaluation" / name for name in ["metrics.json", "per_query.parquet", "per_solute.csv"]]
    if not all(path.is_file() for path in required):
        return False
    metrics = read_json(required[0])
    inputs = {str((Path(path) if Path(path).is_absolute() else root / path).resolve()): value
              for path, value in metrics.get("inputs", {}).items()}
    hashes = {str((root / expected["paths"]["queries"]).resolve()): expected["sha256"]["queries"],
              str((root / expected["paths"]["targets"]).resolve()): expected["sha256"]["targets"],
              str((folder / "predictions_val.parquet").resolve()): digest(folder / "predictions_val.parquet")}
    if any(inputs.get(path) != value for path, value in hashes.items()):
        raise QueueStop(f"Seed {seed} existing validation metrics have mismatched/missing input hashes; inspect before replacing them")
    return True


def training_command(root, seed, resume=False):
    wrapper = [str(root / ".venv/bin/python"), "scripts/run_experiment.py", "--run-id", run_id(seed),
               "--kind", "chemprop", "--protocol", "identity", "--seed", str(seed)]
    trainer = [str(root / ".venv-chemprop/bin/python"), "-u", "-m", "solvent_select.chemprop_runner", "train",
               "--data", "data/processed/measurements.parquet", "--splits", "data/splits/identity.csv",
               "--config", "configs/baselines_chemprop.yaml", "--seed", str(seed), "--accelerator", "gpu",
               "--purpose", "formal", "--output", str(Path("artifacts/runs") / run_id(seed))]
    if resume:
        wrapper.append("--resume")
        trainer.append("--resume")
    return wrapper + ["--"] + trainer


def evaluation_command(root, seed):
    folder = Path("artifacts/runs") / run_id(seed)
    return [str(root / ".venv/bin/python"), "-u", "-m", "solvent_select.evaluate",
            "--queries", "data/processed/identity/val_queries.jsonl", "--targets", "data/targets/identity/val.jsonl",
            "--predictions", str(folder / "predictions_val.parquet"), "--property-predictions",
            "--output", str(folder / "evaluation")]


def inspect_run(root, seed, expected, resume_seed=None, inspector=find_active_processes):
    folder = run_folder(root, seed)
    experiment, metadata = read_json(folder / "experiment.json"), read_json(folder / "metadata.json")
    active = inspector(root, seed, experiment, metadata)
    if active:
        raise QueueStop(f"Seed {seed} is still active; attach to its console.log instead of starting another process.",
                        code=2, detail={"active_processes": active, "attach_command": ["tail", "-f", str(folder / "console.log")]})
    if experiment is None and metadata is None:
        if folder.exists() and any(folder.iterdir()):
            raise QueueStop(f"Untracked files already exist in {folder}; inspect the incomplete run")
        if resume_seed == seed:
            raise QueueStop(f"Seed {seed} has no existing run to resume")
        return "train"
    if experiment is None or metadata is None:
        raise QueueStop(f"Seed {seed} is missing experiment.json or metadata.json; inspect the partial run before proceeding")
    validate_metadata(root, seed, metadata, expected)
    if experiment.get("kind") != "chemprop" or experiment.get("protocol") != "identity" or experiment.get("seed") != seed:
        raise QueueStop(f"Seed {seed} wrapper provenance does not match the fixed series")
    if metadata.get("status") == "completed" and experiment.get("status") == "completed":
        if not (folder / "predictions_val.parquet").is_file():
            raise QueueStop(f"Seed {seed} completed training but predictions are missing; recover predictions without retraining")
        return "reuse" if validate_evaluation(root, seed, expected) else "evaluate"
    if metadata.get("status") == "completed":
        raise QueueStop(f"Seed {seed} model completed but wrapper status is {experiment.get('status')}; inspect the discrepancy, do not retrain")
    if metadata.get("status") not in RECOVERABLE:
        raise QueueStop(f"Seed {seed} has unknown model status {metadata.get('status')}")
    if resume_seed != seed:
        raise QueueStop(f"Seed {seed} is {metadata.get('status')} and not live; explicit --resume {seed} is required")
    if not (folder / "checkpoints/last.ckpt").is_file():
        raise QueueStop(f"Seed {seed} has no complete last.ckpt; cannot resume")
    return "resume"


class Series:
    def __init__(self, root, resume_seed=None, *, popen=subprocess.Popen, inspector=find_active_processes):
        self.root = Path(root).resolve()
        self.resume_seed = resume_seed
        self.popen = popen
        self.inspector = inspector
        self.directory = self.root / "artifacts/chemprop_series"
        self.state_path = self.directory / "series_state.json"
        self.state = {}
        self.process = None
        self.stop_signal = None

    def persist(self, **updates):
        self.state.update(updates, updated_utc=now())
        atomic_json(self.state_path, self.state)

    def stop(self, signum, frame=None):
        self.stop_signal = signum
        self.persist(status="stopping", stop_signal=signum)
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signum)

    def launch(self, seed, phase, command):
        if self.stop_signal is not None:
            raise QueueStop("Series stop requested; no further process will be launched", code=128 + self.stop_signal)
        log_path = self.directory / f"seed{seed}_{phase}.log"
        self.persist(current_seed=seed, phase=phase, command=command, wrapper_pid=None, child_pid=None,
                     phase_log=str(log_path))
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONUNBUFFERED"] = "1"
        started = time.monotonic()
        with log_path.open("a", buffering=1) as log:
            self.process = self.popen(command, cwd=self.root, stdout=log, stderr=subprocess.STDOUT, env=env)
            self.persist(wrapper_pid=self.process.pid if phase == "train" else None,
                         subprocess_pid=self.process.pid)
            if self.stop_signal is not None:
                self.process.send_signal(self.stop_signal)
            while True:
                try:
                    code = self.process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    experiment = read_json(run_folder(self.root, seed) / "experiment.json") or {}
                    self.persist(child_pid=experiment.get("pid"), wrapper_status=experiment.get("status"))
            self.process = None
        self.state.setdefault("phase_results", []).append({"seed": seed, "phase": phase,
            "command": command, "exit_code": code, "elapsed_seconds": time.monotonic() - started, "log": str(log_path)})
        self.persist(last_exit_code=code, subprocess_pid=None)
        if self.stop_signal is not None:
            raise QueueStop("Stop signal forwarded; queue halted before the next seed", code=128 + self.stop_signal)
        if code != 0:
            raise QueueStop(f"Seed {seed} {phase} exited with code {code}; no later seed will start", code=1)

    def execute(self):
        with queue_lock(self.root / "artifacts/chemprop_series.lock"):
            expected = expected_inputs(self.root)
            previous = read_json(self.state_path)
            history_path = None
            if previous is not None:
                history_path = self.directory / "history" / f"session_{time.time_ns()}.json"
                atomic_json(history_path, previous)
            self.state = {"series": "identity_chemprop_seeds_17_42_2026", "seeds": list(SEEDS),
                          "status": "running", "started_utc": now(), "queue_pid": os.getpid(),
                          "resume_seed": self.resume_seed, "input_fingerprints": expected,
                          "results": [], "phase_results": [], "previous_state_path": str(history_path) if history_path else None,
                          "validation_only": True, "gpu_lock_owned_by": "chemprop_runner", "current_seed": None}
            self.persist()
            old_handlers = {sig: signal.getsignal(sig) for sig in [signal.SIGTERM, signal.SIGINT]}
            for sig in old_handlers:
                signal.signal(sig, self.stop)
            try:
                if self.resume_seed is not None:
                    # Validate the explicit recovery target before starting any other job.
                    action = inspect_run(self.root, self.resume_seed, expected, self.resume_seed, self.inspector)
                    if action != "resume":
                        raise QueueStop("--resume must identify an incomplete training run with last.ckpt; completed models are reused without --resume")
                for seed in SEEDS:
                    self.persist(current_seed=seed, phase="inspect", command=None)
                    if self.stop_signal is not None:
                        raise QueueStop("Queue stop requested", code=128 + self.stop_signal)
                    # Refuse changed data/config/code even between seeds of this session.
                    if expected_inputs(self.root) != expected:
                        raise QueueStop("Fixed series inputs changed during execution")
                    action = inspect_run(self.root, seed, expected, self.resume_seed, self.inspector)
                    print(f"seed={seed} action={action}", flush=True)
                    if action in {"train", "resume"}:
                        self.launch(seed, "train", training_command(self.root, seed, resume=action == "resume"))
                        metadata = read_json(run_folder(self.root, seed) / "metadata.json") or {}
                        experiment = read_json(run_folder(self.root, seed) / "experiment.json") or {}
                        if metadata.get("status") != "completed" or experiment.get("status") != "completed":
                            raise QueueStop(f"Seed {seed} exited zero but model/wrapper is not completed; refusing to advance", code=1)
                        # This also checks lineage and the prediction file after success.
                        following = inspect_run(self.root, seed, expected, None, self.inspector)
                        if following not in {"evaluate", "reuse"}:
                            raise QueueStop(f"Unexpected post-training state for seed {seed}: {following}")
                    if not validate_evaluation(self.root, seed, expected):
                        self.launch(seed, "evaluate", evaluation_command(self.root, seed))
                        if not validate_evaluation(self.root, seed, expected):
                            raise QueueStop(f"Seed {seed} evaluator returned without complete artifacts", code=1)
                    folder = run_folder(self.root, seed)
                    self.state["results"].append({"seed": seed, "run_id": run_id(seed), "status": "completed",
                        "training_action": action, "metadata": str(folder / "metadata.json"),
                        "metadata_sha256": digest(folder / "metadata.json"),
                        "predictions": str(folder / "predictions_val.parquet"),
                        "predictions_sha256": digest(folder / "predictions_val.parquet"),
                        "metrics": str(folder / "evaluation/metrics.json"),
                        "metrics_sha256": digest(folder / "evaluation/metrics.json")})
                    self.persist(phase="seed_completed")
                self.persist(status="completed", phase="completed", current_seed=None, command=None,
                             wrapper_pid=None, child_pid=None, finished_utc=now())
                return 0
            except BaseException as error:
                seed = self.state.get("current_seed") or self.resume_seed
                code = error.code if isinstance(error, QueueStop) else 1
                status = "interrupted" if self.stop_signal is not None else "active_run_detected" if code == 2 else "failed"
                model_metadata = read_json(run_folder(self.root, seed) / "metadata.json") if seed else None
                model_completed = bool(model_metadata and model_metadata.get("status") == "completed")
                resume_command = ([str(self.root / ".venv/bin/python"), "scripts/run_chemprop_series.py", "--resume", str(seed)]
                                  if seed and not model_completed and code != 2 else None)
                self.persist(status=status, error=repr(error), detail=getattr(error, "detail", {}),
                             finished_utc=now(), recovery={
                                 "inspect_run": str(run_folder(self.root, seed)) if seed else None,
                                 "checkpoint": str(run_folder(self.root, seed) / "checkpoints/last.ckpt") if seed else None,
                                 "resume_command": resume_command,
                                 "validation_recovery_command": evaluation_command(self.root, seed) if model_completed else None,
                                 "queue_command_after_recovery": [str(self.root / ".venv/bin/python"), "scripts/run_chemprop_series.py"],
                                 "note": "Inspect failure and live PIDs first; explicit resume requires unchanged inputs and a complete checkpoint. Never resume a completed model."})
                print(str(error), file=sys.stderr, flush=True)
                return code
            finally:
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=int, choices=SEEDS, metavar="SEED")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        return Series(args.project_root, args.resume).execute()
    except QueueStop as error:
        print(str(error), file=sys.stderr)
        if error.detail:
            print(json.dumps(error.detail, indent=2), file=sys.stderr)
        return error.code


if __name__ == "__main__":
    raise SystemExit(main())

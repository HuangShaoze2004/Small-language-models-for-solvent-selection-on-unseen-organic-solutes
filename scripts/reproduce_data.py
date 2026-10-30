#!/usr/bin/env python3
"""Rebuild all study data from the fixed raw CSV in a new isolated directory.

This command does not overwrite the project's data. It reuses the verified
raw CSV through a symlink, copies the archived superseded split for provenance,
then calls the current data.build pipeline and compares all scientific artifact
bytes with the active project. Build output and failed attempts remain inspectable.

Run with the locked main environment:
    .venv/bin/python scripts/reproduce_data.py
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from solvent_select.data import build as build_data  # noqa: E402
from solvent_select.strict_split import protected_hashes  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def git_identity(root: Path) -> dict:
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    patch = git("diff", "HEAD", "--", "src/solvent_select/data.py", "src/solvent_select/strict_split.py")
    return {"commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--porcelain"),
            "reconstruction_code_diff_sha256": hashlib.sha256((patch or "").encode()).hexdigest()}


def software_versions() -> dict:
    versions = {"python": platform.python_version(), "python_executable": sys.executable,
                "platform": platform.platform()}
    for package in ["rdkit", "numpy", "pandas", "pyarrow", "pyyaml", "scipy", "scikit-learn"]:
        versions[package] = importlib.metadata.version(package)
    return versions


def artifact_paths(root: Path) -> list[str]:
    paths = set(protected_hashes(root))
    amendment = json.loads((root / "artifacts/audit/scaffold_amendment.json").read_text())
    paths.update(amendment["amended_artifact_hashes"])
    return sorted(paths)


def auxiliary_paths(root: Path) -> list[str]:
    paths = ["data/raw/BigSolDBv2.0.csv", "configs/data.yaml",
             "artifacts/audit/superseded_scaffold_manifest.csv",
             "artifacts/audit/scaffold_amendment.json", "artifacts/audit/data_audit.json",
             "artifacts/audit/data_manifest.json", "artifacts/audit/solubench_overlap.json",
             "src/solvent_select/data.py", "src/solvent_select/strict_split.py"]
    return [path for path in paths if (root / path).exists()]


def reproduce(root: Path, evidence_path: Path, isolated_directory: Path | None = None) -> dict:
    import yaml

    root = root.resolve()
    config_path = root / "configs/data.yaml"
    raw_path = root / "data/raw/BigSolDBv2.0.csv"
    archive_path = root / "artifacts/audit/superseded_scaffold_manifest.csv"
    cfg = yaml.safe_load(config_path.read_text())
    if sha256(raw_path) != cfg["raw_sha256"]:
        raise ValueError("Raw CSV does not match the preregistered checksum")
    expected_archive = json.loads((root / "artifacts/audit/scaffold_amendment.json").read_text())["superseded_manifest_sha256"]
    if sha256(archive_path) != expected_archive:
        raise ValueError("Archived superseded split does not match its preserved checksum")
    paths = artifact_paths(root)
    expected_hashes = {path: sha256(root / path) for path in paths}
    untouched_before = {path: sha256(root / path) for path in auxiliary_paths(root)}
    if isolated_directory is None:
        (root / ".tmp").mkdir(exist_ok=True)
        isolated_directory = Path(tempfile.mkdtemp(prefix="full_data_rebuild_", dir=root / ".tmp"))
    else:
        isolated_directory = isolated_directory.resolve()
        if isolated_directory.exists() and any(isolated_directory.iterdir()):
            raise FileExistsError("Reconstruction requires a fresh, empty isolated directory")
        isolated_directory.mkdir(parents=True, exist_ok=True)
    if isolated_directory == root or root.is_relative_to(isolated_directory):
        raise ValueError("The isolated directory cannot be the project or an ancestor")
    previous_evidence = None
    if evidence_path.exists():
        prior = json.loads(evidence_path.read_text())
        previous_evidence = {"path": str(Path(prior["isolated_directory"]) / "rebuild_evidence.json"),
                             "status": prior["status"], "summary_sha256": sha256(evidence_path)}
    (isolated_directory / "data/raw").mkdir(parents=True, exist_ok=True)
    (isolated_directory / "artifacts/audit").mkdir(parents=True, exist_ok=True)
    (isolated_directory / "data/raw/BigSolDBv2.0.csv").symlink_to(raw_path)
    shutil.copyfile(archive_path, isolated_directory / "artifacts/audit/superseded_scaffold_manifest.csv")
    log_path = isolated_directory / "rebuild.log"
    solubench = json.loads((root / "artifacts/audit/solubench_overlap.json").read_text())
    reserved_qids = {key: solubench[key] for key in ["overlap_safe_qids", "strict_condition_qids"]}
    evidence = {
        "status": "RUNNING", "purpose": "full_raw_to_processed_data_reproduction",
        "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "source_project": str(root), "isolated_directory": str(isolated_directory),
        "command": f"{sys.executable} {Path(__file__).resolve()} --project-root {root}",
        "software_versions": software_versions(), "git": git_identity(root),
        "script_sha256": sha256(Path(__file__)), "config_sha256": sha256(config_path),
        "raw_csv_sha256": sha256(raw_path), "raw_access": "verified local CSV through read-only-use symlink; no download or upload",
        "superseded_archive_sha256": sha256(archive_path),
        "old_scaffold_manifest_supplied": False, "old_scaffold_amendment_supplied": False,
        "rebuild_log": str(log_path), "expected_artifact_hashes": expected_hashes,
        "main_auxiliary_hashes_before": untouched_before,
        "solubench_reservation": {
            "reselected": False, "benchmark_answers_used": False,
            "reserved_qids_sha256": hashlib.sha256(json.dumps(reserved_qids, sort_keys=True).encode()).hexdigest(),
            "audit_sha256": sha256(root / "artifacts/audit/solubench_overlap.json"),
        },
        "reconstruction_integration_relation": {
            "kind": "pure_reconstruction_integration_no_scientific_data_or_reservation_change",
            "data_code_sha256_at_solubench_reservation": solubench["input_fingerprints"]["data_code_sha256"],
            "data_code_sha256_current": sha256(root / "src/solvent_select/data.py"),
            "current_git_commit": git_identity(root)["commit"],
            "note": "The integrated builder now creates identity then the globally protected-test scaffold protocol; SoluBench qids remain sealed unchanged.",
        },
        "not_required_to_be_byte_identical": {
            "artifacts/audit/data_audit.json": "creation timestamp and integrated current protocol descriptions",
            "artifacts/audit/scaffold_amendment.json": "historical experiment scan, timestamp and prior validation history",
            "artifacts/audit/data_manifest.json": "all-protocol file inventory; scientific artifact hashes are checked separately",
        },
        "previous_evidence": previous_evidence,
    }
    atomic_json(isolated_directory / "rebuild_evidence.json", evidence)
    atomic_json(evidence_path, evidence)
    started = time.perf_counter()
    caught = None
    print(f"Rebuilding data in {isolated_directory}", flush=True)
    try:
        with log_path.open("w") as log, redirect_stdout(log), redirect_stderr(log):
            audit = build_data(isolated_directory, config_path)
        comparisons = []
        for path in paths:
            rebuilt = isolated_directory / path
            actual_hash = sha256(rebuilt) if rebuilt.exists() else None
            comparisons.append({"path": path, "expected_sha256": expected_hashes[path],
                                "rebuilt_sha256": actual_hash, "byte_identical": actual_hash == expected_hashes[path],
                                "rebuilt_bytes": rebuilt.stat().st_size if rebuilt.exists() else None})
        evidence["artifact_comparisons"] = comparisons
        evidence["artifacts_compared"] = len(comparisons)
        evidence["artifacts_byte_identical"] = sum(row["byte_identical"] for row in comparisons)
        evidence["rebuilt_data_counts"] = {"measurements": audit["final_measurements"],
                                           "canonical_solutes": audit["canonical_solutes"],
                                           "splits": audit["splits"],
                                           "strict_excluded": audit["strict_scaffold_development_exclusions"]}
        evidence["test_metrics_computed"] = False
        if evidence["artifacts_byte_identical"] != len(comparisons):
            raise AssertionError("Rebuilt scientific artifacts differ; inspect hashes without modifying scientific data to force agreement")
    except BaseException as error:
        caught = error
        evidence["error"] = repr(error)
        evidence["traceback"] = traceback.format_exc()
    finally:
        after = {path: sha256(root / path) for path in paths}
        untouched_after = {path: sha256(root / path) for path in untouched_before}
        evidence["main_scientific_files_unchanged"] = after == expected_hashes
        evidence["main_auxiliary_hashes_after"] = untouched_after
        evidence["main_auxiliary_files_unchanged"] = untouched_after == untouched_before
        evidence["elapsed_seconds"] = time.perf_counter() - started
        evidence["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        evidence["finished_utc"] = datetime.now(timezone.utc).isoformat()
        evidence["rebuild_log_sha256"] = sha256(log_path) if log_path.exists() else None
        evidence["status"] = "PASS" if caught is None and evidence["main_scientific_files_unchanged"] and evidence["main_auxiliary_files_unchanged"] else "FAIL"
        atomic_json(isolated_directory / "rebuild_evidence.json", evidence)
        atomic_json(evidence_path, evidence)
    print(json.dumps({key: evidence.get(key) for key in ["status", "artifacts_compared", "artifacts_byte_identical",
                                                       "main_scientific_files_unchanged", "main_auxiliary_files_unchanged",
                                                       "elapsed_seconds", "isolated_directory", "error"]}, indent=2), flush=True)
    if caught is not None:
        raise caught
    if evidence["status"] != "PASS":
        raise RuntimeError("Source project changed during isolated rebuild; inspect structured evidence")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--isolated-directory", type=Path)
    args = parser.parse_args()
    destination = args.evidence or args.project_root / "artifacts/audit/full_data_rebuild.json"
    reproduce(args.project_root, destination.resolve(), args.isolated_directory)

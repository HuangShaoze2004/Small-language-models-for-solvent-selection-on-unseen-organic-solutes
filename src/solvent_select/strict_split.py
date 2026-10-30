"""Scaffold extrapolation with one globally protected identity-test partition.

The original independent scaffold split is superseded before scientific use:
it could expose identity-test solutes through another protocol's development
pool. This amendment retains every identity-test solute, removes scaffold-near
development solutes, and partitions the remaining development scaffold groups.
No target value participates in the selection or task reallocation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import tempfile
import time

import pandas as pd

from .data import stable_id


EXPECTED_CURRENT_COUNTS = {
    "train": {"solutes": 743, "records": 52727},
    "val": {"solutes": 93, "records": 6243},
    "test": {"solutes": 144, "records": 10655},
    "excluded": {"solutes": 463, "records": 33890},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes(root: Path) -> dict[str, str]:
    paths = [root / "data/processed/measurements.parquet",
             root / "data/processed/solute_identities.parquet",
             root / "data/splits/identity.csv"]
    paths.extend(sorted((root / "data/processed/identity").glob("*.jsonl")))
    paths.extend(sorted((root / "data/targets/identity").glob("*.jsonl")))
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def observed_global_audit_hashes(root: Path) -> dict[str, str]:
    # These contain timestamps/all-protocol summaries. Record that this
    # amendment leaves them alone, without making future timestamps immutable.
    return {name: sha256(root / name) for name in ["artifacts/audit/data_manifest.json", "artifacts/audit/data_audit.json"]}


def assign_development_groups(groups: dict[str, list[str]], seed: int) -> dict[str, str]:
    """Same label-blind cumulative-midpoint rule as data.assign_splits, with two bins."""
    total = sum(len(members) for members in groups.values())
    boundary = total * 8 / 9
    allocated, assignments = 0, {}
    for group in sorted(groups, key=lambda key: stable_id(seed, key)):
        midpoint = allocated + len(groups[group]) / 2
        partition = "train" if midpoint < boundary else "val"
        for solute in groups[group]:
            if solute in assignments:
                raise ValueError("A solute occurs in more than one scaffold group")
            assignments[solute] = partition
        allocated += len(groups[group])
    if total >= 30 and set(assignments.values()) != {"train", "val"}:
        raise ValueError("Degenerate development split")
    return assignments


def derive_manifest(identity: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"solute_id", "leakage_group", "scaffold_group", "split"}
    if required - set(identity):
        raise ValueError(f"Missing identity-manifest columns: {required-set(identity)}")
    if identity.solute_id.isna().any() or identity.solute_id.duplicated().any():
        raise ValueError("Identity manifest must contain each solute exactly once")
    if not set(identity.split).issubset({"train", "val", "test"}):
        raise ValueError("Invalid source partition")
    if identity.groupby("leakage_group").split.nunique().max() != 1:
        raise ValueError("Identity manifest already leaks a parent group")
    test = identity.loc[identity.split.eq("test")].copy()
    reserved_scaffolds = set(test.scaffold_group)
    dev = identity.loc[identity.split.isin(["train", "val"])].copy()
    excluded = dev.loc[dev.scaffold_group.isin(reserved_scaffolds)].copy()
    excluded = excluded.rename(columns={"split": "original_split"})
    excluded["reason"] = "shares_scaffold_group_with_global_identity_test"
    retained = dev.loc[~dev.scaffold_group.isin(reserved_scaffolds)].copy()
    groups = retained.groupby("scaffold_group").solute_id.apply(list).to_dict()
    assignments = assign_development_groups(groups, seed)
    assignments.update({solute: "test" for solute in test.solute_id})
    result = identity.loc[identity.solute_id.isin(assignments)].copy()
    result["split"] = result.solute_id.map(assignments)
    for column in ["leakage_group", "scaffold_group"]:
        if result.groupby(column).split.nunique().max() != 1:
            raise AssertionError(f"New strict manifest leaks {column}")
    if set(result.loc[result.split.eq("test"), "solute_id"]) != set(test.solute_id):
        raise AssertionError("Global identity-test membership changed")
    if set(test.solute_id) & set(result.loc[result.split.ne("test"), "solute_id"]):
        raise AssertionError("Global test solute entered strict development")
    return result, excluded


def assert_no_superseded_experiments(root: Path, old_sha: str) -> list[dict]:
    """Inspect run manifests/ledger, excluding data summaries and model internals."""
    checked, candidates = [], []
    run_root = root / "artifacts/runs"
    for name in ["metadata.json", "experiment.json", "run.json"]:
        candidates.extend(run_root.rglob(name))
    ledger = root / "artifacts/experiments.jsonl"
    objects = []
    for path in sorted(set(candidates)):
        objects.append((str(path.relative_to(root)), json.loads(path.read_text())))
    if ledger.exists():
        objects.extend((f"{ledger.relative_to(root)}:{i+1}", json.loads(line))
                       for i, line in enumerate(ledger.read_text().splitlines()) if line.strip())
    for path, entry in objects:
        protocol = entry.get("protocol")
        explicit_split = str(entry.get("splits", entry.get("split_manifest", "")))
        argv = entry.get("argv", entry.get("command", []))
        argv_text = " ".join(map(str, argv)) if isinstance(argv, list) else str(argv)
        is_scaffold = protocol == "scaffold" or Path(explicit_split).name == "scaffold.csv" or "data/splits/scaffold.csv" in argv_text
        if not is_scaffold:
            checked.append({"path": path, "protocol": protocol or "non-scaffold", "status": entry.get("status")})
            continue
        split_hash = entry.get("split_sha256")
        if split_hash is None or split_hash == old_sha:
            raise RuntimeError(f"A possibly superseded-scaffold experiment exists: {path}; independent review is required")
        checked.append({"path": path, "protocol": "amended_scaffold", "status": entry.get("status")})
    return checked


def stream_tasks(root: Path, staging: Path, assignments: dict[str, str]) -> dict[str, dict]:
    """Copy original answer lines unchanged; only source identity controls assignment."""
    counts = {split: Counter() for split in ["train", "val", "test", "excluded"]}
    seen = set()
    with ExitStack() as stack:
        query_handles, target_handles = {}, {}
        for split in ["train", "val", "test"]:
            qpath = staging / "data/processed/scaffold" / f"{split}_queries.jsonl"
            tpath = staging / "data/targets/scaffold" / f"{split}.jsonl"
            qpath.parent.mkdir(parents=True, exist_ok=True)
            tpath.parent.mkdir(parents=True, exist_ok=True)
            query_handles[split] = stack.enter_context(qpath.open("wb"))
            target_handles[split] = stack.enter_context(tpath.open("wb"))
        for source_split in ["train", "val", "test"]:
            with (root / "data/processed/identity" / f"{source_split}_queries.jsonl").open("rb") as qf, \
                    (root / "data/targets/identity" / f"{source_split}.jsonl").open("rb") as tf:
                for qline, tline in zip_longest(qf, tf):
                    if qline is None or tline is None:
                        raise ValueError("Unpaired source queries and targets")
                    query, target = json.loads(qline), json.loads(tline)
                    qid = query["query_id"]
                    if target["query_id"] != qid:
                        raise ValueError("Mismatched query and target IDs")
                    if qid in seen:
                        raise ValueError("Query ID occurs in multiple identity partitions")
                    seen.add(qid)
                    if query["split"] != source_split:
                        raise ValueError("Source query partition disagrees with its file")
                    partition = assignments.get(query["solute_id"], "excluded")
                    if source_split == "test" and partition != "test":
                        raise AssertionError("Global test query changed partition")
                    if source_split != "test" and partition == "test":
                        raise AssertionError("New test query was introduced")
                    counts[partition][query["task"]] += 1
                    if partition == "excluded":
                        continue
                    if partition == source_split:
                        query_handles[partition].write(qline)
                    else:
                        query["split"] = partition
                        query_handles[partition].write((json.dumps(query, separators=(",", ":")) + "\n").encode())
                    target_handles[partition].write(tline)
    return {split: dict(count) for split, count in counts.items()}


def validate_real_conditions(root: Path, measurements: pd.DataFrame) -> dict:
    fields = ["record_id", "solute_id", "solvent_smiles", "source_doi", "temperature_k"]
    lookup = measurements[fields].set_index("record_id").to_dict("index")
    totals = Counter()
    for split in ["train", "val", "test"]:
        with (root / "data/processed/scaffold" / f"{split}_queries.jsonl").open() as handle:
            for line in handle:
                query = json.loads(line)
                if query["split"] != split:
                    raise AssertionError("Strict query in the wrong partition")
                matches = [lookup[candidate["record_id"]] for candidate in query["candidates"]]
                if any(m["solute_id"] != query["solute_id"] or m["temperature_k"] != query["temperature_k"] for m in matches):
                    raise AssertionError("Candidate measurements do not match the real query condition")
                if len({m["source_doi"] for m in matches}) != 1:
                    raise AssertionError("Candidate measurements have different sources")
                if len({candidate["record_id"] for candidate in query["candidates"]}) != len(matches):
                    raise AssertionError("Duplicate candidate measurements")
                if any(m["solvent_smiles"] != candidate["smiles"] for m, candidate in zip(matches, query["candidates"])):
                    raise AssertionError("Candidate solvent structure differs from its measurement")
                totals[split] += 1
    return {"passed": True, "queries_checked": dict(totals), "labels_used": False}


def build(root: Path, *, seed: int = 20261025, rebuild: bool = False) -> dict:
    root = root.resolve()
    audit_path = root / "artifacts/audit/scaffold_amendment.json"
    before = protected_hashes(root)
    global_before = observed_global_audit_hashes(root)
    previous = json.loads(audit_path.read_text()) if audit_path.exists() else None
    if previous:
        if before != previous["protected_identity_hashes_before"] or seed != previous["seed"]:
            raise ValueError("Protected identity inputs or strict-split seed changed; a new explicit protocol version is required")
        output_matches = all((root / path).exists() and sha256(root / path) == expected
                             for path, expected in previous["amended_artifact_hashes"].items())
        if output_matches and not rebuild:
            print(json.dumps({"status": "UNCHANGED", "counts": previous["counts"]}, indent=2), flush=True)
            return previous
        if not rebuild:
            raise ValueError("Amended outputs changed; --rebuild is required and must reproduce their exact frozen hashes")
    manifest_path = root / "data/splits/scaffold.csv"
    archive_path = root / "artifacts/audit/superseded_scaffold_manifest.csv"
    if archive_path.exists():
        old_sha = sha256(archive_path)
    else:
        old_sha = sha256(manifest_path)
    experiment_scan = previous["experiment_manifests_checked"] if previous else assert_no_superseded_experiments(root, old_sha)
    identity = pd.read_csv(root / "data/splits/identity.csv", dtype=str)
    amended, excluded = derive_manifest(identity, seed)
    measurements = pd.read_parquet(root / "data/processed/measurements.parquet")
    if set(measurements.solute_id) != set(identity.solute_id):
        raise ValueError("Identity measurement table and manifest have different solute universes")
    retained = measurements.loc[measurements.solute_id.isin(amended.solute_id)].copy()
    record_counts = measurements.groupby("solute_id").size()
    excluded["record_count"] = excluded.solute_id.map(record_counts)
    counts = {}
    for split in ["train", "val", "test"]:
        ids = set(amended.loc[amended.split.eq(split), "solute_id"])
        counts[split] = {"solutes": len(ids), "records": int(retained.solute_id.isin(ids).sum())}
    counts["excluded"] = {"solutes": len(excluded), "records": int(excluded.record_count.sum())}
    if counts != EXPECTED_CURRENT_COUNTS:
        raise ValueError(f"Unexpected strict-protocol counts; do not silently change the study: {counts}")
    if not archive_path.exists():
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest_path, archive_path)
    if sha256(archive_path) != old_sha:
        raise AssertionError("Superseded manifest was not preserved exactly")
    (root / ".tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="strict_split_", dir=root / ".tmp") as temp:
        staging = Path(temp)
        (staging / "data/splits").mkdir(parents=True)
        (staging / "data/processed").mkdir(parents=True)
        amended.to_csv(staging / "data/splits/scaffold.csv", index=False)
        excluded.to_csv(staging / "data/splits/scaffold_excluded.csv", index=False)
        retained.to_parquet(staging / "data/processed/measurements_scaffold.parquet", index=False)
        assignments = dict(zip(amended.solute_id, amended.split))
        query_counts = stream_tasks(root, staging, assignments)
        condition_checks = validate_real_conditions(staging, retained)
        staged_paths = sorted(path for path in staging.rglob("*") if path.is_file())
        hashes = {str(path.relative_to(staging)): sha256(path) for path in staged_paths}
        if previous and hashes != previous["amended_artifact_hashes"]:
            raise ValueError("Rebuild did not reproduce the frozen amendment; leaving existing output files untouched")
        if protected_hashes(root) != before:
            raise AssertionError("Protected identity inputs changed during amendment construction")
        for path in staged_paths:
            destination = root / path.relative_to(staging)
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
    after = protected_hashes(root)
    global_after = observed_global_audit_hashes(root)
    if before != after:
        raise AssertionError("Protected identity files were modified")
    if global_before != global_after:
        raise AssertionError("Global audit files changed during strict construction")
    for split in ["train", "val", "test", "excluded"]:
        counts[split]["queries"] = query_counts[split]
    result = {
        "status": "AMENDED_BEFORE_STRICT_SCIENTIFIC_USE",
        "protocol": "globally_protected_test_scaffold_extrapolation_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": seed,
        "reason": "The superseded independent scaffold partition could place identity-test solutes into another protocol's train/validation pool. The new strict protocol never uses global identity-test solutes in any development pool.",
        "superseded_status": "superseded-unused",
        "superseded_manifest": str(archive_path.relative_to(root)), "superseded_manifest_sha256": old_sha,
        "superseded_scientific_experiments_found": [], "experiment_manifests_checked": experiment_scan,
        "protected_identity_hashes_before": before, "protected_identity_hashes_after": after,
        "observed_global_audit_hashes_before": global_before,
        "observed_global_audit_hashes_after": global_after,
        "amended_artifact_hashes": hashes, "counts": counts,
        "assignment_policy": "Keep all identity-test solutes; exclude every development solute sharing any test scaffold_group; sort remaining scaffold groups by stable_id(seed,group) and assign by cumulative midpoint to 8/9 train and 1/9 val.",
        "label_based_selection": False, "test_metrics_computed": False,
        "overlap_checks": {"identity_test_in_strict_train_or_val": 0,
                           "scaffold_groups_crossing_strict_partitions": 0,
                           "parent_groups_crossing_strict_partitions": 0,
                           "identity_test_membership_preserved": True},
        "real_condition_checks": condition_checks,
        "training_size_caveat": "The strict protocol has a smaller training set. Differences from the identity protocol combine reduced training coverage and scaffold extrapolation; do not attribute them solely to structural novelty.",
        "measurements_path": "data/processed/measurements_scaffold.parquet",
        "fail_closed_contract": "Use scaffold.csv only with measurements_scaffold.parquet. The full measurements.parquet contains excluded solute IDs and must fail manifest-universe validation.",
        "rebuild_command": "PYTHONPATH=src .venv/bin/python -m solvent_select.strict_split --rebuild",
        "implementation_sha256": sha256(Path(__file__)),
    }
    if previous:
        # Preserve the original reservation timestamp and amendment history.
        result = previous
    else:
        temporary = audit_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(audit_path)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "overlap_checks": result["overlap_checks"],
                      "real_condition_checks": result["real_condition_checks"]}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    build(args.root, seed=args.seed, rebuild=args.rebuild)

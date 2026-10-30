"""Reserve a label-blind, overlap-safe SoluBench Task 1/2 supplement.

This module never builds SFT examples or adds benchmark answers to the main
query/target files. Native SoluBench uses log10(mol/L), not the main log10(x)
endpoint; the output is an audit/reservation manifest, not an evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
import pandas as pd

from .data import canonical_identity, exclusion_parent, normalize_doi, stable_id


REPOSITORY = "https://github.com/levakrasnovs/SoluBench"
REVISION = "00ad03f584a1764a8aeaf30aec66b3e584aa45c6"
FILES = {
    "data/task1_pairwise_solvent_comparison.jsonl": "3269d1e1bd6ad5fe074dc4e60c61d1824f21e921",
    "data/task2_best_solvent_selection.jsonl": "6d7b55151c2d34346a06e5dc8f9631f64ac52e6c",
    "task_generation.py": "04ade9ac3b070db57dcc8b64dff852381038a31a",
    "README.md": "7b60a7e6ff1fd974a5f8798bae1419187d9771ef",
}
TASK_PATHS = {
    "task1": "task1_pairwise_solvent_comparison.jsonl",
    "task2": "task2_best_solvent_selection.jsonl",
}
PUBLIC_KEYS = {
    "task1": {"id", "DOI", "SMILES", "Temperature_K", "Solvent_A", "Solvent_B"},
    "task2": {"id", "DOI", "SMILES", "Temperature_K", "All_solvents"},
}
LABEL_KEYS = {"Answer", "LogS_A", "LogS_B", "Delta_LogS", "Best_solvent", "Best_LogS",
              "Second_best_solvent", "Second_best_LogS", "Delta_LogS_best_minus_second", "All_LogS"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def download_files(destination: Path) -> dict:
    """Verify against blob IDs from the pinned author repository tree."""
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for remote, expected in FILES.items():
        path = destination / Path(remote).name
        url = f"https://raw.githubusercontent.com/levakrasnovs/SoluBench/{REVISION}/{remote}"
        if path.exists():
            payload = path.read_bytes()
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "solvent-select-research-audit/0.1"})
            last_error = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        payload = response.read()
                    break
                except Exception as error:
                    last_error = error
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Could not download {url}") from last_error
        actual = git_blob_sha1(payload)
        if actual != expected:
            raise ValueError(f"Pinned Git blob checksum mismatch for {remote}: {actual}")
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(path)
        manifest[remote] = {"url": url, "local_path": str(path), "bytes": len(payload),
                            "git_blob_sha1": actual, "sha256": hashlib.sha256(payload).hexdigest()}
    return manifest


def sanitize_task_row(row: dict, task: str) -> dict:
    """The only task fields consumed by overlap/condition eligibility logic."""
    missing = PUBLIC_KEYS[task] - row.keys()
    if missing:
        raise ValueError(f"Missing public task metadata: {missing}")
    result = {key: row[key] for key in PUBLIC_KEYS[task]}
    if LABEL_KEYS.intersection(result):
        raise AssertionError("Target-bearing fields passed the metadata allowlist")
    return result


def load_task_metadata(path: Path, task: str) -> tuple[list[dict], list[str]]:
    records, schemas = [], set()
    with path.open() as handle:
        for line in handle:
            raw = json.loads(line)
            schemas.update(raw)
            records.append(sanitize_task_row(raw, task))
    if len({row["id"] for row in records}) != len(records):
        raise ValueError(f"Duplicate qids in {task}")
    return records, sorted(schemas)


def load_source_metadata(path: Path) -> pd.DataFrame:
    """Only labels' availability is used to reconstruct published row selection.

    No numerical concentration or solubility value survives this projection.
    The published generator drops missing molarity labels before selecting T.
    """
    columns = ["SMILES_Solute", "Solvent", "SMILES_Solvent", "Temperature_K", "Source", "LogS(mol/L)"]
    frame = pd.read_csv(path, usecols=columns)
    frame["molar_label_available"] = frame["LogS(mol/L)"].notna()
    frame = frame.drop(columns="LogS(mol/L)")
    frame["source_doi"] = frame.Source.map(normalize_doi)
    frame["raw_row"] = np.arange(len(frame))
    return frame


def selected_source_rows(metadata: pd.DataFrame, task: str) -> dict[tuple[str, str, str], dict]:
    """Reconstruct source-temperature selection from metadata, without labels."""
    frame = metadata.loc[metadata.molar_label_available].copy()
    frame = frame.dropna(subset=["source_doi", "SMILES_Solute", "Solvent", "Temperature_K"])
    if task == "task1":
        frame["in_range"] = frame.Temperature_K.between(297, 301)
        frame["distance"] = abs(frame.Temperature_K - 299.0)
        frame = frame.sort_values(["source_doi", "SMILES_Solute", "Solvent", "in_range", "distance", "raw_row"],
                                  ascending=[True, True, True, False, True, True], kind="stable")
    else:
        frame = frame.loc[frame.Temperature_K.between(297, 301)].copy()
        frame["distance"] = abs(frame.Temperature_K - 298.15)
        frame = frame.sort_values(["source_doi", "SMILES_Solute", "Solvent", "distance", "raw_row"], kind="stable")
    frame = frame.drop_duplicates(["source_doi", "SMILES_Solute", "Solvent"])
    return {(r["source_doi"], r["SMILES_Solute"], r["Solvent"]):
            {"temperature_k": float(r["Temperature_K"]), "solvent_smiles": r["SMILES_Solvent"], "raw_row": int(r["raw_row"])}
            for r in frame.to_dict("records")}


def identify_solute(smiles: str) -> dict:
    canonical, molecule = canonical_identity(smiles)
    parent, _ = exclusion_parent(molecule)
    return {"solute_id": stable_id("solute", canonical), "solute_smiles": canonical,
            "leakage_group": stable_id("parent", parent)}


def candidate_names(row: dict, task: str) -> list[str]:
    if task == "task1":
        return [str(row["Solvent_A"]), str(row["Solvent_B"])]
    return str(row["All_solvents"]).split(";")


def condition_metadata(row: dict, task: str, selected: dict) -> dict:
    from rdkit import Chem

    names = candidate_names(row, task)
    source = normalize_doi(row["DOI"])
    candidates, reasons = [], []
    for name in names:
        match = selected.get((source, row["SMILES"], name))
        if match is None:
            reasons.append("source_candidate_not_reconstructed")
            candidates.append({"name": name, "solvent_smiles": None, "temperature_k": None})
            continue
        canonical = None
        try:
            canonical, molecule = canonical_identity(match["solvent_smiles"])
            if len(Chem.GetMolFrags(molecule)) != 1:
                reasons.append("not_unambiguous_pure_solvent")
                canonical = None
        except (ValueError, RuntimeError):
            reasons.append("invalid_or_polymeric_solvent_structure")
        candidates.append({"name": name, "solvent_smiles": canonical,
                           "temperature_k": match["temperature_k"], "raw_row": match["raw_row"]})
    temperatures = [c["temperature_k"] for c in candidates]
    all_known = all(t is not None and np.isfinite(t) for t in temperatures)
    exact_same_temperature = bool(all_known and len(set(temperatures)) == 1)
    declared_consistent = bool(exact_same_temperature and temperatures[0] == float(row["Temperature_K"]))
    if not exact_same_temperature:
        reasons.append("candidate_temperatures_not_identical_or_unknown")
    if exact_same_temperature and not declared_consistent:
        reasons.append("declared_temperature_differs_from_candidate_measurements")
    if len({c["solvent_smiles"] for c in candidates if c["solvent_smiles"] is not None}) != len(candidates):
        reasons.append("nonunique_or_unknown_candidate_identity")
    return {"candidates": candidates, "declared_temperature_k": float(row["Temperature_K"]),
            "exact_same_temperature": exact_same_temperature, "declared_temperature_consistent": declared_consistent,
            "condition_eligible": not reasons, "condition_exclusion_reasons": sorted(set(reasons))}


def reserve_metadata(rows: list[dict], task: str, manifest: pd.DataFrame,
                     selected: dict) -> tuple[list[dict], dict]:
    """Reserve only canonical main-test identities whose whole parent group is held out."""
    if manifest.groupby("leakage_group").split.nunique().max() != 1:
        raise ValueError("Main split already leaks a parent group")
    blocked = set(manifest.loc[manifest.split.isin(["train", "val"]), "leakage_group"])
    test_groups = set(manifest.loc[manifest.split.eq("test"), "leakage_group"])
    test_ids = set(manifest.loc[manifest.split.eq("test"), "solute_id"])
    cache, selections, excluded = {}, [], Counter()
    canonical_ids, parent_groups = set(), set()
    for row in rows:
        smiles = row["SMILES"]
        try:
            if smiles not in cache:
                cache[smiles] = identify_solute(smiles)
            identity = cache[smiles]
        except (ValueError, RuntimeError):
            excluded["invalid_or_out_of_scope_solute"] += 1
            continue
        canonical_ids.add(identity["solute_id"])
        parent_groups.add(identity["leakage_group"])
        if identity["leakage_group"] in blocked:
            excluded["train_or_validation_parent_group"] += 1
            continue
        if identity["leakage_group"] not in test_groups or identity["solute_id"] not in test_ids:
            excluded["not_a_curated_main_test_identity"] += 1
            continue
        selected_row = {"qid": row["id"], "task": task, **identity,
                        "source_doi": normalize_doi(row["DOI"]),
                        **condition_metadata(row, task, selected)}
        selections.append(selected_row)
    return selections, {"raw_questions": len(rows), "raw_unique_solute_smiles": len({r["SMILES"] for r in rows}),
                         "canonical_solutes": len(canonical_ids), "parent_groups": len(parent_groups),
                         "exclusions_by_reason": dict(excluded),
                         "overlap_safe_questions": len(selections),
                         "overlap_safe_solutes": len({r["solute_id"] for r in selections}),
                         "strict_condition_questions": sum(r["condition_eligible"] for r in selections),
                         "strict_condition_solutes": len({r["solute_id"] for r in selections if r["condition_eligible"]}),
                         "strict_condition_exclusions": dict(Counter(reason for r in selections for reason in r["condition_exclusion_reasons"]))}


def build(root: Path) -> dict:
    destination = root / "data/raw/solubench"
    downloads = download_files(destination)
    split_path = root / "data/splits/identity.csv"
    identities_path = root / "data/processed/solute_identities.parquet"
    source_path = root / "data/raw/BigSolDBv2.0.csv"
    manifest = pd.read_csv(split_path, dtype=str)
    identities = pd.read_parquet(identities_path, columns=["solute_id", "solute_smiles", "leakage_group"])
    if "leakage_group" not in manifest:
        manifest = manifest.merge(identities[["solute_id", "leakage_group"]], on="solute_id", validate="one_to_one")
    else:
        check = manifest.merge(identities[["solute_id", "leakage_group"]], on="solute_id", suffixes=("_split", "_identity"), validate="one_to_one")
        if not check.leakage_group_split.eq(check.leakage_group_identity).all():
            raise ValueError("Split and identity artifact disagree")
    source_meta = load_source_metadata(source_path)
    tasks, schemas, reserved, task_rows = {}, {}, {}, {}
    all_raw_smiles, all_parent_groups = set(), set()
    for task, filename in TASK_PATHS.items():
        rows, schemas[task] = load_task_metadata(destination / filename, task)
        task_rows[task] = rows
        all_raw_smiles.update(r["SMILES"] for r in rows)
        selected = selected_source_rows(source_meta, task)
        reserved[task], tasks[task] = reserve_metadata(rows, task, manifest, selected)
        for smiles in {row["SMILES"] for row in rows}:
            try:
                all_parent_groups.add(identify_solute(smiles)["leakage_group"])
            except (ValueError, RuntimeError):
                pass
    forbidden_groups = set(manifest.loc[manifest.split.isin(["train", "val"]), "leakage_group"])
    all_selected = [r for rows in reserved.values() for r in rows]
    if {r["leakage_group"] for r in all_selected} & forbidden_groups:
        raise AssertionError("Reserved supplement overlaps training or validation")
    result = {
        "status": "RESERVED_NOT_EVALUATED", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository": REPOSITORY, "revision": REVISION, "license": "CC-BY-4.0 (pinned README)",
        "downloads": downloads, "source_schemas": schemas,
        "input_fingerprints": {"identity_split_sha256": sha256(split_path),
                                "solute_identities_sha256": sha256(identities_path),
                                "BigSolDBv2.0_sha256": sha256(source_path),
                                "data_code_sha256": sha256(Path(__file__).with_name("data.py")),
                                "audit_code_sha256": sha256(Path(__file__))},
        "selection_policy": "canonical main identity-test solutes only; exclude all train+validation parent groups; no answer values used",
        "native_target": "log10_molar_concentration_mol_per_liter",
        "main_target": "log10_mole_fraction",
        "unit_compatible_with_main_predictions": False,
        "tasks": tasks,
        "overlap_safe_qids": {task: [r["qid"] for r in rows] for task, rows in reserved.items()},
        "strict_condition_qids": {task: [r["qid"] for r in rows if r["condition_eligible"]] for task, rows in reserved.items()},
        "reserved_metadata": reserved,
        "overlap_checks": {"train_validation_parent_group_intersection": [],
                           "selection_based_on_outcome_values": False,
                           "supplement_answers_added_to_main_training": False,
                           "selected_parent_groups": len({r["leakage_group"] for r in all_selected}),
                           "selected_solutes": len({r["solute_id"] for r in all_selected})},
        "reserve_all_tasks_counterfactual": {
            "task1_task2_union_raw_solute_smiles": len(all_raw_smiles),
            "remaining_BigSolDB_raw_solute_smiles": len(set(source_meta.SMILES_Solute) - all_raw_smiles),
            "remaining_curated_solutes_after_parent_exclusion": int((~manifest.leakage_group.isin(all_parent_groups)).sum()),
            "used_for_primary_training_design": False},
        "limitations": [
            "Both tasks derive native answers from author-converted log10(mol/L); a log10(x) model cannot be scored against them as if the units were identical.",
            "Task 1 chooses a per-solvent temperature near 299 K, preferring 297-301 K but allowing fallback outside that range; Task 2 independently chooses each solvent near 298.15 K within 297-301 K.",
            "The single supplied Temperature_K is copied from the best-solvent measurement; only the strict subset with equal reconstructed candidate temperatures avoids this label-derived condition ambiguity.",
            "Task 1 answers alternate by row index; qids, original row indices, Answer, LogS, best-solvent fields and canaries must never enter model inputs.",
            "The original benchmark selects relatively separated outcomes (Task 1 delta >=0.5 dex and Task 2 best-second >=0.1 dex), so this is not an unselected sample of all solvent choices.",
            "The supplement shares BigSolDB source data and main held-out solutes; it is a related benchmark subset, not independent external chemical coverage or a complete leaderboard run.",
            "No supplement performance was calculated. Any unit-adapted evaluation must be named explicitly and frozen before accessing its answers.",
        ],
    }
    path = root / "artifacts/audit/solubench_overlap.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ["revision", "input_fingerprints", "overlap_safe_qids", "strict_condition_qids"]:
            if previous[key] != result[key]:
                raise ValueError(f"Existing reserved supplement changed ({key}); preserve and explicitly version the audit instead of overwriting")
        print(json.dumps({"status": "UNCHANGED", "tasks": previous["tasks"]}, indent=2), flush=True)
        return previous
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps({"status": result["status"], "tasks": tasks,
                      "overlap_checks": result["overlap_checks"],
                      "reserve_all_tasks_counterfactual": result["reserve_all_tasks_counterfactual"]}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    build(args.root.resolve())

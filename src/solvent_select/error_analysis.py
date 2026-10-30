"""Predefined error slices with validation defaults and a frozen-test gate.

Structure annotations use label-free query/measurement metadata. Nearest-neighbor
similarity uses training structures only. Saved per-query evaluation metrics
provide outcomes. Optional recorded targets/predictions enrich cases; test inputs
require a status-FROZEN manifest authorizing the exact query-file hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import kendalltau
import yaml

METRICS = ("pair_accuracy", "nontie_pair_accuracy", "top1", "rank_tau", "regret", "covered")
SCORE_KEYS = ("query_id", "solute_id", "condition_id", "task", "candidate_count")
METADATA_COLUMNS = ("record_id", "solute_id", "solute_smiles", "solvent_smiles",
                    "temperature_k", "source_doi", "leakage_group")
FUNCTIONAL_GROUPS = ("carboxylic_acid", "ester", "amide", "amine", "alcohol", "phenol",
                     "nitrile", "halogen", "aromatic", "sulfonyl")
SOLVENT_GROUPS = ("water", "alcohol", "amide", "nitrile", "ester", "ketone", "ether",
                  "aromatic", "halogenated", "hydrocarbon", "other")
TASKS = ("pair", "rank")


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config.get("schema_version") != 1 or config.get("allowed_partition") != "val":
        raise ValueError("This protocol permits validation analysis only")
    if tuple(config["solute_functional_groups"]) != FUNCTIONAL_GROUPS:
        raise ValueError("Functional-group definitions must enumerate every predefined group")
    if tuple(config["solvent_group_order"]) != SOLVENT_GROUPS:
        raise ValueError("Solvent definitions must enumerate every predefined group")
    if set(config["solvent_smarts"]) != set(SOLVENT_GROUPS) - {"water", "hydrocarbon", "other"}:
        raise ValueError("Solvent SMARTS do not match the predefined class list")
    if config["temperature_boundaries_kelvin"] != [283.15, 303.15, 323.15]:
        raise ValueError("Temperature boundaries differ from the predefined protocol")
    if config["similarity_edges"] != [0., .2, .4, .6, .8, 1.]:
        raise ValueError("Similarity boundaries differ from the predefined protocol")
    if config["morgan"] != {"radius": 2, "fingerprint_size": 2048, "include_chirality": True}:
        raise ValueError("Morgan settings differ from the predefined protocol")
    if len(config["temperature_labels"]) != 4 or len(config["similarity_labels"]) != 5:
        raise ValueError("Incorrect number of interval labels")
    if config["minimum_independent_solutes"] != 10:
        raise ValueError("The minimum independent-solute count must be 10")
    if config["tie_dex"] != .1:
        raise ValueError("The tie threshold differs from the scoring protocol")
    if config["failure_examples"] != {"greatest_regret_per_task": 5, "hash_sample_errors_per_task": 5, "hash_seed": 20261025}:
        raise ValueError("Case selection differs from the predefined protocol")
    return config


def molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid retained structure: {smiles}")
    return mol


def compile_patterns(patterns):
    result = {name: Chem.MolFromSmarts(pattern) for name, pattern in patterns.items()}
    if any(pattern is None for pattern in result.values()):
        raise ValueError("Invalid SMARTS definition")
    return result


def matched_groups(mol, patterns):
    return [name for name, pattern in patterns.items() if mol.HasSubstructMatch(pattern)]


def solvent_groups(mol, patterns):
    labels = set(matched_groups(mol, patterns))
    if Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) == "O":
        labels.add("water")
    elements = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    if 6 in elements and set(elements) <= {1, 6}:
        labels.add("hydrocarbon")
    if not labels:
        labels.add("other")
    return [name for name in SOLVENT_GROUPS if name in labels]


def interval_label(value, edges, labels, bounded=False):
    if not np.isfinite(value):
        raise ValueError("Nonfinite interval value")
    if bounded and not edges[0] <= value <= edges[-1]:
        raise ValueError("Similarity must lie in [0, 1]")
    boundaries = edges[1:-1] if bounded else edges
    return labels[int(np.searchsorted(boundaries, value, side="right"))]


def authorize_partition(queries, queries_path, config_path, freeze_manifest=None):
    partitions = {query.get("split") for query in queries}
    if partitions == {"val"}:
        return "val"
    if partitions != {"test"}:
        raise ValueError("Only one explicitly labelled validation or frozen-test query set is allowed")
    if freeze_manifest is None or not Path(freeze_manifest).is_file():
        raise ValueError("Test analysis requires a FROZEN manifest before scores or targets are read")
    freeze = json.loads(Path(freeze_manifest).read_text())
    if freeze.get("status") != "FROZEN" or freeze.get("query_sha256") != sha256(queries_path):
        raise ValueError("Test freeze status or query hash is not authorized")
    for key, path in (("error_analysis_config_sha256", config_path), ("error_analysis_code_sha256", __file__)):
        if key in freeze and freeze[key] != sha256(path):
            raise ValueError(f"Frozen error-analysis implementation changed: {key}")
    return "test"


def validate_inputs(queries, per_query, metadata, manifest, permitted_partition="val"):
    if not queries or any(q.get("split") != permitted_partition for q in queries):
        raise ValueError("Queries do not match the authorized partition")
    if manifest.solute_id.duplicated().any() or metadata.record_id.duplicated().any():
        raise ValueError("Duplicate manifest or measurement identities")
    if not set(manifest.split) <= {"train", "val", "test"} or "train" not in set(manifest.split):
        raise ValueError("Invalid split manifest")
    if set(manifest.solute_id) != set(metadata.solute_id):
        raise ValueError("Measurement universe does not match the supplied split manifest")
    if manifest.groupby("leakage_group").split.nunique().max() != 1:
        raise ValueError("Parent groups cross the supplied split boundary")
    if metadata.groupby("solute_id").solute_smiles.nunique().max() != 1:
        raise ValueError("One solute ID has multiple structures")
    identity = metadata[["solute_id", "leakage_group"]].drop_duplicates()
    merged = identity.merge(manifest[["solute_id", "leakage_group"]], on="solute_id", validate="one_to_one")
    if not (merged.leakage_group_x == merged.leakage_group_y).all():
        raise ValueError("Parent groups disagree between manifest and measurements")
    query_ids = [q["query_id"] for q in queries]
    if len(set(query_ids)) != len(query_ids) or per_query.query_id.duplicated().any():
        raise ValueError("Repeated query IDs")
    if set(query_ids) != set(per_query.query_id):
        raise ValueError("Per-query scores must exactly match the requested query cohort")
    split_by_id = manifest.set_index("solute_id").split.to_dict()
    records = metadata.set_index("record_id")
    scores = per_query.set_index("query_id")
    conditions, inverse_conditions = {}, {}
    for query in queries:
        if split_by_id.get(query["solute_id"]) != permitted_partition:
            raise ValueError("A query solute is not validation/authorized test in the supplied manifest")
        if query["task"] not in TASKS:
            raise ValueError("Unknown task")
        if not query.get("condition_id"):
            raise ValueError("Physical condition IDs are required")
        candidates = query["candidates"]
        if len({c["id"] for c in candidates}) != len(candidates) or len({c["record_id"] for c in candidates}) != len(candidates):
            raise ValueError("Repeated candidate identity")
        if len(candidates) < 2 or len(candidates) > 8 or (query["task"] == "pair" and len(candidates) != 2):
            raise ValueError("Candidate count outside the fixed task scope")
        source_dois = set()
        for candidate in candidates:
            record = records.loc[candidate["record_id"]]
            if (record.solute_id != query["solute_id"] or record.solute_smiles != query["solute_smiles"]
                    or record.solvent_smiles != candidate["smiles"] or record.temperature_k != query["temperature_k"]):
                raise ValueError("Candidate/query measurement metadata disagree")
            source_dois.add(record.source_doi)
        if len(source_dois) != 1:
            raise ValueError("Candidates do not share one source DOI")
        physical = (query["solute_id"], next(iter(source_dois)), query["temperature_k"])
        key = (query["solute_id"], query["condition_id"])
        if conditions.setdefault(key, physical) != physical or inverse_conditions.setdefault(physical, key) != key:
            raise ValueError("Condition IDs do not bijectively identify physical conditions")
        row = scores.loc[query["query_id"]]
        if any(row[field] != query[field] for field in ("solute_id", "condition_id", "task")):
            raise ValueError("Score/query identity metadata disagree")
        if row.candidate_count != len(candidates):
            raise ValueError("Score/query candidate counts disagree")
        primary = "pair_accuracy" if query["task"] == "pair" else "top1"
        if row[primary] not in (0., 1.) or row.covered not in (0., 1.):
            raise ValueError("Invalid correctness or coverage value")
        if not np.isfinite(row.regret) or row.regret < 0:
            raise ValueError("Invalid regret value")
        for metric in METRICS:
            value = row[metric]
            if pd.notna(value) and not np.isfinite(value):
                raise ValueError("Nonfinite metric value")
    return records


def verify_query_partition_before_scores(queries, manifest, partition):
    """Do not trust the split label in a query without checking the manifest."""
    if manifest.solute_id.duplicated().any():
        raise ValueError("Duplicate manifest identities")
    assigned = manifest.set_index("solute_id").split.to_dict()
    if any(assigned.get(query["solute_id"]) != partition for query in queries):
        raise ValueError("Query membership is not authorized by the split manifest; scores were not read")


def read_authorized_scores(path, queries):
    """Inspect only score-file identifiers before deserializing any outcomes."""
    identifiers = pd.read_parquet(path, columns=list(SCORE_KEYS))
    expected = {query["query_id"]: query for query in queries}
    if len(expected) != len(queries) or identifiers.query_id.duplicated().any() or set(identifiers.query_id) != set(expected):
        raise ValueError("Score cohort is not authorized; outcome columns were not read")
    for row in identifiers.itertuples():
        query = expected[row.query_id]
        if any(getattr(row, field) != query[field] for field in ("solute_id", "condition_id", "task")) or row.candidate_count != len(query["candidates"]):
            raise ValueError("Score identity metadata disagree; outcome columns were not read")
    return pd.read_parquet(path, columns=[*SCORE_KEYS, *METRICS])


def annotate_queries(queries, metadata, manifest, config):
    train_ids = set(manifest.loc[manifest.split == "train", "solute_id"])
    train = metadata[metadata.solute_id.isin(train_ids)].drop_duplicates("solute_id").sort_values("solute_id")
    if train.empty:
        raise ValueError("No training structures available for similarity")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=config["morgan"]["radius"], fpSize=config["morgan"]["fingerprint_size"],
        includeChirality=config["morgan"]["include_chirality"])
    train_fps = [generator.GetFingerprint(molecule(s)) for s in train.solute_smiles]
    train_id_order = train.solute_id.tolist()
    functional_patterns = compile_patterns(config["solute_functional_groups"])
    solvent_patterns = compile_patterns(config["solvent_smarts"])
    solutes, solvents = {}, {}
    rows = []
    for query in queries:
        sid = query["solute_id"]
        if sid not in solutes:
            mol = molecule(query["solute_smiles"])
            similarities = DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(mol), train_fps)
            neighbor = int(np.argmax(similarities))
            score = float(similarities[neighbor])
            solutes[sid] = {"nearest_train_solute_id": train_id_order[neighbor], "train_similarity": score,
                            "similarity_bin": interval_label(score, config["similarity_edges"], config["similarity_labels"], True),
                            "functional_groups": matched_groups(mol, functional_patterns)}
        classes = set()
        for candidate in query["candidates"]:
            smiles = candidate["smiles"]
            if smiles not in solvents:
                solvents[smiles] = solvent_groups(molecule(smiles), solvent_patterns)
            classes.update(solvents[smiles])
        rows.append({"query_id": query["query_id"], "temperature_k": query["temperature_k"],
                     "temperature_bin": interval_label(query["temperature_k"], config["temperature_boundaries_kelvin"], config["temperature_labels"]),
                     **solutes[sid], "solvent_classes": [name for name in SOLVENT_GROUPS if name in classes]})
    return pd.DataFrame(rows), {"train_solute_count": len(train), "query_solute_count": len(solutes),
                                "train_solute_ids_sha256": hashlib.sha256("\n".join(train_id_order).encode()).hexdigest()}


def summarize_slices(details, config):
    definitions = [("all", "all", pd.Series(True, index=details.index), False)]
    for family, column, labels in (("temperature", "temperature_bin", config["temperature_labels"]),
                                   ("train_similarity", "similarity_bin", config["similarity_labels"])):
        definitions.extend((family, label, details[column] == label, False) for label in labels)
    for family, column, labels in (("solute_functional_group", "functional_groups", FUNCTIONAL_GROUPS),
                                   ("candidate_solvent_class", "solvent_classes", SOLVENT_GROUPS)):
        definitions.extend((family, label, details[column].map(lambda values: label in values), True) for label in labels)
    summaries, solute_rows = [], []
    for family, label, mask, overlapping in definitions:
        for task in TASKS:
            selected = details[mask & (details.task == task)]
            conditions = selected.groupby(["solute_id", "condition_id"], sort=True)[list(METRICS)].mean()
            solutes = conditions.groupby("solute_id").mean()
            row = {"slice_family": family, "slice": label, "task": task,
                   "queries": len(selected), "physical_conditions": len(conditions), "solutes": len(solutes),
                   "status": "SUFFICIENT_FOR_DESCRIPTION" if len(solutes) >= config["minimum_independent_solutes"] else "INSUFFICIENT_SOLUTES",
                   "overlapping_multilabel_slice": overlapping}
            for metric in METRICS:
                row[metric] = float(solutes[metric].mean()) if solutes[metric].notna().any() else None
                row[f"{metric}_support_solutes"] = int(solutes[metric].notna().sum())
            summaries.append(row)
            for sid, values in solutes.iterrows():
                solute_rows.append({"slice_family": family, "slice": label, "task": task, "solute_id": sid,
                                    "physical_conditions": int((conditions.index.get_level_values("solute_id") == sid).sum()),
                                    **values.to_dict()})
    return pd.DataFrame(summaries), pd.DataFrame(solute_rows)


def select_cases(details, queries, records, config):
    by_id = {query["query_id"]: query for query in queries}
    settings = config["failure_examples"]
    cases = []
    for task in TASKS:
        task_rows = details[details.task == task].copy()
        primary = "pair_accuracy" if task == "pair" else "top1"
        task_rows["is_error"] = (task_rows[primary] < 1) | (task_rows.covered < 1)
        task_rows["selection_hash"] = task_rows.query_id.map(
            lambda qid: hashlib.sha256(f"{settings['hash_seed']}|{task}|{qid}".encode()).hexdigest())
        worst = task_rows.sort_values(["regret", "query_id"], ascending=[False, True]).head(settings["greatest_regret_per_task"])
        sampled = task_rows[task_rows.is_error].sort_values(["selection_hash", "query_id"]).head(settings["hash_sample_errors_per_task"])
        for strategy, selected in (("greatest_regret", worst), ("hash_sample_error", sampled)):
            for rank, (_, row) in enumerate(selected.iterrows(), start=1):
                query = by_id[row.query_id]
                candidates = [{"id": c["id"], "smiles": c["smiles"], "record_id": c["record_id"],
                               "source_doi": records.loc[c["record_id"], "source_doi"]} for c in query["candidates"]]
                cases.append({"task": task, "selection": strategy, "selection_rank": rank,
                              "selection_hash": row.selection_hash, "query_id": row.query_id,
                              "solute_id": query["solute_id"], "solute_smiles": query["solute_smiles"],
                              "condition_id": query["condition_id"], "temperature_k": query["temperature_k"],
                              "source_doi": candidates[0]["source_doi"], "candidates": candidates,
                              "candidate_record_ids": [c["record_id"] for c in candidates],
                              "pH": None, "solid_form": None, "equilibration_protocol": None,
                              "unrecorded_condition_note": "Unavailable in the curated metadata; not inferred.",
                              "is_primary_metric_error": bool(row.is_error),
                              "train_similarity": row.train_similarity,
                              "nearest_train_solute_id": row.nearest_train_solute_id,
                              "functional_groups": row.functional_groups, "solvent_classes": row.solvent_classes,
                              **{metric: row[metric] for metric in METRICS}})
    return cases


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def enrich_cases(cases, queries, metadata, manifest, targets_path, predictions_path, tie_dex, partition):
    """Attach recorded comparisons and check they reproduce stored case outcomes."""
    query_by_id = {query["query_id"]: query for query in queries}
    targets = read_jsonl(targets_path)
    target_by_id = {row["query_id"]: row for row in targets}
    if len(targets) != len(target_by_id) or set(target_by_id) != set(query_by_id):
        raise ValueError("Case targets must exactly match the authorized query cohort")
    for query in queries:
        values = target_by_id[query["query_id"]]["values"]
        if set(values) != {c["id"] for c in query["candidates"]} or not all(
                isinstance(v, (float, int)) and np.isfinite(v) for v in values.values()):
            raise ValueError("Case targets have invalid candidate values")
    property_predictions = Path(predictions_path).suffix == ".parquet"
    if property_predictions:
        predictions = pd.read_parquet(predictions_path)
        allowed_solutes = set(manifest.loc[manifest.split == partition, "solute_id"])
        allowed_records = set(metadata.loc[metadata.solute_id.isin(allowed_solutes), "record_id"])
        if predictions.record_id.duplicated().any() or not set(predictions.record_id) <= allowed_records:
            raise ValueError("Property predictions contain duplicate or unauthorized-partition record IDs")
        if "split" in predictions and not (predictions.split == partition).all():
            raise ValueError("Property prediction partition disagrees with authorized partition")
        predicted_by_record = predictions.set_index("record_id").pred_logx.to_dict()
    else:
        predictions = read_jsonl(predictions_path)
        predicted_by_query = {row["query_id"]: row for row in predictions}
        if len(predictions) != len(predicted_by_query) or not set(predicted_by_query) <= set(query_by_id):
            raise ValueError("LLM predictions contain duplicate or unauthorized query IDs")
    for case in cases:
        query = query_by_id[case["query_id"]]
        ids = [candidate["id"] for candidate in query["candidates"]]
        target = target_by_id[case["query_id"]]
        actual = target["values"]
        if property_predictions:
            values = {c["id"]: predicted_by_record.get(c["record_id"], np.nan) for c in query["candidates"]}
            parsed = {"predicted_values": values}
            if all(np.isfinite(v) for v in values.values()):
                if query["task"] == "pair":
                    delta = values[ids[0]] - values[ids[1]]
                    parsed["choice"] = "TIE" if abs(delta) <= tie_dex + 1e-12 else ids[0] if delta > 0 else ids[1]
                else:
                    parsed["ranking"] = sorted(values, key=lambda key: (-values[key], key))
            raw = parsed
        else:
            raw = predicted_by_query.get(case["query_id"], {})
            parsed = raw
            for key in ("parsed", "parsed_output", "prediction"):
                if isinstance(raw.get(key), dict):
                    parsed = raw[key]
                    break
            values = parsed.get("predicted_values", {})
            values = values if isinstance(values, dict) else {}
        selected, valid = None, False
        expected_metrics = {metric: np.nan for metric in METRICS}
        if query["task"] == "pair":
            choice = parsed.get("choice")
            valid = isinstance(choice, str) and choice in set(ids) | {"TIE"}
            if valid:
                selected = min(query["candidates"], key=lambda c: c["smiles"])["id"] if choice == "TIE" else choice
            correct = float(valid and choice == target["choice"])
            delta = actual[ids[0]] - actual[ids[1]]
            measured_choice = "TIE" if abs(delta) <= tie_dex + 1e-12 else ids[0] if delta > 0 else ids[1]
            if target["choice"] != measured_choice:
                raise ValueError("Target choice contradicts the measured candidate values")
            expected_metrics.update(pair_accuracy=correct,
                                    nontie_pair_accuracy=correct if measured_choice != "TIE" else np.nan)
        else:
            ranking = parsed.get("ranking")
            valid = (isinstance(ranking, list) and all(isinstance(cid, str) for cid in ranking)
                     and len(ranking) == len(ids) and set(ranking) == set(ids))
            if valid:
                selected = ranking[0]
            correct = float(valid and max(actual.values()) - actual[selected] <= tie_dex + 1e-12)
            tau = -1.
            if valid:
                positions = {cid: -index for index, cid in enumerate(ranking)}
                tau = kendalltau([actual[cid] for cid in ids], [positions[cid] for cid in ids]).statistic
            expected_metrics.update(top1=correct, rank_tau=tau)
        expected_regret = max(actual.values()) - (actual[selected] if valid else min(actual.values()))
        expected_metrics.update(covered=float(valid), regret=expected_regret)
        for metric, expected in expected_metrics.items():
            observed = case[metric]
            equal = pd.isna(expected) and pd.isna(observed)
            if not equal and not (pd.notna(expected) and pd.notna(observed)
                                  and np.isclose(expected, observed, atol=1e-12, rtol=0)):
                raise ValueError(f"Recorded candidate decision does not reproduce saved {metric}")
        best = max(actual.values())
        case.update(actual_best_candidate_ids=sorted(cid for cid, value in actual.items() if value == best),
                    within_tolerance_best_candidate_ids=sorted(cid for cid, value in actual.items() if best-value <= tie_dex+1e-12),
                    actual_choice=target.get("choice"), actual_order=sorted(actual, key=lambda cid: (-actual[cid], cid)),
                    model_choice=parsed.get("choice"), model_ranking=parsed.get("ranking"),
                    model_selected_candidate_id=selected, raw_prediction=raw,
                    experimental_endpoint="log10(mole_fraction)",
                    interpretation="Recorded measured order and model decision only; no causal chemical mechanism inferred.")
        for candidate in case["candidates"]:
            candidate["actual_logx"] = actual[candidate["id"]]
            candidate["predicted_logx"] = values.get(candidate["id"])
    return cases


def run_analysis(queries_path, per_query_path, measurements_path, manifest_path, config_path, output,
                 targets_path=None, predictions_path=None, freeze_manifest=None):
    started = time.perf_counter()
    config = load_config(config_path)
    queries = read_jsonl(queries_path)
    # The test gate runs before opening scores, targets, or measurement metadata.
    partition = authorize_partition(queries, queries_path, config_path, freeze_manifest)
    if (targets_path is None) != (predictions_path is None):
        raise ValueError("Case enrichment requires both targets and predictions")
    manifest = pd.read_csv(manifest_path, dtype={"solute_id": str, "leakage_group": str})
    verify_query_partition_before_scores(queries, manifest, partition)
    per_query = read_authorized_scores(per_query_path, queries)
    # Column projection is deliberate: experimental logx, x, derived concentration,
    # and target fields are never deserialized by this analysis.
    metadata = pd.read_parquet(measurements_path, columns=list(METADATA_COLUMNS))
    records = validate_inputs(queries, per_query, metadata, manifest, partition)
    annotations, similarity_audit = annotate_queries(queries, metadata, manifest, config)
    details = per_query.merge(annotations, on="query_id", validate="one_to_one")
    summary, per_solute = summarize_slices(details, config)
    cases = select_cases(details, queries, records, config)
    if targets_path is not None:
        cases = enrich_cases(cases, queries, metadata, manifest, targets_path, predictions_path, config["tie_dex"], partition)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Analysis output is not empty; use a new versioned output directory")
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "slice_metrics.csv", index=False)
    per_solute.to_csv(output / "slice_per_solute.csv", index=False)
    annotations.to_parquet(output / "query_annotations.parquet", index=False)
    with (output / "failure_cases.jsonl").open("w") as stream:
        for case in cases:
            stream.write(json.dumps(json_safe(case), allow_nan=False) + "\n")
    case_csv = pd.DataFrame(cases)
    for column in ("candidates", "candidate_record_ids", "functional_groups", "solvent_classes",
                   "actual_best_candidate_ids", "within_tolerance_best_candidate_ids", "actual_order",
                   "model_ranking", "raw_prediction"):
        if column in case_csv:
            case_csv[column] = case_csv[column].map(lambda value: json.dumps(json_safe(value), allow_nan=False))
    case_csv.to_csv(output / "failure_cases.csv", index=False)
    result = {"status": "VALIDATION_DIAGNOSTICS_ONLY" if partition == "val" else "FROZEN_TEST_DIAGNOSTICS", "protocol": config["protocol"],
              "partition": partition, "queries": len(queries), "solutes": int(per_query.solute_id.nunique()),
              "independence_unit": "solute", "hierarchy": ["physical_condition", "solute", "population"],
              "similarity_audit": similarity_audit, "definitions": config["definitions"],
              "functional_group_smarts": config["solute_functional_groups"], "solvent_smarts": config["solvent_smarts"],
              "solvent_rule_definitions": config["solvent_rule_definitions"],
              "failure_selection": config["failure_examples"], "all_predefined_slices_reported": True,
              "case_content": "Recorded targets and predictions; selected-case metrics reproduced" if targets_path else "Traceable case indices only; measured candidate values and model choices not supplied",
              "test_labels_read": partition == "test" and targets_path is not None,
              "test_analysis_performed": partition == "test",
              "confidence_intervals": "Not calculated; slices are descriptive and can overlap.",
              "elapsed_seconds": time.perf_counter() - started,
              "inputs": {str(Path(path).resolve()): sha256(path) for path in
                         (queries_path, per_query_path, measurements_path, manifest_path, config_path, __file__,
                          *([targets_path, predictions_path] if targets_path else []),
                          *([freeze_manifest] if freeze_manifest else []))},
              "measurement_columns_read": list(METADATA_COLUMNS),
              "slices": summary.to_dict("records"),
              "outputs": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "analysis.json").write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--per-query", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/error_analysis.yaml"))
    parser.add_argument("--targets", type=Path, help="Optional authorized-cohort targets for interpretable cases")
    parser.add_argument("--predictions", type=Path, help="Optional property parquet or LLM JSONL; requires --targets")
    parser.add_argument("--freeze-manifest", type=Path, help="Test only: status FROZEN and exact query_sha256 required")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_analysis(args.queries, args.per_query, args.measurements, args.manifest, args.config, args.output,
                          args.targets, args.predictions, args.freeze_manifest)
    print(json.dumps({"status": result["status"], "queries": result["queries"], "solutes": result["solutes"],
                      "reported_slice_rows": len(result["slices"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

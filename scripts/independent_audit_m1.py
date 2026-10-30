#!/usr/bin/env python3
"""Independent M1 checks of native units, isolation and validation metrics.

Numerical/model checks use ONLY the identity validation partition. No test
target file is opened and no test predictions are produced. All-partition
inspection is restricted to identity/condition metadata and query counts.
The evaluator is imported only for explicitly synthetic scoring counterexamples;
the empirical metric and bootstrap recomputations do not call its functions.
"""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

pa.set_cpu_count(2)
pa.set_io_thread_count(2)
RUNS = ("identity_nearest", "identity_lgbm_17", "identity_lgbm_42", "identity_lgbm_2026")
LEARNING_RUNS = RUNS[1:]


def digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit(strict=False):
    input_paths = [Path(__file__), ROOT / "data/splits/identity.csv",
                   ROOT / "data/processed/measurements.parquet", ROOT / "data/raw/BigSolDBv2.0.csv",
                   ROOT / "data/processed/solute_identities.parquet",
                   ROOT / "data/processed/identity/val_queries.jsonl", ROOT / "data/targets/identity/val.jsonl",
                   ROOT / "artifacts/reports/generated/metrics_by_method.csv"]
    manifest = pd.read_csv(ROOT / "data/splits/identity.csv")
    train_ids = set(manifest.loc[manifest.split == "train", "solute_id"])
    val_ids = set(manifest.loc[manifest.split == "val", "solute_id"])
    train = pd.read_parquet(ROOT / "data/processed/measurements.parquet", filters=[("solute_id", "in", list(train_ids))])
    val = pd.read_parquet(ROOT / "data/processed/measurements.parquet", filters=[("solute_id", "in", list(val_ids))])
    identity = pd.read_parquet(ROOT / "data/processed/solute_identities.parquet").merge(
        manifest[["solute_id", "split"]], on="solute_id", validate="one_to_one")
    metadata = pd.read_parquet(ROOT / "data/processed/measurements.parquet", columns=[
        "record_id", "solute_id", "source_doi", "temperature_k", "scaffold_group", "leakage_group"])
    assert not manifest.solute_id.duplicated().any()
    assert identity.groupby("leakage_group").split.nunique().max() == 1
    wanted_raw_rows = sorted({number for value in val.raw_rows for number in json.loads(value)})
    wanted_set = set(wanted_raw_rows)
    # Only validation source rows are parsed into numeric arrays.
    raw = pd.read_csv(ROOT / "data/raw/BigSolDBv2.0.csv",
                      skiprows=lambda n: n > 0 and n - 1 not in wanted_set,
                      usecols=["SMILES_Solute", "SMILES_Solvent", "Temperature_K", "Source",
                               "Solubility(mole_fraction)", "LogS(mol/L)"])
    assert len(raw) == len(wanted_raw_rows)
    raw.index = wanted_raw_rows
    index = val.set_index("record_id")
    errors, mol_values = [], {}
    for record in val.itertuples():
        original = raw.loc[json.loads(record.raw_rows)]
        expected = float(np.median(np.log10(original["Solubility(mole_fraction)"].to_numpy(float))))
        errors.append(abs(expected - record.logx))
        assert all(float(t) == record.temperature_k for t in original.Temperature_K)
        assert all(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=True)
                   == record.solute_smiles for s in original.SMILES_Solute)
        observed_molar = original["LogS(mol/L)"].dropna()
        if len(observed_molar) == len(original):
            mol_values[record.record_id] = float(observed_molar.median())
    assert max(errors) < 1e-12
    queries = read_jsonl(ROOT / "data/processed/identity/val_queries.jsonl")
    targets = {row["query_id"]: row for row in read_jsonl(ROOT / "data/targets/identity/val.jsonl")}
    assert all(q["split"] == "val" and q["solute_id"] in val_ids for q in queries)
    condition_map, physical_map = defaultdict(set), defaultdict(set)
    reversals, mol_comparable = [], 0
    for query in queries:
        records = [index.loc[c["record_id"]] for c in query["candidates"]]
        assert {r.solute_id for r in records} == {query["solute_id"]}
        assert len({r.source_doi for r in records}) == 1
        assert {r.temperature_k for r in records} == {query["temperature_k"]}
        physical = (query["solute_id"], records[0].source_doi, query["temperature_k"])
        condition_map[query["condition_id"]].add(physical)
        physical_map[physical].add(query["condition_id"])
        for candidate, record in zip(query["candidates"], records):
            assert abs(targets[query["query_id"]]["values"][candidate["id"]] - record.logx) < 1e-12
        if query["task"] == "pair" and all(c["record_id"] in mol_values for c in query["candidates"]):
            mol_comparable += 1
            a, b = query["candidates"]
            dx = targets[query["query_id"]]["values"]["A"] - targets[query["query_id"]]["values"]["B"]
            dc = mol_values[a["record_id"]] - mol_values[b["record_id"]]
            if dx * dc < 0 and abs(dx) > .1 and abs(dc) > .1:
                reversals.append({"query_id": query["query_id"], "solute_smiles": query["solute_smiles"],
                                  "temperature_k": query["temperature_k"], "source_doi": records[0].source_doi,
                                  "candidate_smiles": [a["smiles"], b["smiles"]],
                                  "logx": [records[0].logx, records[1].logx],
                                  "author_logmolL": [mol_values[a["record_id"]], mol_values[b["record_id"]]]})
    assert max(map(len, condition_map.values())) == max(map(len, physical_map.values())) == 1
    result = {
        "status": "PASSED_VALIDATION_AUDIT", "scope": "identity validation only; not final test evaluation",
        "max_cpu_threads": 2, "validation_raw_roundtrip_max_error": max(errors),
        "validation_records_checked": len(val), "validation_queries_checked": len(queries),
        "physical_condition_mapping_bijective": True,
        "unit_scale_audit": {"purpose": "Unit audit, not method selection or performance evaluation",
                             "comparable_validation_pairs": mol_comparable,
                             "order_reversals_exceeding_0_1_dex_on_both_scales": len(reversals),
                             "first_reversal": reversals[0] if reversals else None,
                             "molar_values_source": "Author-derived LogS(mol/L), not independently measured concentration"},
        "independent_metrics": {},
    }
    manual_per_solute = {}
    for run in RUNS:
        folder = ROOT / "artifacts/runs" / run
        pred_path, metric_path = folder / "predictions_val.parquet", folder / "evaluation/metrics.json"
        input_paths.extend([pred_path, metric_path])
        pred = pd.read_parquet(pred_path)
        values = dict(zip(pred.record_id, pred.pred_logx))
        assert len(values) == len(pred) and set(values) == set(val.record_id)
        assert set(pred.split) == {"val"}
        condition, question_acc = defaultdict(list), []
        if run == "identity_nearest":
            train_index = train.set_index("record_id")
            assert set(pred.nearest_record_id) <= set(train_index.index)
            neighbors = np.array([train_index.loc[k].logx for k in pred.nearest_record_id])
            result["nearest_label_provenance_max_error"] = float(np.max(np.abs(neighbors - pred.pred_logx.to_numpy())))
            assert result["nearest_label_provenance_max_error"] == 0
            result["nearest_provenance_all_training"] = True
        for query in queries:
            y = targets[query["query_id"]]["values"]
            p = {c["id"]: values[c["record_id"]] for c in query["candidates"]}
            if query["task"] == "pair":
                a, b = p["A"], p["B"]
                label = "TIE" if abs(a - b) <= .1 + 1e-12 else ("A" if a > b else "B")
                truth = "TIE" if abs(y["A"] - y["B"]) <= .1 + 1e-12 else ("A" if y["A"] > y["B"] else "B")
                assert truth == targets[query["query_id"]]["choice"]
                correct = float(label == truth)
                selected = label if label != "TIE" else min(query["candidates"], key=lambda c: c["smiles"])["id"]
                condition[(query["solute_id"], query["condition_id"], "pair", "pair_accuracy")].append(correct)
                question_acc.append(correct)
            else:
                selected = sorted(p, key=lambda c: (-p[c], c))[0]
                condition[(query["solute_id"], query["condition_id"], "rank", "top1")].append(float(max(y.values()) - y[selected] <= .1 + 1e-12))
            condition[(query["solute_id"], query["condition_id"], query["task"], "regret")].append(max(y.values()) - y[selected])
        solute = defaultdict(list)
        for (sid, cid, task, metric), vals in condition.items():
            solute[(sid, task, metric)].append(sum(vals) / len(vals))
        aggregated, by_solute = defaultdict(list), {}
        for (sid, task, metric), vals in solute.items():
            value = sum(vals) / len(vals)
            aggregated[(task, metric)].append(value)
            by_solute[(sid, task, metric)] = value
        measured = json.loads(metric_path.read_text())
        independent = {}
        for (task, metric), vals in aggregated.items():
            value = sum(vals) / len(vals)
            assert abs(value - measured["by_task"][task][metric]) < 1e-12
            independent[f"{task}/{metric}"] = value
        independent["naive_question_pair_accuracy_for_weighting_diagnostic"] = sum(question_acc) / len(question_acc)
        result["independent_metrics"][run] = independent
        manual_per_solute[run] = by_solute
    summary = pd.read_csv(ROOT / "artifacts/reports/generated/metrics_by_method.csv")
    ci_differences = []
    for task, metric in (("pair", "pair_accuracy"), ("rank", "top1"), ("rank", "regret")):
        ids = sorted(key[0] for key in manual_per_solute[LEARNING_RUNS[0]] if key[1:] == (task, metric))
        array = np.array([sum(manual_per_solute[run][(sid, task, metric)] for run in LEARNING_RUNS) / 3 for sid in ids])
        rng = np.random.default_rng(20261025)
        estimates = array[rng.integers(0, len(array), (10000, len(array)))].mean(axis=1)
        low, high = np.quantile(estimates, [.025, .975])
        row = summary[(summary.method == "lightgbm") & (summary.task == task) & (summary.metric == metric)].iloc[0]
        ci_differences.extend([abs(row.ci_low - low), abs(row.ci_high - high)])
    result["independent_bootstrap_max_bound_difference"] = float(max(ci_differences))
    assert max(ci_differences) < 1e-12
    model_path = ROOT / "artifacts/runs/identity_lgbm_17/model.joblib"
    input_paths.append(model_path)
    bundle = joblib.load(model_path)
    features = bundle["features"]
    sample = val.sort_values("record_id").iloc[::500].copy()
    original_features = features.transform(sample)
    poisoned = sample.copy()
    for field in ("logx", "mole_fraction", "source_doi", "record_id", "solute_id", "solvent_id"):
        poisoned[field] = "injected_nonfeature_value"
    assert (original_features != features.transform(poisoned)).nnz == 0
    stored = pd.read_parquet(ROOT / "artifacts/runs/identity_lgbm_17/predictions_val.parquet").set_index("record_id")
    actual = bundle["model"].predict(original_features, num_threads=2)
    expected = stored.loc[sample.record_id, "pred_logx"].to_numpy()
    prediction_error = float(np.max(np.abs(actual - expected)))
    assert prediction_error < 1e-12
    medians, missing_columns = [], 0
    for role in ("solute_smiles", "solvent_smiles"):
        counts = train[role].value_counts()
        matrix = np.asarray([features._descriptor_cache[s] for s in counts.index])
        weights = counts.to_numpy()
        for column in range(len(features.descriptor_names)):
            valid = np.isfinite(matrix[:, column])
            values = np.repeat(matrix[valid, column], weights[valid])
            medians.append(float(np.median(values)) if len(values) else 0.)
            missing_columns += int((~valid).any())
    medians.append(float(np.median(train.temperature_k.to_numpy(np.float32))))
    median_error = float(np.max(np.abs(np.asarray(medians, dtype=np.float32) - features.medians)))
    assert median_error == 0
    result["feature_and_model_audit"] = {
        "target_id_or_doi_injection_changed_feature_nnz": 0,
        "sample_reprediction_records": len(sample), "sample_reprediction_max_error": prediction_error,
        "train_only_imputation_columns_checked": len(medians), "median_max_error": median_error,
        "descriptor_columns_with_any_missing_in_train": missing_columns,
        "validation_unseen_solvent_count": len(set(val.solvent_smiles) - set(train.solvent_smiles)),
    }
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fingerprints = {s: generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in train.solute_smiles.unique()}
    nearest = pd.read_parquet(ROOT / "artifacts/runs/identity_nearest/predictions_val.parquet").set_index("record_id")
    neighbor_checks = []
    for row in sample.iloc[[0, len(sample) // 2, -1]].itertuples():
        candidates = train[train.solvent_smiles == row.solvent_smiles].sort_values("record_id")
        if candidates.empty:
            continue
        query_fp = generator.GetFingerprint(Chem.MolFromSmiles(row.solute_smiles))
        similarities = {s: DataStructs.TanimotoSimilarity(query_fp, fp) for s, fp in fingerprints.items()}
        scores = candidates.solute_smiles.map(similarities)
        candidates = candidates[np.abs(scores - scores.max()) <= 1e-12]
        selected = candidates.iloc[np.argmin(np.abs(candidates.temperature_k.to_numpy() - row.temperature_k))]
        assert selected.record_id == nearest.loc[row.record_id, "nearest_record_id"]
        neighbor_checks.append(row.record_id)
    result["independently_reselected_neighbor_queries"] = neighbor_checks
    group_counts = metadata.groupby(["solute_id", "source_doi", "temperature_k"]).size()
    task_counts, query_counts = Counter(), defaultdict(Counter)
    query_ids_by_solute = defaultdict(set)
    for path in (ROOT / "data/processed/identity").glob("*_queries.jsonl"):
        input_paths.append(path)
        with path.open() as stream:
            for line in stream:
                query = json.loads(line)
                task_counts[query["task"]] += 1
                query_counts[query["solute_id"]][query["task"]] += 1
                query_ids_by_solute[query["solute_id"]].add(query["query_id"])
    full_pairs = int((group_counts * (group_counts - 1) // 2).sum())
    result["task_scope_metadata_only"] = {
        "full_condition_pair_count": full_pairs, "generated_pair_count": task_counts["pair"],
        "omitted_cross_block_pairs": full_pairs - task_counts["pair"],
        "conditions_over_eight_solvents": int((group_counts > 8).sum()),
        "largest_source_condition_candidates": int(group_counts.max()),
        "interpretation": "Best-solvent utility is conditional on the provided candidate block, not the full original solvent collection.",
    }
    shared = identity.groupby("leakage_group").filter(lambda frame: len(frame) > 1)
    result["identity_audit"] = {
        "canonical_identity_count": len(identity),
        "distinct_identities_with_a_shared_parent": len(shared), "shared_parent_groups": int(shared.leakage_group.nunique()),
        "multifragment_identities_preserved": int(identity.is_multifragment.sum()),
        "parent_group_max_partitions": int(identity.groupby("leakage_group").split.nunique().max()),
        "parent_group_max_scaffolds": int(identity.groupby("leakage_group").scaffold_group.nunique().max()),
    }
    if strict:
        strict_manifest_path = ROOT / "data/splits/scaffold.csv"
        strict_data_path = ROOT / "data/processed/measurements_scaffold.parquet"
        input_paths.extend([strict_manifest_path, strict_data_path])
        sm = pd.read_csv(strict_manifest_path)
        sd = pd.read_parquet(strict_data_path, columns=list(metadata.columns))
        assert not sm.solute_id.duplicated().any()
        original_test = set(manifest.loc[manifest.split == "test", "solute_id"])
        assert set(sm.loc[sm.split == "test", "solute_id"]) == original_test
        assert not original_test.intersection(sm.loc[sm.split != "test", "solute_id"])
        assert set(sd.solute_id) == set(sm.solute_id)
        joined = sd.merge(sm[["solute_id", "split"]], on="solute_id", validate="many_to_one")
        assert joined.groupby("scaffold_group").split.nunique().max() == 1
        assert joined.groupby("leakage_group").split.nunique().max() == 1
        assert set(sd.record_id) == set(metadata.loc[metadata.solute_id.isin(sm.solute_id), "record_id"])
        test_scaffolds = set(identity.loc[identity.solute_id.isin(original_test), "scaffold_group"])
        expected_excluded = set(identity.loc[
            ~identity.solute_id.isin(original_test) & identity.scaffold_group.isin(test_scaffolds), "solute_id"])
        assert set(identity.solute_id) - set(sm.solute_id) == expected_excluded
        excluded_path = ROOT / "data/splits/scaffold_excluded.csv"
        input_paths.append(excluded_path)
        assert set(pd.read_csv(excluded_path).solute_id) == expected_excluded
        strict_counts = {}
        for partition, frame in sm.groupby("split"):
            solutes = set(frame.solute_id)
            query_path = ROOT / "data/processed/scaffold" / f"{partition}_queries.jsonl"
            input_paths.append(query_path)
            emitted_queries = read_jsonl(query_path)
            assert all(q["split"] == partition and q["solute_id"] in solutes for q in emitted_queries)
            assert len({q["query_id"] for q in emitted_queries}) == len(emitted_queries)
            assert {q["query_id"] for q in emitted_queries} == set().union(*(query_ids_by_solute[s] for s in solutes))
            strict_counts[partition] = {"solutes": len(solutes), "records": int(sd.solute_id.isin(solutes).sum()),
                                       "queries": {task: sum(query_counts[s][task] for s in solutes) for task in ("pair", "rank")},
                                       "eligible_solutes": {task: sum(query_counts[s][task] > 0 for s in solutes) for task in ("pair", "rank")}}
        result["strict_global_holdout_audit"] = {"status": "PASSED", "same_test_solutes": len(original_test),
                                                  "counts": strict_counts,
                                                  "quarantined_solutes": len(identity) - len(sm),
                                                  "quarantine_matches_test_scaffold_rule": True,
                                                  "emitted_query_ids_match_original_retained_tasks": True,
                                                  "inspection_scope": "Identity/condition metadata and label-free query files only; no test target values or model scores."}
        amendment_path = ROOT / "artifacts/audit/scaffold_amendment.json"
        amendment = json.loads(amendment_path.read_text())
        input_paths.append(amendment_path)
        for partition, values in strict_counts.items():
            for key in ("solutes", "records", "queries"):
                assert amendment["counts"][partition][key] == values[key]
    else:
        result["strict_global_holdout_audit"] = {"status": "NOT_YET_CHECKED", "resume": "Rerun this script with --check-strict after the replacement strict files are complete."}
    # These hand-built examples are tests, not synthetic scientific results.
    from solvent_select.evaluate import evaluate_queries
    q, t, p = [], [], []
    for sid, cid, count, correct in (("s1", "c1", 9, True), ("s1", "c2", 1, False), ("s2", "c3", 1, False)):
        for number in range(count):
            qid = f"{sid}-{cid}-{number}"
            q.append({"query_id": qid, "solute_id": sid, "condition_id": cid, "task": "pair",
                      "candidates": [{"id": "A", "smiles": "CCO"}, {"id": "B", "smiles": "O"}]})
            t.append({"query_id": qid, "choice": "A", "values": {"A": -1., "B": -2.}})
            p.append({"query_id": qid, "choice": "A" if correct else "B"})
    scores, _, _ = evaluate_queries(q, t, p)
    assert scores["by_task"]["pair"]["pair_accuracy"] == .25
    malformed, _, _ = evaluate_queries(q[:1], t[:1], [{"query_id": q[0]["query_id"], "choice": ["A"]}])
    assert malformed["by_task"]["pair"]["pair_accuracy"] == 0
    assert malformed["by_task"]["pair"]["covered"] == 0
    assert malformed["by_task"]["pair"]["regret"] == 1
    result["synthetic_scoring_counterexamples_not_results"] = {
        "hierarchical_macro_expected_and_observed": .25,
        "naive_question_accuracy_would_be": 9 / 11,
        "malformed_choice_accuracy": malformed["by_task"]["pair"]["pair_accuracy"],
        "malformed_choice_coverage": malformed["by_task"]["pair"]["covered"],
        "malformed_choice_regret": malformed["by_task"]["pair"]["regret"],
    }
    input_paths.extend(ROOT / "src/solvent_select" / name for name in ("data.py", "evaluate.py", "baselines.py", "report.py"))
    result["inputs"] = {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(input_paths))}
    result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/audit/independent_m1.json")
    parser.add_argument("--check-strict", action="store_true")
    args = parser.parse_args()
    try:
        result = audit(strict=args.check_strict)
    except Exception as error:
        result = {"status": "FAILED", "scope": "M1 identity validation audit", "error": repr(error),
                  "traceback": traceback.format_exc(), "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "strict": result.get("strict_global_holdout_audit", {}).get("status")}, indent=2))
    return int(result["status"] == "FAILED")


if __name__ == "__main__":
    raise SystemExit(main())

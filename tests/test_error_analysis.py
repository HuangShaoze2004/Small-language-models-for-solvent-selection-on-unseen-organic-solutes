"""Synthetic scientific-logic tests; fixtures are never research results."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solvent_select.error_analysis import (
    FUNCTIONAL_GROUPS, METRICS, SOLVENT_GROUPS, annotate_queries, compile_patterns, enrich_cases,
    interval_label, load_config, matched_groups, molecule, run_analysis,
    select_cases, sha256, solvent_groups, summarize_slices, validate_inputs,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/error_analysis.yaml"


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def fixture_data():
    structures = {"train": "CN", "s1": "CCO", "s2": "CC#N"}
    rows = []
    for sid, candidates, temp in (("train", ["O"], 298.15),
                                   ("s1", ["O", "CCO"], 283.15),
                                   ("s2", ["CCOC(C)=O", "CC#N"], 323.15)):
        for index, solvent in enumerate(candidates):
            rows.append({"record_id": f"{sid}-{index}", "solute_id": sid,
                         "solute_smiles": structures[sid], "solvent_smiles": solvent,
                         "temperature_k": temp, "source_doi": f"10.fixture/{sid}",
                         "leakage_group": f"parent-{sid}", "logx": "PROHIBITED_LABEL_COLUMN"})
    metadata = pd.DataFrame(rows)
    manifest = pd.DataFrame({"solute_id": ["train", "s1", "s2"],
                             "split": ["train", "val", "val"],
                             "leakage_group": ["parent-train", "parent-s1", "parent-s2"]})
    queries, scores = [], []
    for sid, task in (("s1", "pair"), ("s2", "rank")):
        records = metadata[metadata.solute_id == sid]
        queries.append({"query_id": f"q-{sid}", "solute_id": sid, "solute_smiles": structures[sid],
                        "temperature_k": float(records.temperature_k.iloc[0]), "condition_id": f"c-{sid}",
                        "task": task, "split": "val", "candidates": [
                            {"id": chr(65 + index), "record_id": r.record_id, "smiles": r.solvent_smiles}
                            for index, r in enumerate(records.itertuples())]})
        scores.append({"query_id": f"q-{sid}", "solute_id": sid, "condition_id": f"c-{sid}", "task": task,
                       "candidate_count": 2, "pair_accuracy": 0. if task == "pair" else np.nan,
                       "nontie_pair_accuracy": 0. if task == "pair" else np.nan,
                       "top1": 0. if task == "rank" else np.nan,
                       "rank_tau": -1. if task == "rank" else np.nan, "regret": 1., "covered": 1.})
    return queries, pd.DataFrame(scores), metadata, manifest


def test_interval_boundaries_are_fixed(config):
    temps = [283.149, 283.15, 303.15, 323.15]
    assert [interval_label(t, config["temperature_boundaries_kelvin"], config["temperature_labels"]) for t in temps] == config["temperature_labels"]
    for index, value in enumerate((0., .2, .4, .6, .8)):
        assert interval_label(value, config["similarity_edges"], config["similarity_labels"], True) == config["similarity_labels"][index]
    assert interval_label(1., config["similarity_edges"], config["similarity_labels"], True) == "[0.8,1]"
    with pytest.raises(ValueError):
        interval_label(1.01, config["similarity_edges"], config["similarity_labels"], True)


def test_chemical_counterexamples_and_overlapping_classes(config):
    groups = compile_patterns(config["solute_functional_groups"])
    assert matched_groups(molecule("CN(C)C=O"), groups) == ["amide"]
    assert "amine" not in matched_groups(molecule("C[N+](=O)[O-]"), groups)
    assert set(matched_groups(molecule("Nc1ccccc1"), groups)) == {"amine", "aromatic"}
    assert set(matched_groups(molecule("Oc1ccccc1"), groups)) == {"phenol", "aromatic"}
    assert matched_groups(molecule("CC(=O)O"), groups) == ["carboxylic_acid"]
    assert "amine" not in matched_groups(molecule("C[NH3+]"), groups)
    assert "sulfonyl" not in matched_groups(molecule("CS(C)=O"), groups)
    assert "sulfonyl" in matched_groups(molecule("CS(C)(=O)=O"), groups)
    solvents = compile_patterns(config["solvent_smarts"])
    assert solvent_groups(molecule("CCOC(C)=O"), solvents) == ["ester"]
    assert solvent_groups(molecule("C1CCOC1"), solvents) == ["ether"]
    assert solvent_groups(molecule("O"), solvents) == ["water"]
    assert solvent_groups(molecule("CS(C)=O"), solvents) == ["other"]
    assert solvent_groups(molecule("Cc1ccccc1"), solvents) == ["aromatic", "hydrocarbon"]


def test_similarity_cannot_find_held_out_self(config, fixture_data):
    queries, scores, metadata, manifest = fixture_data
    annotations, audit = annotate_queries(queries, metadata, manifest, config)
    assert audit["train_solute_count"] == 1
    assert set(annotations.nearest_train_solute_id) == {"train"}
    assert (annotations.train_similarity < 1).all()
    poisoned = metadata.copy()
    poisoned["logx"] = "ALTERED_LABEL_CANNOT_AFFECT_ANNOTATIONS"
    other, _ = annotate_queries(queries, poisoned, manifest, config)
    pd.testing.assert_frame_equal(annotations, other)


def test_hierarchical_macro_is_recomputed_within_each_slice(config):
    rows = []
    # Nine successes and one failure for s1 occur in different physical
    # conditions. s2 has one failure. The solute macro is .25, not 9/11.
    for sid, cid, count, correct in (("s1", "c1", 9, 1.), ("s1", "c2", 1, 0.), ("s2", "c3", 1, 0.)):
        for number in range(count):
            row = {"query_id": f"{cid}-{number}", "solute_id": sid, "condition_id": cid, "task": "pair",
                   "temperature_bin": config["temperature_labels"][1], "similarity_bin": config["similarity_labels"][0],
                   "functional_groups": ["alcohol"], "solvent_classes": ["water", "alcohol"],
                   **{metric: np.nan for metric in METRICS}}
            row.update(pair_accuracy=correct, nontie_pair_accuracy=correct, regret=1-correct, covered=1.)
            rows.append(row)
    summary, per_solute = summarize_slices(pd.DataFrame(rows), config)
    water = summary[(summary.slice_family == "candidate_solvent_class") & (summary.slice == "water") & (summary.task == "pair")].iloc[0]
    assert water.pair_accuracy == .25
    assert water.regret == .75
    assert water.solutes == 2 and water.physical_conditions == 3
    assert water.status == "INSUFFICIENT_SOLUTES" and water.overlapping_multilabel_slice
    empty = summary[(summary.slice_family == "solute_functional_group") & (summary.slice == "nitrile") & (summary.task == "pair")].iloc[0]
    assert empty.solutes == 0 and pd.isna(empty.pair_accuracy)
    assert empty.pair_accuracy_support_solutes == 0
    assert len(summary) == 2 * (1 + 4 + 5 + len(FUNCTIONAL_GROUPS) + len(SOLVENT_GROUPS))
    assert len(per_solute) > 2  # Memberships overlap; rows are not independent replicates.


def test_provenance_and_split_inconsistencies_fail(fixture_data):
    queries, scores, metadata, manifest = fixture_data
    validate_inputs(queries, scores, metadata, manifest)
    leaked = manifest.copy()
    leaked.loc[leaked.solute_id == "s1", "split"] = "train"
    with pytest.raises(ValueError, match="not validation"):
        validate_inputs(queries, scores, metadata, leaked)
    bad = metadata.copy()
    bad.loc[bad.record_id == "s1-1", "source_doi"] = "other-source"
    with pytest.raises(ValueError, match="source DOI"):
        validate_inputs(queries, scores, bad, manifest)
    with pytest.raises(ValueError, match="exactly match"):
        validate_inputs(queries, scores.iloc[:1], metadata, manifest)


def test_case_selection_is_order_independent_and_keeps_provenance(config, fixture_data):
    queries, scores, metadata, manifest = fixture_data
    annotations, _ = annotate_queries(queries, metadata, manifest, config)
    details = scores.merge(annotations, on="query_id")
    records = metadata.set_index("record_id")
    first = select_cases(details, queries, records, config)
    second = select_cases(details.iloc[::-1], queries[::-1], records, config)
    # NaNs are not equal; compare selection identity and source data explicitly.
    columns = ("selection", "selection_rank", "query_id", "candidate_record_ids", "source_doi")
    assert [[case[c] for c in columns] for case in first] == [[case[c] for c in columns] for case in second]
    assert len(first) == 4 and all(case["pH"] is None for case in first)
    assert all(case["candidate_record_ids"] for case in first)


def test_end_to_end_projects_away_labels_and_blocks_test_before_reads(tmp_path, fixture_data):
    queries, scores, metadata, manifest = fixture_data
    qp, sp, mp, fp = (tmp_path / name for name in ("queries.jsonl", "scores.parquet", "measurements.parquet", "manifest.csv"))
    qp.write_text("".join(json.dumps(q) + "\n" for q in queries))
    scores.to_parquet(sp)
    metadata.to_parquet(mp)
    manifest.to_csv(fp, index=False)
    output = tmp_path / "report"
    report = run_analysis(qp, sp, mp, fp, CONFIG_PATH, output)
    assert report["status"] == "VALIDATION_DIAGNOSTICS_ONLY"
    assert "logx" not in report["measurement_columns_read"]
    assert len(report["slices"]) == 62
    assert json.loads((output / "analysis.json").read_text())["test_labels_read"] is False
    with pytest.raises(FileExistsError):
        run_analysis(qp, sp, mp, fp, CONFIG_PATH, output)
    queries[0]["split"] = "test"
    qp.write_text("".join(json.dumps(q) + "\n" for q in queries))
    with pytest.raises(ValueError, match="validation or frozen-test"):
        run_analysis(qp, tmp_path / "does_not_exist", mp, fp, CONFIG_PATH, tmp_path / "never_written")


def test_enriched_cases_match_measured_and_predicted_order(tmp_path, fixture_data):
    queries, scores, metadata, manifest = fixture_data
    qp, sp, mp, fp, tp, pp = (tmp_path / name for name in (
        "queries.jsonl", "scores.parquet", "measurements.parquet", "manifest.csv", "targets.jsonl", "predictions.parquet"))
    qp.write_text("".join(json.dumps(q) + "\n" for q in queries))
    scores.to_parquet(sp)
    metadata.to_parquet(mp)
    manifest.to_csv(fp, index=False)
    targets = [{"query_id": q["query_id"], "choice": "A", "values": {"A": -1., "B": -2.}} for q in queries]
    tp.write_text("".join(json.dumps(row) + "\n" for row in targets))
    predictions = pd.DataFrame([{"record_id": candidate["record_id"], "pred_logx": -2. if candidate["id"] == "A" else -1., "split": "val"}
                                for query in queries for candidate in query["candidates"]])
    predictions.to_parquet(pp)
    output = tmp_path / "enriched"
    run_analysis(qp, sp, mp, fp, CONFIG_PATH, output, tp, pp)
    cases = [json.loads(line) for line in (output / "failure_cases.jsonl").read_text().splitlines()]
    assert all(case["actual_best_candidate_ids"] == ["A"] and case["model_selected_candidate_id"] == "B" for case in cases)
    assert cases[0]["candidates"][0]["actual_logx"] == -1.
    assert cases[0]["candidates"][0]["predicted_logx"] == -2.
    # LLM serialization must retain the raw text, without guessing absent values.
    llm = tmp_path / "llm.jsonl"
    llm.write_text("".join(json.dumps({"query_id": q["query_id"], "raw_output": "recorded model response",
                                      "prediction": {"choice": "B"} if q["task"] == "pair" else {"ranking": ["B", "A"]}}) + "\n"
                           for q in queries))
    other = tmp_path / "enriched_llm"
    run_analysis(qp, sp, mp, fp, CONFIG_PATH, other, tp, llm)
    first = json.loads((other / "failure_cases.jsonl").open().readline())
    assert first["raw_prediction"]["raw_output"] == "recorded model response"
    assert first["candidates"][0]["predicted_logx"] is None
    predictions["pred_logx"] *= -1
    predictions.to_parquet(pp)
    with pytest.raises(ValueError, match="does not reproduce"):
        run_analysis(qp, sp, mp, fp, CONFIG_PATH, tmp_path / "wrong_model", tp, pp)


def test_test_gate_requires_status_and_exact_hash_before_score_read(tmp_path, fixture_data):
    queries, scores, metadata, manifest = fixture_data
    for query in queries:
        query["split"] = "test"
    manifest.loc[manifest.split == "val", "split"] = "test"
    qp, sp, mp, fp = (tmp_path / name for name in ("queries.jsonl", "scores.parquet", "measurements.parquet", "manifest.csv"))
    qp.write_text("".join(json.dumps(q) + "\n" for q in queries))
    freeze = tmp_path / "freeze.json"
    with pytest.raises(ValueError, match="FROZEN"):
        run_analysis(qp, sp, mp, fp, CONFIG_PATH, tmp_path / "no_freeze")
    freeze.write_text(json.dumps({"status": "DRAFT", "query_sha256": sha256(qp)}))
    with pytest.raises(ValueError, match="not authorized"):
        run_analysis(qp, sp, mp, fp, CONFIG_PATH, tmp_path / "draft", freeze_manifest=freeze)
    freeze.write_text(json.dumps({"status": "FROZEN", "query_sha256": "wrong"}))
    with pytest.raises(ValueError, match="not authorized"):
        run_analysis(qp, sp, mp, fp, CONFIG_PATH, tmp_path / "wrong_hash", freeze_manifest=freeze)
    # This positive-path test uses synthetic fixtures only, never actual test data.
    freeze.write_text(json.dumps({"status": "FROZEN", "query_sha256": sha256(qp), "error_analysis_config_sha256": sha256(CONFIG_PATH)}))
    scores.to_parquet(sp)
    metadata.to_parquet(mp)
    manifest.to_csv(fp, index=False)
    result = run_analysis(qp, sp, mp, fp, CONFIG_PATH, tmp_path / "synthetic_frozen", freeze_manifest=freeze)
    assert result["partition"] == "test" and result["status"] == "FROZEN_TEST_DIAGNOSTICS"
    assert result["test_analysis_performed"]
    freeze.write_text(json.dumps({"status": "FROZEN", "query_sha256": sha256(qp), "error_analysis_config_sha256": "changed"}))
    with pytest.raises(ValueError, match="implementation changed"):
        run_analysis(qp, sp, mp, fp, CONFIG_PATH, tmp_path / "changed_config", freeze_manifest=freeze)


def test_mislabelled_test_queries_are_rejected_before_scores(tmp_path, fixture_data):
    queries, _, _, manifest = fixture_data
    manifest.loc[manifest.split == "val", "split"] = "test"
    qp, fp = tmp_path / "queries.jsonl", tmp_path / "manifest.csv"
    qp.write_text("".join(json.dumps(q) + "\n" for q in queries))
    manifest.to_csv(fp, index=False)
    with pytest.raises(ValueError, match="scores were not read"):
        run_analysis(qp, tmp_path / "absent_scores", tmp_path / "absent_measurements", fp,
                     CONFIG_PATH, tmp_path / "not_created")


def test_case_rejects_changed_lower_ranking_with_same_top_one_and_regret(tmp_path):
    # Keeping the first-ranked candidate unchanged preserves top1 and regret,
    # but swapping lower ranks changes tau. All stored metrics must agree.
    query = {"query_id": "q", "task": "rank", "candidates": [
        {"id": cid, "smiles": smiles, "record_id": cid} for cid, smiles in zip("ABC", ("O", "CCO", "CC#N"))]}
    target = tmp_path / "targets.jsonl"
    target.write_text(json.dumps({"query_id": "q", "values": {"A": -1., "B": -2., "C": -3.}}) + "\n")
    prediction = tmp_path / "prediction.jsonl"
    prediction.write_text(json.dumps({"query_id": "q", "prediction": {"ranking": ["B", "C", "A"]}}) + "\n")
    case = {"query_id": "q", "pair_accuracy": np.nan, "nontie_pair_accuracy": np.nan,
            "top1": 0., "regret": 1., "covered": 1., "rank_tau": 1/3,
            "candidates": query["candidates"]}
    with pytest.raises(ValueError, match="rank_tau"):
        enrich_cases([case], [query], pd.DataFrame(), pd.DataFrame(), target, prediction, .1, "val")


def test_wrong_score_file_is_rejected_before_outcome_columns(tmp_path, fixture_data, monkeypatch):
    queries, scores, _, manifest = fixture_data
    qp, sp, fp = tmp_path / "queries.jsonl", tmp_path / "scores.parquet", tmp_path / "manifest.csv"
    qp.write_text("".join(json.dumps(q) + "\n" for q in queries))
    manifest.to_csv(fp, index=False)
    scores["query_id"] = ["other-q1", "other-q2"]
    scores.to_parquet(sp)
    original, columns_read = pd.read_parquet, []
    def observed_read(path, **kwargs):
        columns_read.append(kwargs.get("columns"))
        return original(path, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", observed_read)
    with pytest.raises(ValueError, match="outcome columns were not read"):
        run_analysis(qp, sp, tmp_path / "absent_measurements", fp, CONFIG_PATH, tmp_path / "never_written")
    assert len(columns_read) == 1
    assert not set(METRICS).intersection(columns_read[0])

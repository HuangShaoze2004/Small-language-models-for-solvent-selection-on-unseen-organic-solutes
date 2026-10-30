"""Numerical fixtures are unit tests only, never scientific result artifacts."""
import json
from pathlib import Path

import pandas as pd
import pytest

from solvent_select.report import audit_tables, build_report, collect_runs, sha256_file, summarize_runs


def test_scaffold_amendment_overrides_superseded_counts(tmp_path):
    audit = tmp_path / "data_audit.json"
    audit.write_text(json.dumps({"splits": {
        "identity": {"train": {"solutes": 8, "records": 70}},
        "scaffold": {"train": {"solutes": 8, "records": 70}},
    }}))
    amendment = tmp_path / "scaffold_amendment.json"
    amendment.write_text(json.dumps({"counts": {
        "train": {"solutes": 5, "records": 40},
        "val": {"solutes": 1, "records": 10},
        "test": {"solutes": 1, "records": 10},
        "excluded": {"solutes": 3, "records": 30},
    }}))
    _, counts = audit_tables(audit)
    assert counts[(counts.split_family == "identity") & (counts.partition == "train")].solutes.item() == 8
    strict = counts[counts.split_family == "scaffold"]
    assert strict[strict.partition == "train"].solutes.item() == 5
    assert strict[strict.partition == "excluded"].solutes.item() == 3
    for encoded in strict.source_refs:
        ref = json.loads(encoded)[0]
        assert ref["sha256"] == sha256_file(amendment)
        assert ref["field"].startswith("counts.")


def make_run(root: Path, name="model_17", seed=17, cohort="a" * 64, elapsed=None, values=(1., 0.)):
    run = root / name
    evaluation = run / "evaluation"
    evaluation.mkdir(parents=True)
    frame = pd.DataFrame({
        "solute_id": ["s1", "s2"], "task": ["pair", "pair"],
        "pair_accuracy": values, "nontie_pair_accuracy": values,
        "top1": [None, None], "rank_tau": [None, None],
        "regret": [0., .4], "covered": [1., 1.],
    })
    frame.to_csv(evaluation / "per_solute.csv", index=False)
    task_metrics = {c: (None if frame[c].isna().all() else float(frame[c].mean()))
                    for c in frame.columns if c not in {"solute_id", "task"}}
    metrics = {"queries": 8, "solutes": 2, "by_task": {"pair": {"solutes": 2, "queries": 8, **task_metrics}},
               "inputs": {"data/processed/identity/val_queries.jsonl": cohort}}
    (evaluation / "metrics.json").write_text(json.dumps(metrics))
    metadata = {"status": "completed", "method": "model", "seed": seed}
    if elapsed is not None:
        metadata["elapsed_seconds"] = elapsed
    (run / "metadata.json").write_text(json.dumps(metadata))
    return run


def test_missing_cost_and_ci_are_not_filled(tmp_path):
    runs = tmp_path / "runs"
    make_run(runs)
    output = tmp_path / "report"
    manifest = build_report(runs, output, repetitions=0, plots=False)
    summary = pd.read_csv(output / "metrics_by_method.csv")
    assert summary.ci_low.isna().all()
    assert summary.ci_high.isna().all()
    assert pd.read_csv(output / "costs_by_run.csv").empty
    assert manifest["has_model_results"]
    assert all(row["sources"] for row in manifest["table_row_provenance"]["metrics_by_run"])


def test_source_hashes_and_seed_average_are_correct(tmp_path):
    runs = tmp_path / "runs"
    first = make_run(runs, "model_17", 17, elapsed=3., values=(1., 0.))
    make_run(runs, "model_42", 42, elapsed=4., values=(0., 1.))
    original_hash = sha256_file(first / "evaluation" / "per_solute.csv")
    run_table, costs, per_run, _ = collect_runs(runs)
    summary = summarize_runs(run_table, per_run, repetitions=1000)
    row = summary[summary.metric == "pair_accuracy"].iloc[0]
    assert row.value == .5
    # Averaging the two seeds gives .5 for every solute, so a solute bootstrap
    # must have a zero-width CI. Bootstrapping the four rows would be wrong.
    assert row.ci_low == row.ci_high == .5
    assert row.n_runs == row.n_seeds == 2
    assert set(costs.cost_scope) == {"training_run_wall"}
    assert sha256_file(first / "evaluation" / "per_solute.csv") == original_hash


def test_different_query_cohorts_are_never_pooled(tmp_path):
    runs = tmp_path / "runs"
    make_run(runs, "model_17", 17, cohort="a" * 64)
    make_run(runs, "model_42", 42, cohort="b" * 64)
    table, _, per_run, _ = collect_runs(runs)
    summary = summarize_runs(table, per_run, repetitions=0)
    accuracy = summary[summary.metric == "pair_accuracy"]
    assert len(accuracy) == 2
    assert set(accuracy.n_runs) == {1}


def test_same_cohort_and_hyperparameters_cannot_pool_different_training_banks(tmp_path):
    runs=tmp_path/'runs'
    for seed,bank in [(17,'a'*64),(42,'b'*64)]:
        run=make_run(runs,f'model_{seed}',seed)
        path=run/'metadata.json';metadata=json.loads(path.read_text())
        metadata['measurement_sha256']=bank;metadata['split_sha256']='c'*64
        path.write_text(json.dumps(metadata))
    table,_,per_run,_=collect_runs(runs)
    with pytest.raises(ValueError,match='cannot be pooled'):
        summarize_runs(table,per_run,repetitions=100)


def test_disagreeing_aggregate_is_rejected(tmp_path):
    run = make_run(tmp_path / "runs")
    path = run / "evaluation" / "metrics.json"
    data = json.loads(path.read_text())
    data["by_task"]["pair"]["pair_accuracy"] = .9
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Metric/CSV mismatch"):
        collect_runs(tmp_path / "runs")


def test_audit_only_report_and_figure_creation(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "raw_rows": 100, "pre_duplicate_rows": 90, "final_measurements": 89,
        "splits": {"identity": {
            "train": {"solutes": 8, "records": 70, "queries": {"pair": 10, "rank": 2}},
            "val": {"solutes": 1, "records": 9, "queries": {"pair": 2, "rank": 1}},
            "test": {"solutes": 1, "records": 10, "queries": {"pair": 3, "rank": 1}},
        }}}))
    output = tmp_path / "report"
    manifest = build_report(tmp_path / "missing_runs", output, audit, plots=True)
    assert not manifest["has_model_results"]
    assert pd.read_csv(output / "metrics_by_run.csv").empty
    assert list(pd.read_csv(output / "data_flow.csv").measurements) == [100, 90, 89]
    assert (output / "figures" / "data_curation_flow.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "figures" / "partition_sizes.png").read_bytes().startswith(b"\x89PNG")
    for item in manifest["outputs"]:
        assert sha256_file(item["path"]) == item["sha256"]


def test_modified_generated_file_is_protected(tmp_path):
    output = tmp_path / "report"
    build_report(tmp_path / "missing_runs", output, plots=False)
    path = output / "metrics_by_run.csv"
    path.write_text("user modification\n")
    with pytest.raises(FileExistsError, match="was edited"):
        build_report(tmp_path / "missing_runs", output, plots=False)


def test_default_validation_report_skips_test_cohort(tmp_path):
    run = make_run(tmp_path / "runs")
    path = run / "evaluation" / "metrics.json"
    data = json.loads(path.read_text())
    data["inputs"] = {"data/processed/identity/test_queries.jsonl": "a" * 64}
    path.write_text(json.dumps(data))
    table, _, _, warnings = collect_runs(tmp_path / "runs")
    assert table.empty
    assert any("partition 'test'" in message for message in warnings)


def test_configurations_cannot_be_disguised_as_seeds(tmp_path):
    runs = tmp_path / "runs"
    first = make_run(runs, "model_17", 17)
    second = make_run(runs, "model_42", 42)
    for run, leaves in ((first, 31), (second, 63)):
        path = run / "metadata.json"
        data = json.loads(path.read_text())
        data["config"] = {"num_leaves": leaves}
        path.write_text(json.dumps(data))
    table, _, per_run, _ = collect_runs(runs)
    with pytest.raises(ValueError, match="Different configurations"):
        summarize_runs(table, per_run, repetitions=0)


def test_rebuild_removes_only_unchanged_generated_outputs(tmp_path):
    runs = tmp_path / "runs"
    run = make_run(runs)
    output = tmp_path / "report"
    build_report(runs, output, repetitions=0, plots=False)
    assert list(output.glob("metrics_*.tex"))
    user_note = output / "human_note.txt"
    user_note.write_text("Keep this note")
    path = run / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["status"] = "failed"
    path.write_text(json.dumps(metadata))
    manifest = build_report(runs, output, repetitions=0, plots=False)
    assert not list(output.glob("metrics_*.tex"))
    assert user_note.read_text() == "Keep this note"
    assert not manifest["has_model_results"]


def test_inference_metadata_is_identified_and_not_mislabelled_training_cost(tmp_path):
    runs = tmp_path / 'runs'
    run = make_run(runs, 'evidence')
    (run / 'metadata.json').unlink()
    predictions = run / 'predictions.jsonl'
    predictions.write_text('{}\n')
    metadata = {'status': 'completed', 'elapsed_seconds': 5., 'fingerprint': {
        'prediction_kind': 'evidence', 'config': {'retrieval_mode': 'condition', 'seed': 17},
        'adapter_sha256': None}}
    predictions.with_suffix('.jsonl.run.json').write_text(json.dumps(metadata))
    metrics_path = run / 'evaluation/metrics.json'
    metrics = json.loads(metrics_path.read_text())
    metrics['inputs'][str(predictions)] = sha256_file(predictions)
    metrics_path.write_text(json.dumps(metrics))
    rows, costs, _, _ = collect_runs(runs)
    assert set(rows.method) == {'direct_evidence:condition'}
    assert set(costs.cost_scope) == {'inference_run_wall'}
    assert rows.seed.isna().all()


def test_consistency_scores_are_not_pooled_with_task_accuracy(tmp_path):
    run = tmp_path / 'runs/check/evaluation'
    run.mkdir(parents=True)
    (run / 'metrics.json').write_text(json.dumps({'by_variant_task': {}, 'partition': 'val'}))
    rows, costs, _, warnings = collect_runs(tmp_path / 'runs')
    assert rows.empty
    assert any('consistency' in item for item in warnings)


def test_training_cost_includes_archived_interrupted_segments(tmp_path):
    runs=tmp_path/'runs'
    run=make_run(runs,'model_17',elapsed=4.)
    archive=run/'recovery_segments/segment1';archive.mkdir(parents=True)
    (archive/'metadata.json').write_text(json.dumps({'status':'interrupted','elapsed_seconds':3.}))
    _,costs,_,_=collect_runs(runs)
    row=costs[costs.cost_scope=='training_run_wall'].iloc[0]
    assert row.seconds==7.
    assert len(json.loads(row.source_refs))==2


def test_resumed_inference_cost_does_not_assign_last_segment_to_full_cohort(tmp_path):
    runs = tmp_path / 'runs'
    run = make_run(runs, 'resumed_inference')
    (run / 'metadata.json').unlink()
    predictions = run / 'predictions.jsonl'
    predictions.write_text(''.join(json.dumps(row) + '\n' for row in [
        {'query_id': 'old1', 'batch_id': 'before_pause', 'batch_wall_seconds': 3.},
        {'query_id': 'old2', 'batch_id': 'before_pause', 'batch_wall_seconds': 3.},
        {'query_id': 'new1', 'batch_id': 'after_resume', 'batch_wall_seconds': 5.},
    ]))
    predictions.with_suffix('.jsonl.run.json').write_text(json.dumps({
        'status': 'completed', 'elapsed_seconds': 6., 'already_complete': 2, 'completed': 3,
        'fingerprint': {'prediction_kind': 'llm', 'config': {'retrieval_mode': 'none', 'seed': 17}}}))
    metrics_path = run / 'evaluation/metrics.json'
    metrics = json.loads(metrics_path.read_text())
    metrics['inputs'][str(predictions)] = sha256_file(predictions)
    metrics_path.write_text(json.dumps(metrics))
    _, costs, _, _ = collect_runs(runs)
    assert set(costs.cost_scope) == {'inference_last_segment_wall', 'inference_successful_batches_wall'}
    last = costs[costs.cost_scope == 'inference_last_segment_wall'].iloc[0]
    batches = costs[costs.cost_scope == 'inference_successful_batches_wall'].iloc[0]
    assert last.seconds == 6. and last.records == 1
    assert batches.seconds == 8. and batches.records == 3

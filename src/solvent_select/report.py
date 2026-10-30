"""Rebuild evidence-linked tables and figures from completed evaluations.

The reporter never opens target or prediction files. Evaluation JSON and
per-solute CSV are the numerical sources; adjacent metadata supplies cost.
Missing measurements remain missing, and different query hashes form separate
cohorts. The default reports validation data only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import textwrap
from typing import Any

import numpy as np
import pandas as pd

METRICS = ("pair_accuracy", "nontie_pair_accuracy", "top1", "regret", "covered", "rank_tau")
PRIMARY_METRICS = {"pair": ("pair_accuracy", "regret"), "rank": ("top1", "regret")}
METRIC_LABELS = {
    "pair_accuracy": "Comparison accuracy (%)",
    "nontie_pair_accuracy": "Non-tie comparison accuracy (%)",
    "top1": "Best-solvent hit rate (%)",
    "regret": "Selection regret (log10 mole fraction)",
    "covered": "Valid-output coverage (%)",
    "rank_tau": "Kendall tau",
}
SUMMARY_COLUMNS = [
    "cohort_id", "split_family", "partition", "method", "task", "metric", "value",
    "ci_low", "ci_high", "bootstrap_repetitions", "bootstrap_seed", "solutes",
    "support_solutes", "n_runs", "n_seeds", "seeds", "run_ids", "source_refs",
]
RUN_COLUMNS = [
    "run_id", "cohort_id", "split_family", "partition", "method", "seed", "task",
    "metric", "value", "solutes", "support_solutes", "queries", "source_refs",
]
COST_COLUMNS = [
    "run_id", "cohort_id", "split_family", "partition", "method", "seed", "cost_scope",
    "seconds", "records", "source_refs",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def bootstrap_mean(values, repetitions: int, seed: int) -> tuple[float | None, float | None]:
    """Percentile CI from real per-solute observations; zero means no CI."""
    if repetitions < 0:
        raise ValueError("Bootstrap repetitions cannot be negative")
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if repetitions == 0 or len(array) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for start in range(0, repetitions, 500):
        size = min(500, repetitions - start)
        estimates[start:start + size] = array[rng.integers(0, len(array), (size, len(array)))].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def provenance_ref(path: Path, field: str | None = None) -> dict:
    result = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    if field:
        result["field"] = field
    return result


def _query_context(metrics: dict, run_id: str) -> tuple[str, str, str]:
    """Use existing provenance, not inferred labels from a run's friendly name."""
    query_inputs = [(str(path), digest) for path, digest in metrics.get("inputs", {}).items()
                    if "quer" in Path(path).name.lower()]
    partition = str(metrics.get("evaluation_partition", metrics.get("partition", "unknown")))
    family = str(metrics.get("split_family", "unknown"))
    if len(query_inputs) == 1:
        path, digest = query_inputs[0]
        parts = Path(path).parts
        for candidate in ("identity", "scaffold"):
            if candidate in parts:
                family = candidate
        name = Path(path).name.lower()
        for candidate in ("train", "val", "test"):
            if re.search(rf"(^|[_\-.]){candidate}([_\-.]|$)", name):
                partition = candidate
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise ValueError(f"Invalid query hash in {run_id}")
        return str(digest), family, partition
    # Unidentified populations must never be pooled solely because run names match.
    return f"unverified-{run_id}", family, partition


def _run_identity(run: Path, metrics: dict) -> tuple[dict, Path | None, str, Any]:
    metadata_path = next((run / name for name in ("metadata.json", "run.json") if (run / name).exists()), None)
    metadata = read_json(metadata_path) if metadata_path else {}
    if metadata_path is None:
        for source in metrics.get('inputs', {}):
            prediction = Path(source)
            sidecar = prediction.with_suffix(prediction.suffix + '.run.json')
            if 'predict' not in prediction.name or not sidecar.exists():
                continue
            metadata_path = sidecar
            metadata = read_json(sidecar)
            fingerprint = metadata.get('fingerprint', {})
            config = fingerprint.get('config', {})
            mode = config.get('retrieval_mode', 'none')
            metadata = {**metadata, 'config': config, '_record_kind': 'inference',
                        'implementation_sha256': fingerprint.get('implementation_sha256'),
                        'model_revision': config.get('revision')}
            if fingerprint.get('prediction_kind') == 'evidence':
                metadata.update(method=f'direct_evidence:{mode}', seed=None)
            else:
                adaptation = 'base'
                experiment = read_json(run / 'experiment.json') if (run / 'experiment.json').exists() else {}
                argv = experiment.get('command', [])
                if fingerprint.get('adapter_sha256'):
                    adaptation = 'adapted_unidentified'
                    if '--adapter' in argv:
                        training_path = Path(argv[argv.index('--adapter') + 1]).parent / 'run.json'
                        if training_path.exists():
                            training = read_json(training_path)
                            adaptation = 'sft' if training['config']['retrieval_mode'] == 'none' else 'rag_sft'
                            metadata['_training_path'] = str(training_path)
                            metadata['training_config'] = {k: v for k, v in training['config'].items() if k != 'seed'}
                            metadata['training_inputs'] = training.get('inputs')
                metadata.update(method=f'llm:{adaptation}:{mode}', seed=config.get('seed'))
            break
    method = metadata.get("method", metrics.get("method"))
    config = metadata.get("config", {})
    if not method and config.get("model_id"):
        method = f"sft:{config.get('retrieval_mode', 'none')}"
    method = str(method or run.name)
    seed = metadata.get("seed", config.get("seed", metrics.get("seed")))
    return metadata, metadata_path, method, seed


def _cost_rows(run, context, metadata, metadata_path, metrics):
    rows = []
    field = 'total_elapsed_seconds' if 'total_elapsed_seconds' in metadata else 'elapsed_seconds'
    elapsed = finite_number(metadata.get(field))
    cost_sources = [provenance_ref(metadata_path, field)] if metadata_path else []
    if field == 'elapsed_seconds' and elapsed is not None and metadata.get('status') == 'completed':
        for prior_path in sorted(run.glob('recovery_segments/*/metadata.json')):
            prior = read_json(prior_path)
            seconds = finite_number(prior.get('elapsed_seconds'))
            if prior.get('status') in ('interrupted', 'failed') and seconds is not None:
                elapsed = (elapsed or 0.) + seconds
                cost_sources.append(provenance_ref(prior_path, 'elapsed_seconds'))
    if elapsed is not None and elapsed >= 0 and metadata_path and metadata.get('_record_kind') != 'inference':
        rows.append({**context, "cost_scope": "training_run_wall", "seconds": elapsed,
                     "records": metadata.get("train_records", metadata.get("train_examples")),
                     "source_refs": json.dumps(cost_sources)})
    if metadata.get('_training_path'):
        training_path = Path(metadata['_training_path'])
        training = read_json(training_path)
        field = 'total_elapsed_seconds' if 'total_elapsed_seconds' in training else 'elapsed_seconds'
        seconds = finite_number(training.get(field))
        if seconds is not None:
            rows.append({**context, 'cost_scope': 'shared_adapter_training_wall', 'seconds': seconds,
                         'records': training.get('train_examples'), 'shared_training_run': str(training_path.parent),
                         'source_refs': json.dumps([provenance_ref(training_path, field)])})
    # Timing records are distinct from total training/validation elapsed time.
    for source_path in metrics.get("inputs", {}):
        prediction = Path(source_path)
        if "predict" not in prediction.name.lower():
            continue
        if not prediction.is_absolute():
            prediction = Path.cwd() / prediction
        for timing_path, scope in (
            (prediction.with_suffix(".timing.json"), "inference_batch_wall"),
            (prediction.with_suffix(prediction.suffix + ".run.json"), "inference_run_wall"),
        ):
            if not timing_path.exists():
                continue
            timing = read_json(timing_path)
            if timing.get("status") not in (None, "completed"):
                continue
            seconds = finite_number(timing.get("elapsed_seconds"))
            if seconds is None or seconds < 0:
                continue
            resumed = bool(timing.get('already_complete', 0))
            effective_scope = 'inference_last_segment_wall' if resumed and scope == 'inference_run_wall' else scope
            records = timing.get("records", timing.get("completed"))
            if resumed and effective_scope == 'inference_last_segment_wall' and records is not None:
                records -= timing['already_complete']
            rows.append({**context, "cost_scope": effective_scope, "seconds": seconds,
                         "records": records,
                         "source_refs": json.dumps([provenance_ref(timing_path, "elapsed_seconds")])})
        # Each batch latency is repeated on all of its question rows. Count
        # batches once, including the preserved pre-resume rows. The current
        # invocation's elapsed_seconds is not a full-cohort runtime on resume.
        if prediction.suffix == '.jsonl' and prediction.exists():
            batches, count = {}, 0
            with prediction.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if 'batch_id' not in item or 'batch_wall_seconds' not in item:
                        continue
                    seconds = finite_number(item['batch_wall_seconds'])
                    if seconds is None or seconds < 0:
                        raise ValueError('Invalid persisted inference batch time')
                    key = item['batch_id']
                    if key in batches and batches[key] != seconds:
                        raise ValueError('Inconsistent repeated batch timing')
                    batches[key] = seconds
                    count += 1
            if batches:
                rows.append({**context, 'cost_scope': 'inference_successful_batches_wall',
                             'seconds': sum(batches.values()), 'records': count,
                             'source_refs': json.dumps([provenance_ref(prediction,
                                 'sum batch_wall_seconds once per unique batch_id; all saved segments; excludes loading and discarded interrupted batches')])})
    return rows


def collect_runs(runs_dir: Path, partition: str = "val") -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    rows, costs, per_run, warnings = [], [], {}, []
    configuration_fingerprints = {}
    for metric_path in sorted(runs_dir.glob("*/evaluation*/metrics.json")):
        run = metric_path.parent.parent
        run_id = f"{run.name}/{metric_path.parent.name}"
        metrics = read_json(metric_path)
        if 'by_variant_task' in metrics:
            warnings.append(f'Skipped {run_id}: consistency uses a distinct metric schema.')
            continue
        cohort, family, found_partition = _query_context(metrics, run_id)
        if partition != "all" and partition != found_partition:
            warnings.append(f"Skipped {run_id}: partition {found_partition!r}, requested {partition!r}.")
            continue
        metadata, metadata_path, method, seed = _run_identity(run, metrics)
        if metadata.get("status") not in (None, "completed"):
            warnings.append(f"Skipped {run_id}: run status {metadata.get('status')!r}.")
            continue
        csv_path = metric_path.parent / "per_solute.csv"
        if not csv_path.exists():
            raise ValueError(f"Missing per-solute numerical source: {csv_path}")
        frame = pd.read_csv(csv_path, dtype={"solute_id": str, "task": str})
        if frame.duplicated(["solute_id", "task"]).any():
            raise ValueError(f"Duplicate solute/task in {csv_path}")
        context = {"run_id": run_id, "cohort_id": cohort, "split_family": family,
                   "partition": found_partition, "method": method, "seed": seed}
        comparison_config = dict(metadata.get("config", {}))
        comparison_config.pop("seed", None)
        comparison_payload = {"config": comparison_config, "code_sha256": metadata.get("code_sha256"),
                              "training_config": metadata.get('training_config'),
                              "training_inputs": metadata.get('training_inputs'),
                              "measurement_sha256": metadata.get('measurement_sha256'),
                              "split_sha256": metadata.get('split_sha256'),
                              "inference_measurements_sha256": metadata.get('fingerprint',{}).get('measurements_sha256'),
                              "inference_split_sha256": metadata.get('fingerprint',{}).get('manifest_sha256'),
                              "implementation_sha256": metadata.get("implementation_sha256"),
                              "model_revision": metadata.get("model_revision")}
        configuration_fingerprints[run_id] = hashlib.sha256(json.dumps(comparison_payload, sort_keys=True).encode()).hexdigest()
        per_run[run_id] = frame
        for task, task_metrics in metrics.get("by_task", {}).items():
            group = frame[frame.task == task]
            if len(group) != task_metrics.get("solutes"):
                raise ValueError(f"Solute count disagrees between metrics and CSV: {run_id}/{task}")
            for metric in METRICS:
                reported = finite_number(task_metrics.get(metric))
                if metric not in group:
                    if reported is not None:
                        raise ValueError(f"Missing {metric} in {csv_path}")
                    continue
                observations = pd.to_numeric(group[metric], errors="raise")
                if np.isinf(observations.to_numpy(dtype=float)).any():
                    raise ValueError(f"Infinite metric in {csv_path}")
                expected = finite_number(observations.mean())
                if (reported is None) != (expected is None) or (
                    reported is not None and not np.isclose(reported, expected, rtol=1e-9, atol=1e-12)
                ):
                    raise ValueError(f"Metric/CSV mismatch: {run_id}/{task}/{metric}: {reported} vs {expected}")
                if reported is None:
                    continue
                sources = [provenance_ref(metric_path, f"by_task.{task}.{metric}"),
                           provenance_ref(csv_path, f"task={task}; column={metric}")]
                if metadata_path:
                    sources.append(provenance_ref(metadata_path, "method, seed, config, implementation provenance"))
                rows.append({**context, "task": task, "metric": metric, "value": reported,
                             "solutes": len(group), "support_solutes": int(observations.notna().sum()),
                             "queries": task_metrics.get("queries"), "source_refs": json.dumps(sources)})
        costs.extend(_cost_rows(run, context, metadata, metadata_path, metrics))
    table = pd.DataFrame(rows, columns=RUN_COLUMNS)
    table.attrs["configuration_fingerprints"] = configuration_fingerprints
    return table, pd.DataFrame(costs, columns=COST_COLUMNS), per_run, warnings


def summarize_runs(run_table: pd.DataFrame, per_run: dict[str, pd.DataFrame], repetitions=10000, seed=20261025):
    rows = []
    grouping = ["cohort_id", "split_family", "partition", "method", "task", "metric"]
    if run_table.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    for values, group in run_table.groupby(grouping, sort=True, dropna=False):
        info = dict(zip(grouping, values))
        config_ids = run_table.attrs.get("configuration_fingerprints", {})
        if len({config_ids.get(run_id) for run_id in group.run_id}) > 1:
            raise ValueError(f"Different configurations/implementations cannot be pooled as learning seeds: {info}")
        task, metric = info["task"], info["metric"]
        series = []
        support_ids = None
        for row in group.itertuples():
            frame = per_run[row.run_id]
            current = frame[frame.task == task].set_index("solute_id")[metric].sort_index()
            if support_ids is None:
                support_ids = set(current.index)
            elif set(current.index) != support_ids:
                raise ValueError(f"Same cohort has unequal solute IDs: {row.run_id}")
            series.append(current.rename(row.run_id))
        # Require each seed to support a statistic. No seed is silently omitted
        # for a particular solute when forming the seed-averaged estimator.
        matrix = pd.concat(series, axis=1)
        solute_means = matrix.mean(axis=1, skipna=False)
        supported = solute_means.dropna()
        low, high = bootstrap_mean(supported, repetitions, seed)
        seeds = sorted({str(int(s)) if isinstance(s, (int, float)) and float(s).is_integer() else str(s)
                        for s in group.seed if pd.notna(s)})
        if len(seeds) != len(group) and len(group) > 1:
            raise ValueError(f"Repeated or unspecified learning seed within one method/cohort: {info}")
        refs = []
        for encoded in group.source_refs:
            refs.extend(json.loads(encoded))
        rows.append({**info, "value": finite_number(supported.mean()), "ci_low": low, "ci_high": high,
                     "bootstrap_repetitions": repetitions if low is not None else 0,
                     "bootstrap_seed": seed if low is not None else None,
                     "solutes": len(matrix), "support_solutes": len(supported),
                     "n_runs": len(group), "n_seeds": len(seeds), "seeds": ";".join(seeds),
                     "run_ids": ";".join(group.run_id), "source_refs": json.dumps(refs)})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def audit_tables(audit_path: Path | None, scaffold_amendment: Path | None = None):
    flow, splits = [], []
    if audit_path is None or not audit_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    audit = read_json(audit_path)
    for field, label in (("raw_rows", "Raw measurements"), ("pre_duplicate_rows", "After scope/structure filters"),
                         ('post_duplicate_measurements', 'After duplicate/conflict processing'),
                         ("final_measurements", "After source-conflict quarantine" if 'source_quarantined_records' in audit else "After duplicate/conflict processing")):
        value = audit.get(field)
        if value is not None:
            flow.append({"stage": label, "measurements": int(value),
                         "source_refs": json.dumps([provenance_ref(audit_path, field)])})
    effective_splits = dict(audit.get("splits", {}))
    amendment = None
    if scaffold_amendment is None:
        candidate = audit_path.with_name("scaffold_amendment.json")
        scaffold_amendment = candidate if candidate.exists() else None
    if scaffold_amendment is not None:
        amendment = read_json(scaffold_amendment)
        counts = amendment.get("counts", {})
        if not {"train", "val", "test"} <= set(counts):
            raise ValueError("Scaffold amendment lacks complete partition counts")
        manifest_key = "data/splits/scaffold.csv"
        expected_hash = amendment.get("amended_artifact_hashes", {}).get(manifest_key)
        if expected_hash:
            manifest_path = Path(__file__).resolve().parents[2] / manifest_key
            if not manifest_path.exists() or sha256_file(manifest_path) != expected_hash:
                raise ValueError("Scaffold amendment does not match the active split manifest")
        effective_splits["scaffold"] = counts
    for family, partitions in effective_splits.items():
        for partition, record in partitions.items():
            source = scaffold_amendment if family == "scaffold" and amendment is not None else audit_path
            field = f"counts.{partition}" if source == scaffold_amendment else f"splits.{family}.{partition}"
            row = {"split_family": family, "partition": partition, "solutes": record.get("solutes"),
                   "measurements": record.get("records"), "pair_queries": record.get("queries", {}).get("pair"),
                   "rank_queries": record.get("queries", {}).get("rank"),
                   "source_refs": json.dumps([provenance_ref(source, field)])}
            splits.append(row)
    return pd.DataFrame(flow), pd.DataFrame(splits)


def _tex_escape(value) -> str:
    text = str(value)
    for before, after in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
                          ("$", "\\$"), ("#", "\\#"), ("_", "\\_")):
        text = text.replace(before, after)
    return text


def write_latex_table(frame: pd.DataFrame, path: Path, columns: list[str], caption: str) -> None:
    lines = ["% Generated from observed artifacts; see artifact_manifest.json.",
             "\\begin{table}[htbp]", "\\centering\\small", f"\\caption{{{_tex_escape(caption)}}}",
             "\\begin{tabular}{" + "l" * len(columns) + "}", "\\toprule",
             " & ".join(_tex_escape(c.replace("_", " ")) for c in columns) + " " + chr(92) * 2, "\\midrule"]
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            value = row.get(col)
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                cells.append("--")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.4f}")
            else:
                cells.append(_tex_escape(value))
        lines.append(" & ".join(cells) + " " + chr(92) * 2)
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _pyplot(output: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save_figure(fig, folder: Path, stem: str):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", dpi=200, bbox_inches="tight")


def plot_metrics(summary: pd.DataFrame, output: Path) -> None:
    if summary.empty:
        return
    plt = _pyplot(output)
    for (cohort, family, partition, task), group in summary.groupby(["cohort_id", "split_family", "partition", "task"]):
        for metric in PRIMARY_METRICS.get(task, ()):
            selected = group[(group.metric == metric) & group.value.notna()].sort_values("method")
            if selected.empty:
                continue
            factor = 100 if metric in ("pair_accuracy", "top1") else 1
            fig, ax = plt.subplots(figsize=(max(5.5, len(selected) * 1.25), 4))
            x = np.arange(len(selected))
            ax.bar(x, selected.value * factor, color="#37688f", width=.65)
            for xpos, row in zip(x, selected.itertuples()):
                if pd.notna(row.ci_low) and pd.notna(row.ci_high):
                    ax.vlines(xpos, row.ci_low * factor, row.ci_high * factor, color="black", linewidth=1)
                    ax.hlines([row.ci_low * factor, row.ci_high * factor], xpos - .08, xpos + .08, color="black", linewidth=1)
            ax.set_xticks(x, [f"{row.method}\n{row.n_runs} run(s)" for row in selected.itertuples()], rotation=20, ha="right")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_title(f"{family} / {partition} / {task}\nquery cohort {str(cohort)[:10]}")
            has_ci = selected.ci_low.notna().any()
            fig.text(.01, -.035, "Bars: solute means; whiskers: solute-bootstrap 95% CI." if has_ci else
                     "Bars: solute means; no confidence intervals available.", fontsize=8)
            fig.tight_layout()
            stem = f"{family}_{partition}_{task}_{metric}_{str(cohort)[:10]}"
            _save_figure(fig, output / "figures", stem)
            plt.close(fig)


def plot_costs(summary: pd.DataFrame, costs: pd.DataFrame, output: Path) -> None:
    if summary.empty or costs.empty:
        return
    plt = _pyplot(output)
    group_cols = ["cohort_id", "split_family", "partition", "method", "cost_scope"]
    averaged = costs.groupby(group_cols, as_index=False).agg(seconds=("seconds", "mean"), timed_runs=("run_id", "nunique"))
    for (cohort, family, partition, scope), timed in averaged.groupby(["cohort_id", "split_family", "partition", "cost_scope"]):
        for task in ("pair", "rank"):
            quality = summary[(summary.cohort_id == cohort) & (summary.task == task) & (summary.metric == "regret")]
            merged = quality.merge(timed, on=["cohort_id", "split_family", "partition", "method"], how="inner")
            merged = merged[(merged.seconds > 0) & merged.value.notna()]
            if merged.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(merged.seconds, merged.value, color="#37688f", marker="o")
            for row in merged.itertuples():
                ax.annotate(f"{row.method} (timed n={row.timed_runs})", (row.seconds, row.value),
                            xytext=(5, 5), textcoords="offset points", fontsize=8)
            ax.set_xscale("log")
            ax.set_xlabel(scope.replace("_", " ") + " (seconds per run)")
            ax.set_ylabel(METRIC_LABELS["regret"])
            ax.set_title(f"{family} / {partition} / {task}\nquery cohort {str(cohort)[:10]}")
            fig.tight_layout()
            _save_figure(fig, output / "figures", f"cost_{scope}_{family}_{partition}_{task}_{str(cohort)[:10]}")
            plt.close(fig)


def plot_audit(flow: pd.DataFrame, splits: pd.DataFrame, output: Path) -> None:
    if flow.empty and splits.empty:
        return
    plt = _pyplot(output)
    if not flow.empty:
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.set_axis_off()
        for index, row in enumerate(flow.itertuples()):
            x = (index + .5) / len(flow)
            label = "\n".join(textwrap.wrap(row.stage, width=22))
            ax.text(x, .5, f"{label}\n{row.measurements:,}", ha="center", va="center",
                    transform=ax.transAxes, bbox={"boxstyle": "round,pad=.5", "facecolor": "#e7eff5", "edgecolor": "#37688f"}, fontsize=9)
            if index:
                ax.annotate("", xy=(x-.13, .5), xytext=(x-1/len(flow)+.13, .5), xycoords="axes fraction",
                            arrowprops={"arrowstyle": "->", "color": "#37688f"})
        ax.set_title("Measured-record curation flow")
        _save_figure(fig, output / "figures", "data_curation_flow")
        plt.close(fig)
    if not splits.empty:
        pivot = splits.pivot(index="split_family", columns="partition", values="solutes")
        columns = [name for name in ("train", "val", "test") if name in pivot]
        # A missing partition is not a zero; incomplete rows are omitted from this figure.
        pivot = pivot[columns].dropna()
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            bottom = np.zeros(len(pivot))
            for name, color in zip(columns, ("#37688f", "#e1a754", "#78a487")):
                ax.bar(pivot.index, pivot[name], bottom=bottom, label=name, color=color)
                bottom += pivot[name].to_numpy()
            ax.set_ylabel("Canonical solute identities")
            ax.set_title("Partition sizes (counts, not performance)")
            ax.legend(frameon=False)
            quarantined = splits[(splits.split_family == "scaffold") & (splits.partition == "excluded")]
            if not quarantined.empty:
                count = int(quarantined.solutes.iloc[0])
                fig.text(.01, -.02, f"Strict scaffold track: {count} development solutes quarantined; same protected test set.", fontsize=8)
            fig.tight_layout()
            _save_figure(fig, output / "figures", "partition_sizes")
            plt.close(fig)


def write_stage_report(summary: pd.DataFrame, costs: pd.DataFrame, output: Path, partition: str) -> Path:
    destination = output / "stage_report.md"
    heading = "Validation-stage report" if partition == "val" else "Evaluation-stage report"
    lines = [f"# {heading}", "", f"Requested partition: **{partition}**.", "",
             "This is an automatically rebuilt experimental report. It is not a submission-ready paper or evidence that the retrieval hypothesis has been confirmed.", "",
             "Values below come from the adjacent CSV tables; source paths, hashes and derivations are in `artifact_manifest.json`. Different query hashes are separate populations.", ""]
    if summary.empty:
        lines.append("No completed model evaluations were available for this requested partition. No performance numbers or intervals were invented.")
    else:
        lines.extend(["| Split / cohort | Method | Task | Metric | Value | 95% CI | Runs | Supported solutes |",
                      "|---|---|---|---|---:|---|---:|---:|"])
        for row in summary.itertuples():
            if row.metric not in PRIMARY_METRICS.get(row.task, ()):
                continue
            value = "missing" if pd.isna(row.value) else f"{row.value:.6f}"
            interval = "not available" if pd.isna(row.ci_low) or pd.isna(row.ci_high) else f"[{row.ci_low:.6f}, {row.ci_high:.6f}]"
            lines.append(f"| {row.split_family} / {str(row.cohort_id)[:10]} | {row.method} | {row.task} | {row.metric} | {value} | {interval} | {row.n_runs} | {row.support_solutes} |")
    lines.extend(["", "Accuracy/hit rates in the tables are fractions, while the corresponding figures display percentages. Regret is in decimal-log mole-fraction units. Best-solvent hits refer to the supplied block of at most eight candidates, not necessarily all measured solvents for that condition.", "",
                  "Intervals resample whole solutes after averaging the available learning seeds per solute; they do not fully describe training randomness. The exact seed list and bootstrap settings are in `metrics_by_method.csv`. No paired-difference significance claim is inferred from overlapping or non-overlapping marginal intervals.", "",
                  "Cost figures use only recorded times. Training-run wall time can include preprocessing and validation and is not called inference latency. Missing timing values are omitted, not replaced by zero."])
    if costs.empty:
        lines.extend(["", "No completed timing records were available; quality–cost figures were therefore not generated."])
    if partition == "val":
        lines.extend(["", "These results are validation observations used during development. Final-test conclusions, condition-aware retrieval gains, scaffold robustness and publication readiness remain separate checks."])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def build_report(runs_dir: str | Path, output: str | Path, audit_path: str | Path | None = None,
                 partition="val", repetitions=10000, seed=20261025, plots=True,
                 scaffold_amendment: str | Path | None = None) -> dict:
    runs_dir, output = Path(runs_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit_path = Path(audit_path) if audit_path else None
    run_table, costs, per_run, warnings = collect_runs(runs_dir, partition)
    summary = summarize_runs(run_table, per_run, repetitions, seed)
    amendment_path = Path(scaffold_amendment) if scaffold_amendment else None
    flow, splits = audit_tables(audit_path, amendment_path)
    previous_manifest = output / "artifact_manifest.json"
    if previous_manifest.exists():
        previous = read_json(previous_manifest)
        previous_paths = []
        for item in previous.get("outputs", []):
            path = Path(item["path"]).resolve()
            if not path.is_relative_to(output.resolve()):
                raise ValueError("Previous report manifest contains a path outside the output directory")
            if path.exists() and sha256_file(path) != item["sha256"]:
                raise FileExistsError(f"Generated artifact was edited; use a new report output directory: {path}")
            previous_paths.append(path)
        # Only remove this reporter's unchanged, reproducible outputs. This
        # prevents a deleted run from surviving in a later report as a stale plot.
        for path in previous_paths:
            path.unlink(missing_ok=True)
    tables = {"metrics_by_run": run_table, "metrics_by_method": summary, "costs_by_run": costs,
              "data_flow": flow, "split_counts": splits}
    outputs = []
    for name, table in tables.items():
        destination = output / f"{name}.csv"
        table.to_csv(destination, index=False)
        outputs.append(destination)
    outputs.append(write_stage_report(summary, costs, output, partition))
    # Separate populations avoid an apparently comparable mixed-split table.
    if not summary.empty:
        for (cohort, family, found_partition, task), group in summary.groupby(["cohort_id", "split_family", "partition", "task"]):
            selected = group[group.metric.isin(PRIMARY_METRICS.get(task, ()))]
            destination = output / f"metrics_{family}_{found_partition}_{task}_{str(cohort)[:10]}.tex"
            write_latex_table(selected, destination, ["method", "metric", "value", "ci_low", "ci_high", "n_runs"],
                              f"{family} {found_partition}, {task}; query cohort {str(cohort)[:10]}. Values are fractions unless labelled regret; intervals resample solutes, not seeds.")
            outputs.append(destination)
    for name, table, columns in (("data_flow", flow, ["stage", "measurements"]),
                                  ("split_counts", splits, ["split_family", "partition", "solutes", "measurements", "pair_queries", "rank_queries"])):
        if not table.empty:
            destination = output / f"{name}.tex"
            write_latex_table(table, destination, columns, "Observed audit counts; source fields and hashes are recorded in the artifact manifest.")
            outputs.append(destination)
    if plots:
        plot_metrics(summary, output)
        plot_costs(summary, costs, output)
        plot_audit(flow, splits, output)
        outputs.extend(sorted((output / "figures").glob("*.pdf")))
        outputs.extend(sorted((output / "figures").glob("*.png")))
    row_provenance = {}
    for name, table in tables.items():
        row_provenance[name] = [{"row": index, "sources": json.loads(row.get("source_refs", "[]"))}
                                for index, row in table.iterrows()]
    manifest = {
        "schema_version": 1, "partition_requested": partition,
        "report_code": provenance_ref(Path(__file__)), "bootstrap_repetitions_requested": repetitions,
        "bootstrap_seed": seed, "evaluated_runs": sorted(per_run), "warnings": warnings,
        "has_model_results": not run_table.empty,
        "table_row_provenance": row_provenance,
        "outputs": [provenance_ref(path) for path in outputs],
        "derivations": {
            "metrics_by_run": "Values in evaluation/metrics.json cross-checked against the means in evaluation/per_solute.csv.",
            "metrics_by_method": "For each common query hash and method, average metric per solute over all recorded learning seeds; then average solutes. All seeds must support the statistic. Percentile CI resamples these solute-level means.",
            "costs_by_run": "Only finite observed elapsed times from completed metadata or inference timing files; no imputation. Training and inference scopes are separate.",
            "figures": "Deterministic visualization of the corresponding CSV rows. Accuracy is multiplied by 100 only for display. No CI is drawn without a recorded bootstrap interval.",
            "audit": "Exact scalar counts from data_audit.json, with active scaffold counts overridden by scaffold_amendment.json when present; no measurement or target recomputation. Excluded scaffold compounds are quarantined development identities, not model inputs.",
        },
        "limitations": ["Bootstrap intervals do not fully quantify training-seed variability.",
                        "Training elapsed time may include preprocessing and validation; it is not inference latency.",
                        "Missing costs and undefined rank correlations are not replaced by zero.",
                        "A generated report is not evidence of frozen-test or publication readiness."],
    }
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/audit/data_audit.json"))
    parser.add_argument("--scaffold-amendment", type=Path, help="Override scaffold counts; defaults to a sibling scaffold_amendment.json if present")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/generated"))
    parser.add_argument("--partition", choices=("val", "test", "train", "all"), default="val")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261025)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    manifest = build_report(args.runs_dir, args.output, args.audit, args.partition,
                            args.bootstrap_repetitions, args.bootstrap_seed, plots=not args.no_plots,
                            scaffold_amendment=args.scaffold_amendment)
    print(json.dumps({"evaluated_runs": manifest["evaluated_runs"], "has_model_results": manifest["has_model_results"],
                      "output": str(args.output), "warnings": manifest["warnings"]}, indent=2))


if __name__ == "__main__":
    main()

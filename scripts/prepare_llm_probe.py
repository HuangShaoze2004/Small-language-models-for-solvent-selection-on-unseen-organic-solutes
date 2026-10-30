"""Prepare a label-independent validation probe, never launch a GPU job."""
from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter
from pathlib import Path

from solvent_select.llm import read_jsonl, sample_queries
from solvent_select.views import filter_jsonl, ids_sha256, sha256_file, stable_key, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-source", default="data/processed/identity/val_queries.jsonl")
    parser.add_argument("--targets-source", default="data/targets/identity/val.jsonl")
    parser.add_argument("--output-dir", default="data/views/pilot")
    parser.add_argument("--limit", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--resume-checkpoint", help="Optional read-only callback-state audit of an interrupted checkpoint")
    args = parser.parse_args(argv)
    queries = read_jsonl(args.queries_source)
    if {query.get("split") for query in queries} != {"val"}:
        raise ValueError("The diagnostic probe must contain only validation queries")
    source_solutes = {str(query["solute_id"]) for query in queries}
    if args.limit < len(source_solutes):
        raise ValueError("Probe limit is too small to preserve all evaluable validation solutes")
    selected = sample_queries(queries, args.limit, args.seed)
    selected_ids = [query["query_id"] for query in selected]
    if {str(query["solute_id"]) for query in selected} != source_solutes:
        raise AssertionError("Round-robin sampling failed to retain all source solutes")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "probe_manifest.json"
    identity = {
        "source_query_path": str(args.queries_source),
        "source_query_sha256": sha256_file(args.queries_source),
        "selection_function": "solvent_select.llm.sample_queries",
        "selection_function_sha256": stable_key(0, "source", inspect.getsource(sample_queries)),
        "seed": args.seed, "limit": args.limit,
        "selected_ids_sha256": ids_sha256(selected_ids),
    }
    manifest = {
        "status": "SELECTION_FIXED_TARGETS_NOT_READ",
        "purpose": "Validation-only throughput and behavior diagnosis; not a frozen final benchmark",
        "selection_independent_of_labels": True,
        "selection_fields": ["query_id", "solute_id"],
        "identity": identity,
        "query_ids": selected_ids,
        "source_queries": len(queries), "selected_queries": len(selected),
        "source_evaluable_solutes": len(source_solutes), "selected_evaluable_solutes": len(source_solutes),
        "selected_task_counts": dict(Counter(query["task"] for query in selected)),
        "all_evaluable_solutes_retained": True,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity:
            raise FileExistsError("Existing probe selection differs; choose another --output-dir")
        if previous["status"] == "VALIDATION_PROBE_NOT_FINAL_PROTOCOL":
            for artifact in previous["artifacts"].values():
                if sha256_file(artifact["path"]) != artifact["sha256"]:
                    raise ValueError("Probe artifact changed after generation")
            if sha256_file(args.targets_source) != previous["source_target_sha256"]:
                raise ValueError("Source validation targets changed after generation")
            print(json.dumps({key: previous[key] for key in (
                "status", "selected_queries", "selected_evaluable_solutes", "selected_task_counts")}))
            return
    # Persist the exact selection before opening a target file.
    write_json(manifest_path, manifest)
    query_output, target_output = output / "val_queries.jsonl", output / "val_targets.jsonl"
    filter_jsonl(args.queries_source, query_output, selected_ids)
    filter_jsonl(args.targets_source, target_output, selected_ids)
    manifest.update(
        status="VALIDATION_PROBE_NOT_FINAL_PROTOCOL",
        source_target_path=str(args.targets_source), source_target_sha256=sha256_file(args.targets_source),
        artifacts={"queries": {"path": str(query_output), "sha256": sha256_file(query_output)},
                   "targets": {"path": str(target_output), "sha256": sha256_file(target_output)}})
    if args.resume_checkpoint:
        # Read only an already-written checkpoint; never mutate the live run.
        import transformers
        import ast
        from transformers import TrainingArguments
        from solvent_select import llm
        checkpoint = Path(args.resume_checkpoint)
        state_path = checkpoint / "trainer_state.json"
        state = json.loads(state_path.read_text())
        callback = state.get("stateful_callbacks", {}).get("EarlyStoppingCallback", {})
        counter = callback.get("attributes", {}).get("early_stopping_patience_counter")
        default_restore = TrainingArguments.__dataclass_fields__["restore_callback_states_from_checkpoint"].default
        source = inspect.getsource(llm.train)
        explicit_restore = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TrainingArguments"
            and any(keyword.arg == "restore_callback_states_from_checkpoint"
                    and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                    for keyword in node.keywords)
            for node in ast.walk(ast.parse(source)))
        manifest["readonly_resume_review"] = {
            "transformers_version": transformers.__version__,
            "checkpoint": str(checkpoint), "trainer_state_sha256": sha256_file(state_path),
            "global_step": state.get("global_step"),
            "saved_early_stopping_patience_counter": counter,
            "default_restore_callback_states_from_checkpoint": default_restore,
            "current_llm_training_explicitly_enables_callback_restore": explicit_restore,
            "reviewed_llm_source_sha256": sha256_file(llm.__file__),
            "counter_effect_for_this_interruption": (
                "No counter change: saved counter is zero, equal to new callback initialization"
                if counter == 0 else
                "Potential patience reset: saved counter is nonzero; requires correction before formal training"),
            "required_formal_fix": "Enable restore_callback_states_from_checkpoint and test resume with nonzero patience",
            "current_training_files_modified": False,
        }
    write_json(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in (
        "status", "selected_queries", "selected_evaluable_solutes", "selected_task_counts")}))


if __name__ == "__main__":
    main()

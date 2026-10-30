"""Label-independent resource-budgeted query views, explicitly not frozen.

Selection reads only query identity/task/physical-condition metadata. Selected
IDs are saved before any target file is opened. Original files are untouched.
This module never computes an evaluation metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

SELECTION_FIELDS = ("query_id", "solute_id", "task", "condition_id", "split")
DEFAULT_SEED = 20261025


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed, domain, value):
    return hashlib.sha256(f"{seed}|{domain}|{value}".encode()).hexdigest()


def ids_sha256(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_query_metadata(path, expected_split):
    rows, seen = [], set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            query = json.loads(line)
            row = {key: str(query[key]) for key in SELECTION_FIELDS}
            if row["query_id"] in seen:
                raise ValueError(f"Duplicate query ID in {path}")
            seen.add(row["query_id"])
            if row["split"] != expected_split or row["task"] not in {"pair", "rank"}:
                raise ValueError(f"Unexpected split or task in {path}")
            rows.append(row)
    return rows


def condition_round_robin(rows, seed, domain):
    """Take one query per physical condition before revisiting a condition."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition_id"]].append(row)
    keys = sorted(groups, key=lambda key: stable_key(seed, domain + "/condition", key))
    queues = [deque(sorted(groups[key], key=lambda row: stable_key(seed, domain + "/query", row["query_id"])))
              for key in keys]
    result = []
    while queues:
        remaining = []
        for queue in queues:
            result.append(queue.popleft())
            if queue:
                remaining.append(queue)
        queues = remaining
    return result


def task_solute_round_robin(rows, limit, seed, domain):
    """Balance each task across solutes, then across physical conditions."""
    if limit <= 0:
        return []
    groups = defaultdict(list)
    for row in rows:
        groups[row["solute_id"]].append(row)
    keys = sorted(groups, key=lambda key: stable_key(seed, domain + "/solute", key))
    queues = [deque(condition_round_robin(groups[key], seed, domain + "/" + key)) for key in keys]
    result = []
    while queues and len(result) < limit:
        remaining = []
        for queue in queues:
            result.append(queue.popleft()["query_id"])
            if len(result) >= limit:
                return result
            if queue:
                remaining.append(queue)
        queues = remaining
    return result


def select_train_ids(metadata, limit=10000, seed=DEFAULT_SEED):
    if limit < 1:
        raise ValueError("Training limit must be positive")
    tasks = {task: [row for row in metadata if row["task"] == task] for task in ("pair", "rank")}
    total = min(limit, len(metadata))
    rank_target = total // 4
    quotas = {"pair": min(total - rank_target, len(tasks["pair"])),
              "rank": min(rank_target, len(tasks["rank"]))}
    remaining = total - sum(quotas.values())
    for task in ("pair", "rank"):
        addition = min(remaining, len(tasks[task]) - quotas[task])
        quotas[task] += addition
        remaining -= addition
    selected = []
    for task in ("pair", "rank"):
        selected.extend(task_solute_round_robin(tasks[task], quotas[task], seed, "train/" + task))
    return sorted(selected, key=lambda value: stable_key(seed, "output", value))


def select_eval_ids(metadata, pair_limit=32, seed=DEFAULT_SEED):
    if pair_limit < 1:
        raise ValueError("Evaluation pair limit per solute must be positive")
    selected = [row["query_id"] for row in metadata if row["task"] == "rank"]
    groups = defaultdict(list)
    for row in metadata:
        if row["task"] == "pair":
            groups[row["solute_id"]].append(row)
    for solute in sorted(groups):
        ordered = condition_round_robin(groups[solute], seed, "eval/pair/" + solute)
        selected.extend(row["query_id"] for row in ordered[:pair_limit])
    return sorted(selected, key=lambda value: stable_key(seed, "output", value))


def summarize(metadata, selected_ids):
    chosen = set(selected_ids)
    selected = [row for row in metadata if row["query_id"] in chosen]
    source_solutes = {row["solute_id"] for row in metadata}
    selected_solutes = {row["solute_id"] for row in selected}
    pairs = Counter(row["solute_id"] for row in selected if row["task"] == "pair")
    return {
        "source_queries": len(metadata), "selected_queries": len(selected),
        "source_task_counts": dict(Counter(row["task"] for row in metadata)),
        "selected_task_counts": dict(Counter(row["task"] for row in selected)),
        "source_solutes": len(source_solutes), "selected_solutes": len(selected_solutes),
        "missing_solute_ids": sorted(source_solutes - selected_solutes),
        "source_physical_conditions": len({row["condition_id"] for row in metadata}),
        "selected_physical_conditions": len({row["condition_id"] for row in selected}),
        "max_pair_questions_per_solute": max(pairs.values(), default=0),
        "all_rank_questions_retained": all(row["query_id"] in chosen for row in metadata if row["task"] == "rank"),
    }


def filter_jsonl(source, destination, selected_ids):
    """Inspect only query_id and copy selected payloads without altering values."""
    wanted = set(selected_ids)
    if len(wanted) != len(selected_ids):
        raise ValueError("Duplicate selection IDs")
    kept = {}
    with open(source, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            identifier = str(json.loads(line)["query_id"])
            if identifier in wanted:
                if identifier in kept:
                    raise ValueError(f"Duplicate selected ID in {source}")
                kept[identifier] = line.rstrip("\n") + "\n"
    if wanted - set(kept):
        raise ValueError(f"Missing {len(wanted - set(kept))} selected IDs in {source}")
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for identifier in selected_ids:
            handle.write(kept[identifier])
    temporary.replace(destination)


def build_view(protocol, processed_root, targets_root, output_root,
               train_limit=10000, pair_eval_limit=32, seed=DEFAULT_SEED):
    processed_root, targets_root, output_root = map(Path, (processed_root, targets_root, output_root))
    destination = output_root / protocol
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "view_manifest.json"
    manifest = {
        "schema_version": 1, "status": "SELECTION_FIXED_TARGETS_NOT_READ", "protocol": protocol,
        "selection_independent_of_labels": True, "selection_fields": list(SELECTION_FIELDS),
        "target_values_used_for_selection": False,
        "parameters": {"train_limit": train_limit, "pair_eval_limit_per_solute": pair_eval_limit,
                       "train_pair_to_rank_target_ratio": [3, 1], "seed": seed},
        "implementation_sha256": sha256_file(__file__), "partitions": {},
    }
    selections = {}
    # Finish every metadata-only selection before opening ANY target file.
    for split in ("train", "val", "test"):
        path = processed_root / protocol / f"{split}_queries.jsonl"
        metadata = read_query_metadata(path, split)
        selected = (select_train_ids(metadata, train_limit, seed) if split == "train" else
                    select_eval_ids(metadata, pair_eval_limit, seed))
        selections[split] = selected
        manifest["partitions"][split] = {
            "source_query_path": str(path), "source_query_sha256": sha256_file(path),
            "selected_ids_sha256": ids_sha256(selected), "summary": summarize(metadata, selected),
        }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        keys = ("protocol", "parameters", "implementation_sha256")
        mismatch = any(previous.get(key) != manifest[key] for key in keys) or any(
            previous["partitions"][split][key] != manifest["partitions"][split][key]
            for split in selections for key in ("source_query_sha256", "selected_ids_sha256"))
        if mismatch:
            raise FileExistsError("Existing view differs; choose a new --output-root")
        if previous.get("status") == "CANDIDATE_NOT_FROZEN":
            for split in selections:
                info = previous["partitions"][split]
                if sha256_file(info["selection_path"]) != info["selection_sha256"]:
                    raise ValueError("Existing selection IDs differ from their manifest")
                for kind in ("query", "target"):
                    artifact = info[kind + "_view"]
                    if sha256_file(artifact["path"]) != artifact["sha256"]:
                        raise ValueError("Existing view artifact differs from its manifest")
                if sha256_file(info["source_target_path"]) != info["source_target_sha256"]:
                    raise ValueError("Source targets changed after this view was generated")
            return previous
    for split, selected in selections.items():
        path = destination / f"{split}_selection.json"
        write_json(path, {"status": "CANDIDATE_NOT_FROZEN", "protocol": protocol, "split": split,
                          "selection_independent_of_labels": True, "parameters": manifest["parameters"],
                          "query_ids": selected, "selected_ids_sha256": ids_sha256(selected),
                          "source_query_sha256": manifest["partitions"][split]["source_query_sha256"]})
        manifest["partitions"][split].update(selection_path=str(path), selection_sha256=sha256_file(path))
    write_json(manifest_path, manifest)
    # Selected IDs and their hashes now exist independently of all targets.
    for split, selected in selections.items():
        query_source = processed_root / protocol / f"{split}_queries.jsonl"
        target_source = targets_root / protocol / f"{split}.jsonl"
        query_out, target_out = destination / f"{split}_queries.jsonl", destination / f"{split}_targets.jsonl"
        filter_jsonl(query_source, query_out, selected)
        filter_jsonl(target_source, target_out, selected)
        manifest["partitions"][split].update(
            source_target_path=str(target_source), source_target_sha256=sha256_file(target_source),
            query_view={"path": str(query_out), "sha256": sha256_file(query_out)},
            target_view={"path": str(target_out), "sha256": sha256_file(target_out)})
    manifest["status"] = "CANDIDATE_NOT_FROZEN"
    write_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("identity", "scaffold", "all"), default="all")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--targets-root", default="data/targets")
    parser.add_argument("--output-root", default="data/views")
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--pair-eval-limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    for protocol in (("identity", "scaffold") if args.protocol == "all" else (args.protocol,)):
        manifest = build_view(protocol, args.processed_root, args.targets_root, args.output_root,
                              args.train_limit, args.pair_eval_limit, args.seed)
        print(json.dumps({"protocol": protocol, "status": manifest["status"], "parameters": manifest["parameters"],
                          "summaries": {split: info["summary"] for split, info in manifest["partitions"].items()}},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

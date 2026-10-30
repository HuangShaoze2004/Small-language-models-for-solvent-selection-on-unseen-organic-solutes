"""Synthetic fixtures check label-independent budget views, not model quality."""
import json
from collections import Counter
from pathlib import Path

import pytest

from solvent_select import views


def rows(split="train", solutes=4, conditions=3, pairs=10):
    result = []
    for solute in range(solutes):
        for condition in range(conditions):
            base = {"solute_id": f"{split}-s{solute}", "condition_id": f"{split}-s{solute}-c{condition}",
                    "split": split}
            for pair in range(pairs):
                result.append({**base, "query_id": f"{split}-s{solute}-c{condition}-p{pair}",
                               "task": "pair", "block_id": f"irrelevant-{pair % 2}"})
            result.append({**base, "query_id": f"{split}-s{solute}-c{condition}-rank", "task": "rank"})
    return result


def test_training_quota_task_and_solute_balance():
    metadata = rows()
    selected = set(views.select_train_ids(metadata, limit=40))
    chosen = [row for row in metadata if row["query_id"] in selected]
    assert Counter(row["task"] for row in chosen) == {"pair": 30, "rank": 10}
    assert len({row["solute_id"] for row in chosen}) == 4
    for task in ("pair", "rank"):
        counts = Counter(row["solute_id"] for row in chosen if row["task"] == task)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_training_uses_available_other_task_when_quota_exhausted():
    metadata = [row for row in rows() if row["task"] == "pair"]
    selected = views.select_train_ids(metadata, limit=40)
    assert len(selected) == 40
    assert len(views.select_train_ids(metadata, limit=10000)) == len(metadata)


def test_evaluation_keeps_all_rank_caps_pairs_and_balances_physical_conditions():
    metadata = rows(split="test", conditions=3, pairs=20)
    selected = set(views.select_eval_ids(metadata, pair_limit=5))
    assert all(row["query_id"] in selected for row in metadata if row["task"] == "rank")
    for solute in {row["solute_id"] for row in metadata}:
        chosen = [row for row in metadata if row["query_id"] in selected
                  and row["solute_id"] == solute and row["task"] == "pair"]
        assert len(chosen) == 5
        counts = Counter(row["condition_id"] for row in chosen)
        assert sorted(counts.values()) == [1, 2, 2]


def test_selection_ignores_values_order_and_nonselection_metadata():
    original = rows()
    changed = [dict(row, choice="CHANGED", logx=-999, values={"A": 12345}, block_id="DIFFERENT")
               for row in reversed(original)]
    assert views.select_train_ids(original, 40) == views.select_train_ids(changed, 40)
    assert views.select_eval_ids(original, 5) == views.select_eval_ids(changed, 5)


def test_metadata_projection_rejects_duplicate_ids_and_wrong_split(tmp_path):
    path = tmp_path / "queries.jsonl"
    sample = rows()[0]
    path.write_text(json.dumps(dict(sample, values={"A": -2}, answer="secret")) + "\n")
    projected = views.read_query_metadata(path, "train")
    assert set(projected[0]) == set(views.SELECTION_FIELDS)
    with pytest.raises(ValueError, match="Unexpected"):
        views.read_query_metadata(path, "test")
    path.write_text((json.dumps(sample) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        views.read_query_metadata(path, "train")


def make_sources(root, value):
    processed, targets = root / "processed", root / "targets"
    for split in ("train", "val", "test"):
        (processed / "identity").mkdir(parents=True, exist_ok=True)
        (targets / "identity").mkdir(parents=True, exist_ok=True)
        metadata = rows(split=split)
        (processed / "identity" / f"{split}_queries.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in metadata))
        (targets / "identity" / f"{split}.jsonl").write_text(
            "".join(json.dumps({"query_id": row["query_id"], "values": {"A": value}}) + "\n"
                    for row in reversed(metadata)))
    return processed, targets


def test_every_selection_is_persisted_before_any_target_is_read(tmp_path, monkeypatch):
    processed, targets = make_sources(tmp_path, -2)
    output = tmp_path / "views"
    real_filter = views.filter_jsonl
    target_calls = []
    def checked_filter(source, destination, selected):
        if Path(source).parent.parent == targets:
            manifest = json.loads((output / "identity/view_manifest.json").read_text())
            assert manifest["status"] == "SELECTION_FIXED_TARGETS_NOT_READ"
            for split in ("train", "val", "test"):
                selected_path = output / "identity" / f"{split}_selection.json"
                assert selected_path.is_file()
                assert json.loads(selected_path.read_text())["selection_independent_of_labels"]
            target_calls.append(source)
        return real_filter(source, destination, selected)
    monkeypatch.setattr(views, "filter_jsonl", checked_filter)
    manifest = views.build_view("identity", processed, targets, output, 40, 5)
    assert len(target_calls) == 3
    assert manifest["status"] == "CANDIDATE_NOT_FROZEN"
    assert manifest["selection_independent_of_labels"]
    assert not manifest["target_values_used_for_selection"]
    for split in ("train", "val", "test"):
        info = manifest["partitions"][split]
        queries = [json.loads(line) for line in Path(info["query_view"]["path"]).read_text().splitlines()]
        labels = [json.loads(line) for line in Path(info["target_view"]["path"]).read_text().splitlines()]
        assert [q["query_id"] for q in queries] == [t["query_id"] for t in labels]
        assert all(target["values"] == {"A": -2} for target in labels)


def test_target_perturbation_does_not_change_selected_ids(tmp_path):
    processed, targets = make_sources(tmp_path / "first", -2)
    first = views.build_view("identity", processed, targets, tmp_path / "view1", 40, 5)
    processed, targets = make_sources(tmp_path / "second", 987654)
    second = views.build_view("identity", processed, targets, tmp_path / "view2", 40, 5)
    assert [first["partitions"][split]["selected_ids_sha256"] for split in ("train", "val", "test")] == [
        second["partitions"][split]["selected_ids_sha256"] for split in ("train", "val", "test")]


def test_view_reuse_verifies_artifacts_and_does_not_change_sources(tmp_path):
    processed, targets = make_sources(tmp_path, -2)
    source_hashes = {str(path): views.sha256_file(path)
                     for directory in (processed, targets) for path in directory.rglob("*.jsonl")}
    output = tmp_path / "views"
    first = views.build_view("identity", processed, targets, output, 40, 5)
    second = views.build_view("identity", processed, targets, output, 40, 5)
    assert first == second
    assert source_hashes == {str(path): views.sha256_file(path)
                            for directory in (processed, targets) for path in directory.rglob("*.jsonl")}
    with pytest.raises(FileExistsError, match="differs"):
        views.build_view("identity", processed, targets, output, 44, 5)
    selection = output / "identity/test_selection.json"
    selection.write_text("{}")
    with pytest.raises(ValueError, match="selection IDs"):
        views.build_view("identity", processed, targets, output, 40, 5)


def test_target_filter_does_not_silently_accept_missing_or_duplicate_rows(tmp_path):
    source, destination = tmp_path / "labels.jsonl", tmp_path / "selected.jsonl"
    source.write_text(json.dumps({"query_id": "a", "values": {"A": -2}}) + "\n")
    with pytest.raises(ValueError, match="Missing"):
        views.filter_jsonl(source, destination, ["a", "b"])
    source.write_text(source.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        views.filter_jsonl(source, destination, ["a"])

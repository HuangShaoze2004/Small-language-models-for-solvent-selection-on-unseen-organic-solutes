"""Global holdout protection and real-measurement integrity for strict extrapolation."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from solvent_select.baselines import load_manifest, validate_group_isolation
from solvent_select.strict_split import (
    EXPECTED_CURRENT_COUNTS, assign_development_groups, derive_manifest,
    protected_hashes, validate_real_conditions,
)


def test_development_assignment_is_order_invariant_and_grouped():
    groups = {f"g{i}": [f"s{i}_{j}" for j in range(3)] for i in range(30)}
    first = assign_development_groups(groups, 17)
    second = assign_development_groups(dict(reversed(list(groups.items()))), 17)
    assert first == second
    assert set(first.values()) == {"train", "val"}
    for members in groups.values():
        assert len({first[solute] for solute in members}) == 1


def test_global_holdout_protection_and_scaffold_neighbors_excluded():
    rows = [{"solute_id": "held", "leakage_group": "parent_held", "scaffold_group": "shared", "split": "test"},
            {"solute_id": "neighbor", "leakage_group": "parent_neighbor", "scaffold_group": "shared", "split": "train"},
            {"solute_id": "other", "leakage_group": "parent_other", "scaffold_group": "different", "split": "val"}]
    retained, excluded = derive_manifest(pd.DataFrame(rows), 17)
    assert set(retained.loc[retained.split.eq("test"), "solute_id"]) == {"held"}
    assert set(excluded.solute_id) == {"neighbor"}
    assert "neighbor" not in set(retained.solute_id)
    assert set(retained.scaffold_group) == {"shared", "different"}


@pytest.fixture
def real_root():
    root = Path(__file__).resolve().parents[1]
    if not (root / "artifacts/audit/scaffold_amendment.json").exists():
        pytest.skip("Run the strict-split amendment first")
    return root


def test_real_amendment_counts_and_no_cross_protocol_overlap(real_root):
    identity = pd.read_csv(real_root / "data/splits/identity.csv")
    strict = pd.read_csv(real_root / "data/splits/scaffold.csv")
    excluded = pd.read_csv(real_root / "data/splits/scaffold_excluded.csv")
    measurement = pd.read_parquet(real_root / "data/processed/measurements_scaffold.parquet", columns=["solute_id"])
    test_ids = set(identity.loc[identity.split.eq("test"), "solute_id"])
    assert test_ids == set(strict.loc[strict.split.eq("test"), "solute_id"])
    assert not test_ids.intersection(strict.loc[strict.split.ne("test"), "solute_id"])
    assert not set(excluded.solute_id).intersection(strict.solute_id)
    assert set(strict.solute_id) | set(excluded.solute_id) == set(identity.solute_id)
    for column in ["leakage_group", "scaffold_group"]:
        assert strict.groupby(column).split.nunique().max() == 1
    for split in ["train", "val", "test"]:
        ids = set(strict.loc[strict.split.eq(split), "solute_id"])
        assert len(ids) == EXPECTED_CURRENT_COUNTS[split]["solutes"]
        assert int(measurement.solute_id.isin(ids).sum()) == EXPECTED_CURRENT_COUNTS[split]["records"]
    assert len(excluded) == EXPECTED_CURRENT_COUNTS["excluded"]["solutes"]
    assert int(excluded.record_count.sum()) == EXPECTED_CURRENT_COUNTS["excluded"]["records"]


def test_wrong_full_measurement_table_fails_closed(real_root):
    strict = load_manifest(real_root / "data/splits/scaffold.csv")
    with pytest.raises(ValueError, match="missing from split manifest"):
        validate_group_isolation(real_root / "data/processed/measurements.parquet", strict)
    validate_group_isolation(real_root / "data/processed/measurements_scaffold.parquet", strict)
    validate_group_isolation(real_root / "data/processed/measurements_scaffold.parquet", strict, "scaffold_group")


def test_all_identity_bytes_unchanged_and_test_tasks_identical(real_root):
    audit = json.loads((real_root / "artifacts/audit/scaffold_amendment.json").read_text())
    assert audit["protected_identity_hashes_before"] == audit["protected_identity_hashes_after"]
    assert protected_hashes(real_root) == audit["protected_identity_hashes_before"]
    assert audit["observed_global_audit_hashes_before"] == audit["observed_global_audit_hashes_after"]
    assert audit["superseded_status"] == "superseded-unused"
    old_path = real_root / audit["superseded_manifest"]
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == audit["superseded_manifest_sha256"]
    for section, filename in [("processed", "test_queries.jsonl"), ("targets", "test.jsonl")]:
        assert (real_root / "data" / section / "identity" / filename).read_bytes() == (real_root / "data" / section / "scaffold" / filename).read_bytes()


def test_real_strict_queries_use_only_existing_comparable_measurements(real_root):
    columns = ["record_id", "solute_id", "solvent_smiles", "source_doi", "temperature_k"]
    frame = pd.read_parquet(real_root / "data/processed/measurements_scaffold.parquet", columns=columns)
    checks = validate_real_conditions(real_root, frame)
    assert checks["passed"]
    assert set(checks["queries_checked"]) == {"train", "val", "test"}
    assert not checks["labels_used"]

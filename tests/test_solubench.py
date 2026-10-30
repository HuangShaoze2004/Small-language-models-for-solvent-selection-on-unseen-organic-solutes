"""The supplemental reservation must be independent of benchmark outcomes."""
import copy

import pandas as pd
import pytest

from solvent_select.solubench_audit import (
    condition_metadata, identify_solute, reserve_metadata,
    sanitize_task_row, selected_source_rows,
)


def example():
    row = {"id": "untrusted-id", "DOI": "10.1/example", "SMILES": "CCO", "Temperature_K": 298.15,
           "Solvent_A": "water", "Solvent_B": "methanol", "Answer": "A", "LogS_A": 100,
           "LogS_B": -100, "Delta_LogS": 200, "canary": "do-not-train"}
    identity = identify_solute("CCO")
    manifest = pd.DataFrame([{**identity, "split": "test"}])
    selected = {("10.1/example", "CCO", "water"): {"temperature_k": 298.15, "solvent_smiles": "O", "raw_row": 1},
                ("10.1/example", "CCO", "methanol"): {"temperature_k": 298.15, "solvent_smiles": "CO", "raw_row": 2}}
    return row, manifest, selected


def test_target_fields_and_canary_cannot_cross_allowlist():
    row, _, _ = example()
    public = sanitize_task_row(row, "task1")
    assert set(public) == {"id", "DOI", "SMILES", "Temperature_K", "Solvent_A", "Solvent_B"}
    assert "Answer" not in public and "canary" not in public


def test_reservation_is_independent_of_answers_and_solubilities():
    row, manifest, selected = example()
    before, stats = reserve_metadata([sanitize_task_row(row, "task1")], "task1", manifest, selected)
    changed = dict(row, Answer="B", LogS_A=-10000, LogS_B=10000, Delta_LogS=999999)
    after, later_stats = reserve_metadata([sanitize_task_row(changed, "task1")], "task1", manifest, selected)
    assert before == after and stats == later_stats
    assert before[0]["condition_eligible"]


def test_parent_group_overlap_excludes_salt_variant():
    row, _, selected = example()
    row["SMILES"] = "CC[O-].[Na+]"
    test_identity = identify_solute(row["SMILES"])
    train_identity = identify_solute("CCO")
    assert test_identity["solute_id"] != train_identity["solute_id"]
    assert test_identity["leakage_group"] == train_identity["leakage_group"]
    manifest = pd.DataFrame([{**train_identity, "split": "train"}])
    kept, stats = reserve_metadata([sanitize_task_row(row, "task1")], "task1", manifest, selected)
    assert kept == []
    assert stats["exclusions_by_reason"]["train_or_validation_parent_group"] == 1


def test_temperature_mismatch_and_polymer_are_not_strictly_eligible():
    row, _, selected = example()
    mismatch = copy.deepcopy(selected)
    mismatch[("10.1/example", "CCO", "methanol")]["temperature_k"] = 300
    result = condition_metadata(row, "task1", mismatch)
    assert not result["condition_eligible"]
    assert not result["exact_same_temperature"]
    polymer = copy.deepcopy(selected)
    polymer[("10.1/example", "CCO", "methanol")]["solvent_smiles"] = "-"
    assert not condition_metadata(row, "task1", polymer)["condition_eligible"]


def test_original_temperature_selection_uses_availability_not_values():
    metadata = pd.DataFrame([
        {"source_doi": "10.1/x", "SMILES_Solute": "CCO", "Solvent": "water", "SMILES_Solvent": "O", "Temperature_K": 299,
         "molar_label_available": False, "raw_row": 0},
        {"source_doi": "10.1/x", "SMILES_Solute": "CCO", "Solvent": "water", "SMILES_Solvent": "O", "Temperature_K": 301,
         "molar_label_available": True, "raw_row": 1},
        {"source_doi": "10.1/x", "SMILES_Solute": "CCO", "Solvent": "water", "SMILES_Solvent": "O", "Temperature_K": 302,
         "molar_label_available": True, "raw_row": 2},
    ])
    assert selected_source_rows(metadata, "task1")[("10.1/x", "CCO", "water")]["temperature_k"] == 301
    assert selected_source_rows(metadata, "task2")[("10.1/x", "CCO", "water")]["temperature_k"] == 301


def test_already_leaking_manifest_is_rejected():
    row, manifest, selected = example()
    duplicate = manifest.iloc[0].to_dict()
    duplicate["solute_id"] = "different_salt_identity"
    duplicate["split"] = "train"
    manifest = pd.concat([manifest, pd.DataFrame([duplicate])], ignore_index=True)
    with pytest.raises(ValueError, match="already leaks"):
        reserve_metadata([sanitize_task_row(row, "task1")], "task1", manifest, selected)

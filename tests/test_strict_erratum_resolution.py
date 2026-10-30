"""Synthetic held-out-source gate fixtures, not chemical measurements."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('erratum_resolution',Path(__file__).parents[1]/'scripts/resolve_strict_source_erratum.py')
resolution=importlib.util.module_from_spec(spec);spec.loader.exec_module(resolution)


def metadata():
    return pd.DataFrame({'solute_id':['train','train','val'],'source_doi':['exclusive','shared','shared']})


def test_only_exclusively_training_sources_are_eligible():
    assert resolution.source_scope(metadata(),{'train':'train','val':'val'},{'exclusive'},{'exclusive'})=={'exclusive'}


def test_a_source_shared_with_heldout_solute_is_rejected():
    with pytest.raises(ValueError,match='held-out role'):
        resolution.source_scope(metadata(),{'train':'train','val':'val'},{'shared'},{'shared'})


def test_unknown_roles_and_unselected_quarantine_are_rejected():
    with pytest.raises(ValueError,match='Missing original strict role'):
        resolution.source_scope(metadata(),{'train':'train'},{'exclusive'},set())
    with pytest.raises(ValueError,match='held-out role'):
        resolution.source_scope(metadata(),{'train':'train','val':'val'},{'exclusive'},{'shared'})

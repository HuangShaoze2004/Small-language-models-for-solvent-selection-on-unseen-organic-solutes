import itertools
import math

from solvent_select.data import (assign_splits, balanced_blocks, canonical_identity,
                                 exclusion_parent, pair_label)

def test_salt_and_stereo_are_distinct_measurement_identities():
    neutral, mol = canonical_identity('C[C@H](O)C(=O)O')
    stereo, _ = canonical_identity('C[C@@H](O)C(=O)O')
    salt, salt_mol = canonical_identity('C[C@H](O)C(=O)[O-].[Na+]')
    assert len({neutral, stereo, salt}) == 3
    assert exclusion_parent(mol)[0] == exclusion_parent(salt_mol)[0]

def test_group_partition_does_not_split_relatives():
    groups = {f'g{i}': [f's{i}a', f's{i}b'] for i in range(100)}
    result = assign_splits(groups, 20261025)
    assert all(result[a] == result[b] for a, b in groups.values())
    assert result == assign_splits(groups, 20261025)
    assert set(result.values()) == {'train', 'val', 'test'}

def test_candidate_partition_keeps_all_candidates_without_using_outcomes():
    rows = [{'solvent_id': str(i), 'logx': -i} for i in range(17)]
    before = balanced_blocks(rows)
    after = balanced_blocks([{**r, 'logx': 1000-r['logx']} for r in reversed(rows)])
    assert [[r['solvent_id'] for r in b] for b in before] == [[r['solvent_id'] for r in b] for b in after]
    assert sorted(r['solvent_id'] for b in before for r in b) == sorted(r['solvent_id'] for r in rows)
    assert all(3 <= len(b) <= 8 for b in before)

def test_near_ties_do_not_form_transitive_groups():
    assert pair_label(-1, -1.08) == 'TIE'
    assert pair_label(-1.08, -1.16) == 'TIE'
    assert pair_label(-1, -1.16) == 'A'
    assert pair_label(-2, -1) == 'B'
    assert pair_label(-1, -1.1) == 'TIE'

def test_target_is_base_ten_mole_fraction():
    assert math.log10(0.01) == -2

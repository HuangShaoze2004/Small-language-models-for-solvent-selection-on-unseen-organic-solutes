"""Safety fixtures: no scientific test targets or actual model runs."""
import json

import pytest

from solvent_select.release_guard import verify_assets, verify_release
from solvent_select.views import sha256_file


def test_candidate_manifest_cannot_authorize_any_target_access(tmp_path):
    path=tmp_path/'candidate.json'
    path.write_text(json.dumps({'status':'CANDIDATE_NOT_FROZEN','target_path':'missing'}))
    with pytest.raises(ValueError,match='actual FROZEN'):
        verify_release(path)


def test_asset_verification_hashes_bytes_without_parsing_target_values(tmp_path):
    target=tmp_path/'opaque_target_bytes.jsonl'
    target.write_bytes(b'This intentionally is not JSON; contents must not be deserialized.\xff')
    verify_assets({str(target):sha256_file(target)})
    target.write_bytes(b'altered')
    with pytest.raises(ValueError,match='missing or modified'):
        verify_assets({str(target):'0'*64})


def test_empty_asset_group_does_not_pass_vacuously():
    with pytest.raises(ValueError,match='empty'):
        verify_assets({})

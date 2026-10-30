from pathlib import Path
import json

import pandas as pd
import pytest

from solvent_select.views import sha256_file

ROOT=Path(__file__).parents[1]
pytestmark=pytest.mark.skipif(not (ROOT/'data/processed/sourceqa/measurements.parquet').exists(),reason='Versioned real data not downloaded/processed')


def test_quarantine_is_training_only_and_all_retained_roles_inherited():
    audit=json.loads((ROOT/'artifacts/audit/sourceqa/data_audit.json').read_text())
    assert audit['source_quarantined_records']==919 and len(audit['source_dois'])==18
    removal=pd.read_csv(ROOT/'artifacts/audit/sourceqa/quarantined_records.csv')
    original=pd.read_csv(ROOT/'data/splits/identity.csv')
    assert set(removal.merge(original,on='solute_id').split)=={'train'}
    for protocol in ['identity','scaffold']:
        old=pd.read_csv(ROOT/f'data/splits/{protocol}.csv')
        new=pd.read_csv(ROOT/f'data/splits/sourceqa/{protocol}.csv')
        both=new.merge(old,on='solute_id',suffixes=('_new','_old'))
        assert (both.split_new==both.split_old).all()
        assert new.groupby('leakage_group').split.nunique().max()==1
        if protocol=='scaffold':assert new.groupby('scaffold_group').split.nunique().max()==1
        assert set(new[new.split=='test'].solute_id)==set(old[old.split=='test'].solute_id)


def test_global_test_and_primary_validation_are_byte_identical():
    for protocol,partitions in [('identity',['val','test']),('scaffold',['test'])]:
        for split in partitions:
            assert sha256_file(ROOT/f'data/processed/{protocol}/{split}_queries.jsonl')==sha256_file(ROOT/f'data/processed/sourceqa/{protocol}/{split}_queries.jsonl')
            assert sha256_file(ROOT/f'data/targets/{protocol}/{split}.jsonl')==sha256_file(ROOT/f'data/targets/sourceqa/{protocol}/{split}.jsonl')
            if split=='test':
                assert sha256_file(ROOT/f'data/views/{protocol}/test_queries.jsonl')==sha256_file(ROOT/f'data/views/sourceqa/{protocol}/test_queries.jsonl')


def test_amended_bank_universes_match_and_quarantined_sources_are_absent():
    audit=json.loads((ROOT/'artifacts/audit/sourceqa/data_audit.json').read_text())
    for protocol,name in [('identity','measurements.parquet'),('scaffold','measurements_scaffold.parquet')]:
        data=pd.read_parquet(ROOT/f'data/processed/sourceqa/{name}',columns=['solute_id','source_doi'])
        split=pd.read_csv(ROOT/f'data/splits/sourceqa/{protocol}.csv')
        assert set(data.solute_id)==set(split.solute_id)
        assert not set(data.source_doi)&set(audit['source_dois'])

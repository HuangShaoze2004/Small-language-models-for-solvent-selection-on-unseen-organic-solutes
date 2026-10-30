"""Integration checks on actual acquired data; no model test scores are computed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT/'data/processed/measurements.parquet').exists(), reason='Acquire/curate real data first')

@pytest.fixture(scope='module')
def measurements():
    return pd.read_parquet(ROOT/'data/processed/measurements.parquet')

def test_real_measurements_and_identity_invariants(measurements):
    d=measurements
    assert d.record_id.is_unique
    assert d[['solute_id','solvent_id','temperature_k','source_doi','logx']].notna().all().all()
    assert np.isfinite(d.logx).all() and (d.logx<=0).all()
    assert np.allclose(d.logx,np.log10(d.mole_fraction),rtol=0,atol=1e-12)
    assert d.groupby('solute_id').solute_smiles.nunique().max()==1
    assert d.groupby('solute_smiles').solute_id.nunique().max()==1
    assert (d.solvent_smiles!='-').all()
    assert d[d.is_multifragment].solute_smiles.str.contains('.',regex=False).all()

@pytest.mark.parametrize('protocol',['identity','scaffold'])
def test_real_group_boundaries(measurements,protocol):
    manifest=pd.read_csv(ROOT/f'data/splits/{protocol}.csv')
    assert manifest.solute_id.is_unique
    if protocol == 'identity':
        assert set(manifest.solute_id)==set(measurements.solute_id)
    else:
        strict_measurements=pd.read_parquet(ROOT/'data/processed/measurements_scaffold.parquet',columns=['solute_id'])
        assert set(manifest.solute_id)==set(strict_measurements.solute_id)
        main=pd.read_csv(ROOT/'data/splits/identity.csv')
        assert set(main.loc[main.split=='test','solute_id'])==set(manifest.loc[manifest.split=='test','solute_id'])
        excluded=pd.read_csv(ROOT/'data/splits/scaffold_excluded.csv')
        assert set(manifest.solute_id).isdisjoint(excluded.solute_id)
        assert set(manifest.solute_id)|set(excluded.solute_id)==set(measurements.solute_id)
    assert set(manifest.split)=={'train','val','test'}
    assert manifest.groupby('leakage_group').split.nunique().max()==1
    if protocol=='scaffold': assert manifest.groupby('scaffold_group').split.nunique().max()==1

def test_every_derived_query_uses_real_same_source_same_temperature(measurements):
    rows=measurements.set_index('record_id')[['solute_id','source_doi','temperature_k','solvent_smiles']].to_dict('index')
    manifest=pd.read_csv(ROOT/'data/splits/identity.csv').set_index('solute_id').split.to_dict()
    seen=set()
    for split in ['train','val','test']:
        with open(ROOT/f'data/processed/identity/{split}_queries.jsonl') as stream:
            for line in stream:
                q=json.loads(line)
                assert q['query_id'] not in seen
                seen.add(q['query_id'])
                assert manifest[q['solute_id']]==split==q['split']
                candidates=[rows[c['record_id']] for c in q['candidates']]
                assert {r['solute_id'] for r in candidates}=={q['solute_id']}
                assert {r['temperature_k'] for r in candidates}=={q['temperature_k']}
                assert len({r['source_doi'] for r in candidates})==1
                assert len({r['solvent_smiles'] for r in candidates})==len(candidates)
                assert not {'choice','ranking','values','answer','logx','logS'} & set(q)

def test_nearest_saved_provenance_only_uses_train_records(measurements):
    path=ROOT/'artifacts/runs/identity_nearest/predictions_val.parquet'
    if not path.exists(): pytest.skip('Nearest run not yet available')
    manifest=pd.read_csv(ROOT/'data/splits/identity.csv')
    train_ids=set(manifest.loc[manifest.split=='train','solute_id'])
    train_record_ids=set(measurements.loc[measurements.solute_id.isin(train_ids),'record_id'])
    pred=pd.read_parquet(path)
    assert set(pred.nearest_record_id)<=train_record_ids

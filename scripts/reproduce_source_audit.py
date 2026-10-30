#!/usr/bin/env python3
"""Recompute training-only archive comparisons and verify frozen numerical evidence."""
from pathlib import Path
import json
import tempfile

from solvent_select.source_audit import compare_training
from solvent_select.views import sha256_file,write_json


def main():
    stage=Path(tempfile.mkdtemp(prefix='source_audit_rebuild_',dir=Path('.tmp')))/'audit'
    compare_training('data/processed/measurements.parquet','data/splits/identity.csv',
                     'artifacts/audit/source_discrepancy/nist_download_manifest.json',stage)
    policy=json.loads(Path('configs/source_quarantine_v1_reproduce.json').read_text())
    results=[]
    for source,expected in policy['evidence'].items():
        path=stage/Path(source).name
        actual=sha256_file(path)
        results.append({'file':path.name,'expected_sha256':expected,'actual_sha256':actual,'identical':actual==expected})
    assert all(row['identical'] for row in results)
    write_json('artifacts/audit/sourceqa/source_comparison_reconstruction.json',{
        'status':'PASSED_ACTUAL_NUMERICAL_RECONSTRUCTION','files':results,'staging_root':str(stage),
        'metadata_paths_timestamps_may_differ':True,'test_target_values_read':False,
        'code_sha256':sha256_file(__file__)})
    print('All',len(results),'frozen numerical source-audit files reproduced byte-for-byte')


if __name__=='__main__':main()

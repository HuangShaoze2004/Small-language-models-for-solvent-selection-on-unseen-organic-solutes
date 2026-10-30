#!/usr/bin/env python3
"""Actually reconstruct sourceqa scientific artifacts in a fresh project-local tree."""
from pathlib import Path
import argparse
import json
import shutil
import tempfile
import time

from solvent_select.source_quarantine import build
from solvent_select.views import sha256_file,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize",action="store_true")
    args=parser.parse_args()
    root=Path.cwd()
    stage=Path(tempfile.mkdtemp(prefix='sourceqa_rebuild_',dir=root/'.tmp'))
    original=[root/'data/processed/measurements.parquet',root/'data/processed/measurements_scaffold.parquet',root/'artifacts/audit/data_audit.json']
    original += [root/'data/splits'/name for name in ['identity.csv','scaffold.csv','scaffold_excluded.csv']]
    for protocol in ['identity','scaffold']:
        for partition in ['train','val','test']:
            original += [root/f'data/processed/{protocol}/{partition}_queries.jsonl',root/f'data/targets/{protocol}/{partition}.jsonl']
    for path in original:
        destination=stage/path.relative_to(root)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    before={str(p):sha256_file(p) for p in original}
    started=time.monotonic()
    build(stage,'configs/source_quarantine_v1_reproduce.json')
    expected=json.loads((root/'artifacts/audit/sourceqa/data_manifest.json').read_text())
    comparisons=[]
    for name,digest in expected.items():
        actual=sha256_file(stage/name)
        comparisons.append({'artifact':name,'expected_sha256':digest,'reconstructed_sha256':actual,'identical':actual==digest})
    assert all(row['identical'] for row in comparisons)
    if args.materialize:
        for name,digest in expected.items():
            destination=root/name
            if destination.exists():
                assert sha256_file(destination)==digest, 'Existing artifact differs: '+name
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(stage/name,destination)
    assert before=={str(p):sha256_file(p) for p in original}
    result={'status':'PASSED_ACTUAL_RECONSTRUCTION','artifacts':len(comparisons),'all_scientific_bytes_identical':True,
            'elapsed_seconds':time.monotonic()-started,'staging_root':str(stage),'comparisons':comparisons,
            'original_inputs_unchanged':True,'materialize_missing_outputs':args.materialize,'reproduction_policy':'configs/source_quarantine_v1_reproduce.json',
            'policy_sha256':sha256_file('configs/source_quarantine_v1_reproduce.json'),'code_sha256':sha256_file(__file__),
            'scope':'Original raw-to-curated18artifact reconstruction was verified separately. This actually verifies the additional source-quarantine stage, without exposing final target values or evaluating any model.'}
    write_json('artifacts/audit/sourceqa/reconstruction.json',result)
    print(json.dumps({k:result[k] for k in ['status','artifacts','elapsed_seconds','all_scientific_bytes_identical']}))


if __name__=='__main__':
    main()

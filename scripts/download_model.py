#!/usr/bin/env python3
"""Fetch the immutable public Qwen snapshot and verify every recorded file SHA256."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json

from huggingface_hub import snapshot_download

from solvent_select.llm import MODEL_ID,MODEL_REVISION
from solvent_select.views import sha256_file,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-only',action='store_true')
    args=parser.parse_args()
    expected_path=Path('artifacts/audit/model_snapshot.json')
    expected=json.loads(expected_path.read_text())
    if expected['model']!=MODEL_ID or expected['revision']!=MODEL_REVISION:
        raise ValueError('Expected model manifest and implementation revision differ')
    directory=Path(snapshot_download(repo_id=MODEL_ID,revision=MODEL_REVISION,
                   cache_dir=str(Path('.cache/huggingface/hub').resolve()),
                   allow_patterns=[r['name'] for r in expected['files']],local_files_only=args.local_only))
    checked=[]
    for row in expected['files']:
        path=directory/row['name']
        if path.stat().st_size!=row['bytes'] or sha256_file(path)!=row['sha256']:
            raise ValueError('Immutable model snapshot file failed verification: '+row['name'])
        checked.append({'name':row['name'],'bytes':row['bytes'],'sha256':row['sha256']})
    result={'status':'PASSED_ACTUAL_FILE_VERIFICATION','model':MODEL_ID,'revision':MODEL_REVISION,
            'path':str(directory),'verified_utc':datetime.now(timezone.utc).isoformat(),
            'expected_manifest_sha256':sha256_file(expected_path),'files':checked,
            'license_file_sha256':next(r['sha256'] for r in checked if r['name']=='LICENSE'),
            'command_local_only':args.local_only,'code_sha256':sha256_file(__file__)}
    write_json('artifacts/audit/model_snapshot_recheck.json',result)
    print(json.dumps({'status':result['status'],'files':len(checked),'bytes':sum(r['bytes'] for r in checked),'revision':MODEL_REVISION}))


if __name__=='__main__':
    main()

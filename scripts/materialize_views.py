#!/usr/bin/env python3
"""Restore missing question/target views from immutable saved selection IDs."""
from pathlib import Path
import argparse
import json

from solvent_select.views import filter_jsonl,sha256_file,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/views/sourceqa')
    parser.add_argument('--destination-root')
    parser.add_argument('--audit-output')
    args=parser.parse_args()
    restored=0
    checked=[]
    for protocol in ['identity','scaffold']:
        manifest=Path(args.root)/protocol/'view_manifest.json'
        data=json.loads(manifest.read_text())
        for split,info in data['partitions'].items():
            selection=Path(info['selection_path'])
            assert sha256_file(selection)==info['selection_sha256']
            ids=json.loads(selection.read_text())['query_ids']
            for kind in ['query','target']:
                source=(Path('data/processed/sourceqa')/protocol/(split+'_queries.jsonl')) if kind=='query' else Path(info['source_target_path'])
                expected=info['source_query_sha256'] if kind=='query' else info['source_target_sha256']
                assert sha256_file(source)==expected,'Changed view source: '+str(source)
                artifact=info[kind+'_view'];destination=Path(artifact['path'])
                if args.destination_root:
                    destination=Path(args.destination_root)/destination.resolve().relative_to(Path(args.root).resolve())
                    destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():
                    filter_jsonl(source,destination,ids);restored+=1
                assert sha256_file(destination)==artifact['sha256'],'Restored view differs'
                checked.append({'path':str(destination),'sha256':artifact['sha256']})
    if args.audit_output:
        write_json(args.audit_output,{'status':'PASSED_ACTUAL_VIEW_MATERIALIZATION','restored_files':restored,'checked_files':checked,'code_sha256':sha256_file(__file__),'selection_changed':False,'test_outcomes_evaluated':False})
    print('All registered view files verified; missing files restored:',restored)


if __name__=='__main__':main()

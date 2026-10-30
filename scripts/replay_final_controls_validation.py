#!/usr/bin/env python3
"""Replay final-control computation on real validation queries, never test data.

No fake release is created and no final test guard is bypassed. The computation
helpers are compared against preserved validation outputs from earlier scripts.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from solvent_select.consistency import normalized,read,score,write_rows
from solvent_select.final_controls import evidence_predictor,molecular_rows,property_predictor
from solvent_select.evaluate import property_predictions
from solvent_select.llm import validate_query_manifest,write_json
from solvent_select.retrieval import canonical_smiles
from solvent_select.views import sha256_file
from solvent_select.study import DEFAULT_REGISTRY,load_registry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['classical','chemprop','evidence'],required=True)
    parser.add_argument('--protocol',choices=['identity','scaffold','all'],default='all')
    parser.add_argument('--registry',default=DEFAULT_REGISTRY)
    parser.add_argument('--output-root',default='artifacts/audit/dualqa/final_control_validation_replay')
    args=parser.parse_args()
    registry=load_registry(args.registry)
    output=Path(args.output_root)/args.kind
    if output.exists():raise FileExistsError('Preserve previous replay evidence')
    output.mkdir(parents=True)
    results=[]
    for protocol,config in registry['protocols'].items():
        if args.protocol not in ['all',protocol]:continue
        manifest=config['split']
        folder=Path(config['consistency_data_root'])/f'{protocol}_val'
        cohorts=[read(folder/f'{name}_queries.jsonl') for name in ['base','transformed']]
        for queries in cohorts:
            validate_query_manifest(queries,manifest)
            if {q['split'] for q in queries}!={'val'}:raise ValueError('Only validation queries are authorized')
        prefix=config['run_prefix']+protocol+'_'
        names=([prefix+'nearest']+[prefix+f'lgbm_{s}' for s in [17,42,2026]] if args.kind=='classical'
               else [prefix+f'chemprop_{s}' for s in [17,42,2026]] if args.kind=='chemprop'
               else [prefix+'evidence_text_val'])
        for name in names:
            for canonical in ([False,True] if args.kind=='chemprop' else [args.kind=='evidence']):
                suffix='_canonical_control' if canonical else ''
                destination=output/(name+suffix);destination.mkdir()
                reference=Path(config['consistency_runs_root'])/protocol/(name+suffix)
                model=Path(config['runs_root'])/name
                control={'kind':'evidence' if args.kind=='evidence' else 'property',
                         'model_run':str(model),'canonicalize_query':canonical,
                         'config':'configs/llm_formal_sft.yaml','retrieval_mode':'text',
                         'measurements':config['measurements']}
                if args.kind=='evidence':
                    predictor=evidence_predictor(control,manifest)
                    base_reference=model/'predictions.jsonl'
                    transformed_reference=reference/'predictions.jsonl'
                else:
                    predictor,metadata=property_predictor(control)
                    assert metadata['split_sha256']==sha256_file(manifest)
                    assert metadata['measurement_sha256']==sha256_file(metadata['measurements'])
                    base_reference=reference/'base_predictions.jsonl'
                    transformed_reference=reference/'transformed_predictions.jsonl'
                started=time.monotonic();max_delta=0.;changed_base=0
                for label,queries,reference_path in zip(['base','transformed'],cohorts,[base_reference,transformed_reference]):
                    if args.kind=='evidence':
                        rows,changed=predictor(queries)
                        if label=='base':changed_base=changed
                    else:
                        converted,frame=molecular_rows(queries)
                        if canonical:
                            for column in ['solute_smiles','solvent_smiles']:
                                normalized_values=frame[column].map(canonical_smiles)
                                if label=='base':changed_base+=int((normalized_values!=frame[column]).sum())
                                frame[column]=normalized_values
                        values=predictor(frame)
                        rows=property_predictions(converted,pd.DataFrame({'record_id':frame.record_id,'pred_logx':values}))
                    write_rows(destination/f'{label}_predictions.jsonl',rows)
                    saved={r['query_id']:r for r in read(reference_path)}
                    for query,row in zip(queries,rows):
                        previous=saved[query['query_id']]
                        assert normalized(query,row)==normalized(query,previous)
                        if 'predicted_values' in row:
                            a=np.asarray(list(row['predicted_values'].values()))
                            b=np.asarray([previous['predicted_values'][k] for k in row['predicted_values']])
                            assert np.allclose(a,b,rtol=1e-7,atol=1e-7)
                            max_delta=max(max_delta,float(np.max(np.abs(a-b))))
                score(folder,destination/'base_predictions.jsonl',destination/'transformed_predictions.jsonl',destination/'evaluation')
                result={'name':name+suffix,'status':'PASSED_REAL_VALIDATION_REPLAY','max_property_prediction_difference_dex':max_delta,
                        'base_structure_fields_changed':changed_base,'elapsed_seconds':time.monotonic()-started,
                        'reference_sha256':{str(p):sha256_file(p) for p in [base_reference,transformed_reference]},
                        'cohort_manifest_sha256':sha256_file(folder/'manifest.json')}
                write_json(destination/'replay.json',result);results.append(result)
                print(name,suffix,max_delta,flush=True)
    write_json(output/'summary.json',{'status':'PASSED_REAL_VALIDATION_REPLAY','kind':args.kind,'runs':results,
               'test_target_values_read':False,'test_queries_read':False,'frozen_release_created':False,
               'registry':args.registry,'registry_sha256':sha256_file(args.registry),'protocol':args.protocol,
               'implementation_sha256':sha256_file('src/solvent_select/final_controls.py'),'script_sha256':sha256_file(__file__)})


if __name__=='__main__':main()

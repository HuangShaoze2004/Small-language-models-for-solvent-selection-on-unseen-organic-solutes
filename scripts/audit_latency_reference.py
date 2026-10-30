#!/usr/bin/env python3
"""Retain every CPU timing/reference decision difference without target values."""
import argparse
import json
from pathlib import Path

import pandas as pd

from solvent_select.evaluate import property_predictions
from solvent_select.llm import read_jsonl, validate_query_manifest
from solvent_select.study import load_registry, property_names
from solvent_select.views import sha256_file, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latency-root',default='artifacts/latency/property_v1')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve previous numerical-difference audits')
    cases,counts,pending,sources=[],[],[],{__file__:sha256_file(__file__)}
    for protocol,config in load_registry()['protocols'].items():
        path=Path(config['views'])/'val_queries.jsonl'
        queries=read_jsonl(path)
        validate_query_manifest(queries,config['split'])
        if {q['split'] for q in queries}!={'val'}:
            raise ValueError('Validation-only reference audit')
        by_id={q['query_id']:q for q in queries}
        sources[str(path)]=sha256_file(path)
        for name in property_names(protocol,config):
            folder=Path(args.latency_root)/name
            if not (folder/'run.json').exists():
                pending.append(name);continue
            record=json.loads((folder/'run.json').read_text())
            if record['status']!='COMPLETED_VALIDATION_CPU_TIMING':
                raise ValueError('Incomplete timing cannot supply a reference comparison')
            for source,expected in record['inputs'].items():
                if sha256_file(source)!=expected:raise ValueError('Timing input changed: '+source)
            for filename,expected in record['outputs'].items():
                if sha256_file(folder/filename)!=expected:raise ValueError('Timing output changed: '+filename)
            reference_path=Path(config['runs_root'])/name/'predictions_val.parquet'
            reference={r['query_id']:r for r in property_predictions(queries,pd.read_parquet(reference_path))}
            sources[str(reference_path)]=sha256_file(reference_path)
            sources[str(folder/'run.json')]=sha256_file(folder/'run.json')
            for timing in record['timings']:
                predictions_path=folder/f"decisions_{timing['repetition']}.jsonl"
                current=read_jsonl(predictions_path)
                if len(current)!=len(reference) or {r['query_id'] for r in current}!=set(reference):
                    raise ValueError('CPU and reference decision cohorts differ')
                sources[str(predictions_path)]=sha256_file(predictions_path)
                changed,top1_changed,pair_changed=0,0,0
                for row in current:
                    before=reference[row['query_id']]
                    if (before.get('choice'),before.get('ranking'))==(row.get('choice'),row.get('ranking')):
                        continue
                    changed+=1
                    query=by_id[row['query_id']]
                    best_changed=(query['task']=='rank' and before['ranking'][0]!=row['ranking'][0])
                    top1_changed+=int(best_changed)
                    pair_changed+=int(query['task']=='pair')
                    cases.append({'protocol':protocol,'run':name,'repetition':timing['repetition'],
                        'query_id':row['query_id'],'task':query['task'],'rank_first_candidate_changed':best_changed,
                        'reference':before,'cpu':row,
                        'scope':'CPU/batch-order recomputation only; original scientific output is retained.'})
                if changed!=timing['changed_decisions_from_saved_validation']:
                    raise ValueError('Independent difference recount disagrees with timing metadata')
                counts.append({'run':name,'repetition':timing['repetition'],'queries':len(current),
                    'changed_full_decisions':changed,'changed_pair_choices':pair_changed,
                    'changed_rank_first_candidate':top1_changed,
                    'max_abs_predicted_value_difference':timing['max_abs_value_difference_from_saved_validation']})
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,{'status':'COMPLETED_VALIDATION_CPU_REFERENCE_AUDIT','counts':counts,'all_differences':cases,
        'pending_models':pending,'evidence_sha256':sources,'target_values_read':False,'test_queries_read':False,
        'scientific_outputs_replaced':False,'prediction_thresholds_changed':False,
        'interpretation':'All repeated timing differences are retained. CPU device, batch ordering and numerical computation differ from saved original validation inference; this audit does not identify a unique cause or redefine primary scientific predictions.'})
    print(json.dumps({'model_repetitions':len(counts),'different_decisions_across_repetitions':len(cases),
                      'unique_changed_queries':len({(r['run'],r['query_id']) for r in cases}),
                      'changed_pair_choices':sum(r['changed_pair_choices'] for r in counts),
                      'changed_rank_first_candidates':sum(r['changed_rank_first_candidate'] for r in counts)}))


if __name__=='__main__':main()

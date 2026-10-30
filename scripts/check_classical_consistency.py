#!/usr/bin/env python3
"""Recompute classical predictions from transformed structures, without targets."""
from pathlib import Path
import argparse
import copy
import json
import time

import joblib
import pandas as pd

from solvent_select.consistency import read, score, write_rows
from solvent_select.evaluate import property_predictions
from solvent_select.llm import validate_query_manifest
from solvent_select.views import sha256_file, write_json


def molecular_rows(queries):
    converted, rows = [], []
    for original in queries:
        query = copy.deepcopy(original)
        for candidate in query['candidates']:
            candidate['record_id'] = query['query_id'] + ':' + candidate['id']
            rows.append({'record_id': candidate['record_id'], 'solute_id': query['solute_id'],
                         'solute_smiles': query['solute_smiles'], 'solvent_id': candidate['id'],
                         'solvent_smiles': candidate['smiles'], 'temperature_k': query['temperature_k']})
        converted.append(query)
    return converted, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', nargs='+', default=['identity_nearest', 'identity_lgbm_17', 'identity_lgbm_42', 'identity_lgbm_2026'])
    parser.add_argument('--output-root', default='artifacts/consistency/classical_val')
    parser.add_argument('--runs-root', default='artifacts/runs')
    parser.add_argument('--consistency-root', default='data/consistency/identity_val')
    parser.add_argument('--manifest', default='data/splits/identity.csv')
    args = parser.parse_args()
    folder = Path(args.consistency_root)
    queries = [read(folder / name) for name in ['base_queries.jsonl', 'transformed_queries.jsonl']]
    for cohort in queries:
        validate_query_manifest(cohort, args.manifest)
        if {q['split'] for q in cohort} != {'val'}:
            raise ValueError('This development diagnostic only processes validation queries')
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.run_id:
        destination = output / name
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir()
        model_path = Path(args.runs_root) / name / 'model.joblib'
        bundle = joblib.load(model_path)
        started = time.monotonic()
        for label, cohort in zip(['base', 'transformed'], queries):
            converted, frame = molecular_rows(cohort)
            if bundle['method'] == 'nearest':
                values, _ = bundle['model'].predict(frame)
            else:
                values = bundle['model'].predict(bundle['features'].transform(frame), num_threads=4)
            predictions = property_predictions(converted, pd.DataFrame({'record_id': frame.record_id, 'pred_logx': values}))
            write_rows(destination / f'{label}_predictions.jsonl', predictions)
        result = score(folder, destination / 'base_predictions.jsonl', destination / 'transformed_predictions.jsonl', destination / 'evaluation')
        write_json(destination / 'run.json', {'status': 'completed', 'elapsed_seconds': time.monotonic() - started,
                                             'model': str(model_path), 'model_sha256': sha256_file(model_path),
                                             'test_labels_read': False, 'code_sha256': sha256_file(__file__)})
        results.append({'run_id': name, 'result': result})
        print(name, {k: r['exact_agreement'] for k, r in result['by_variant_task'].items()}, flush=True)
    write_json(output / 'summary.json', {'status': 'VALIDATION_ONLY', 'runs': results})


if __name__ == '__main__':
    main()

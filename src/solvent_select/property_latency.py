"""CPU timing of the fixed property-ranking flow on registered validation queries.

No training, target values, test queries, or changes to scientific evaluations.
Different CPU/GPU numerical decisions are recorded instead of being substituted.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .evaluate import property_predictions, parsed_prediction
from .llm import read_jsonl, validate_query_manifest
from .study import load_registry, property_names
from .views import sha256_file, write_json


def candidate_requests(queries):
    """Deduplicate exact public feature tuples, preserving identity and temperature."""
    converted, lookup, rows = [], {}, []
    for original in queries:
        query = copy.deepcopy(original)
        for candidate in query['candidates']:
            key = (query['solute_smiles'], candidate['smiles'], float(query['temperature_k']))
            if key not in lookup:
                record_id = f'candidate_{len(rows):08d}'
                lookup[key] = record_id
                rows.append({'record_id': record_id, 'solute_id': query['solute_id'],
                             'solute_smiles': key[0], 'solvent_id': candidate['id'],
                             'solvent_smiles': key[1], 'temperature_k': key[2]})
            candidate['record_id'] = lookup[key]
        converted.append(query)
    return converted, pd.DataFrame(rows)


def decisions_equal(first, second):
    a = {row['query_id']: parsed_prediction(row) for row in first}
    b = {row['query_id']: parsed_prediction(row) for row in second}
    if set(a) != set(b):
        raise ValueError('Decision cohorts differ')
    return sum((a[q].get('choice'), a[q].get('ranking')) != (b[q].get('choice'), b[q].get('ranking')) for q in a)


def predictor(run, metadata, threads):
    if metadata['method'] != 'chemprop':
        import joblib
        bundle = joblib.load(run / 'model.joblib')
        def predict(frame):
            if bundle['method'] == 'nearest':
                start = time.perf_counter()
                values = bundle['model'].predict(frame)[0]
                return values, None, time.perf_counter() - start
            # Clear deterministic caches on this in-memory copy, retain all
            # fitted imputation/model parameters and the saved model file.
            features = bundle['features']
            features._fingerprint_cache.clear()
            features._descriptor_cache.clear()
            start = time.perf_counter()
            encoded = features.transform(frame)
            feature_seconds = time.perf_counter() - start
            start = time.perf_counter()
            values = bundle['model'].predict(encoded, num_threads=threads)
            return values, feature_seconds, time.perf_counter() - start
        return predict, {'model': str(run / 'model.joblib'), 'feature_scope':
                        'Nearest combines feature/index construction and forward in forward_seconds; '
                        'LightGBM records query feature construction separately.'}
    import dataclasses
    import lightning.pytorch as pl
    import torch
    from .chemprop_runner import ChempropConfig, make_dataset, make_loader, make_model
    torch.set_num_threads(threads)
    config = dataclasses.replace(ChempropConfig(**metadata['config']), torch_threads=threads, num_workers=0)
    scalers = json.loads((run / 'scalers.json').read_text())
    model = make_model(config)
    checkpoint = torch.load(metadata['best_checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    trainer = pl.Trainer(accelerator='cpu', devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False)
    def predict(frame):
        start = time.perf_counter()
        dataset = make_dataset(frame, scalers, labels=False)
        loader = make_loader(dataset, config, training=False, seed=metadata['seed'])
        feature_seconds = time.perf_counter() - start
        start = time.perf_counter()
        values = torch.cat(trainer.predict(model, dataloaders=loader)).cpu().numpy().reshape(-1)
        values = values * scalers['target_std'] + scalers['target_mean']
        return values, feature_seconds, time.perf_counter() - start
    return predict, {'checkpoint': metadata['best_checkpoint'], 'batch_size': config.batch_size,
                     'feature_scope': 'Graph/loader construction separately; CPU Lightning prediction includes batching and device handling.'}


def benchmark(protocol, config, name, destination, spec, spec_path, registry_path):
    run = Path(config['runs_root']) / name
    metadata = json.loads((run / 'metadata.json').read_text())
    if metadata['status'] != 'completed':
        raise ValueError('Only completed models may be benchmarked')
    inputs = {str(spec_path): sha256_file(spec_path), str(registry_path): sha256_file(registry_path),
              str(Path(__file__)): sha256_file(__file__), str(run / 'metadata.json'): sha256_file(run / 'metadata.json')}
    for key, hash_key, expected_path in [('measurements', 'measurement_sha256', config['measurements']),
                                         ('splits', 'split_sha256', config['split'])]:
        if Path(metadata[key]).resolve() != Path(expected_path).resolve() or sha256_file(expected_path) != metadata[hash_key]:
            raise ValueError('Model data namespace or hash changed')
        inputs[str(expected_path)] = metadata[hash_key]
    queries_path = Path(config['views']) / 'val_queries.jsonl'
    queries = read_jsonl(queries_path)
    validate_query_manifest(queries, config['split'])
    if {query['split'] for query in queries} != {'val'}:
        raise ValueError('Benchmarking is restricted to registered validation queries')
    inputs[str(queries_path)] = sha256_file(queries_path)
    source = run / 'predictions_val.parquet'
    reference = pd.read_parquet(source)
    reference_decisions = property_predictions(queries, reference)
    if not np.isfinite(reference.pred_logx.to_numpy()).all():
        raise ValueError('Nonfinite reference property predictions')
    inputs[str(source)] = sha256_file(source)
    start = time.perf_counter()
    converted, frame = candidate_requests(queries)
    assembly_seconds = time.perf_counter() - start
    # Repeated occurrences can have different measured-source record IDs; the
    # property interface has only structure and temperature and ignores DOI.
    expected_values = reference.set_index('record_id').pred_logx.to_dict()
    candidate_occurrences = [(new['record_id'], expected_values[old['record_id']])
                             for q, c in zip(queries, converted) for old, new in zip(q['candidates'], c['candidates'])]
    start = time.perf_counter()
    predict, implementation = predictor(run, metadata, spec['cpu_threads'])
    loading_seconds = time.perf_counter() - start
    assets = [run / 'model.joblib'] if metadata['method'] != 'chemprop' else [Path(metadata['best_checkpoint']), run / 'scalers.json']
    for path in assets:
        inputs[str(path)] = sha256_file(path)
    destination.mkdir(parents=True)
    frame.to_parquet(destination / 'candidate_inputs.parquet', index=False)
    with (destination / 'query_mapping.jsonl').open('w') as stream:
        for row in converted:
            stream.write(json.dumps(row) + '\n')
    timings = []
    for repetition in range(spec['timing_repetitions']):
        start = time.perf_counter()
        values, feature_seconds, forward_seconds = predict(frame)
        if len(values) != len(frame) or not np.isfinite(values).all():
            raise ValueError('Invalid CPU property predictions')
        predictions = pd.DataFrame({'record_id': frame.record_id, 'pred_logx': values})
        decision_start = time.perf_counter()
        decisions = property_predictions(converted, predictions)
        decision_seconds = time.perf_counter() - decision_start
        application_seconds = time.perf_counter() - start + assembly_seconds
        lookup = predictions.set_index('record_id').pred_logx.to_dict()
        deltas = np.asarray([abs(lookup[record_id] - value) for record_id, value in candidate_occurrences])
        timings.append({'repetition': repetition, 'request_assembly_seconds': assembly_seconds,
                        'feature_seconds': feature_seconds, 'forward_seconds': forward_seconds,
                        'decision_seconds': decision_seconds, 'model_resident_application_seconds': application_seconds,
                        'amortized_seconds_per_public_question': application_seconds / len(queries),
                        'max_abs_value_difference_from_saved_validation': float(deltas.max()),
                        'mean_abs_value_difference_from_saved_validation': float(deltas.mean()),
                        'changed_decisions_from_saved_validation': decisions_equal(decisions, reference_decisions)})
        predictions.to_parquet(destination / f'candidate_predictions_{repetition}.parquet', index=False)
        with (destination / f'decisions_{repetition}.jsonl').open('w') as stream:
            for decision in decisions:
                stream.write(json.dumps(decision) + '\n')
    result = {'status': 'COMPLETED_VALIDATION_CPU_TIMING', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'run': str(run), 'protocol': protocol, 'method': metadata['method'], 'seed': metadata['seed'],
              'accelerator': 'cpu', 'cpu_threads': spec['cpu_threads'], 'public_questions': len(queries),
              'candidate_appearances': len(candidate_occurrences), 'unique_candidate_calls': len(frame),
              'model_loading_seconds': loading_seconds, 'timings': timings, 'implementation': implementation,
              'inputs': inputs, 'test_queries_read': False, 'target_values_read': False,
              'original_scientific_predictions_modified': False,
              'environment': {'python': platform.python_version(), 'loadavg_at_completion': list(os.getloadavg()),
                              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                              'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'rdkit']},
                              'host_scope': 'Shared host; main GPU validation may run concurrently. No isolated latency or energy-cost claim.'},
              'outputs': {p.name: sha256_file(p) for p in destination.iterdir() if p.is_file()}}
    write_json(destination / 'run.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['classical', 'chemprop'], required=True)
    parser.add_argument('--protocol', choices=['identity', 'scaffold', 'all'], default='all')
    parser.add_argument('--config', default='configs/latency_protocol_v1.json')
    parser.add_argument('--registry', default='configs/study_registry_dualqa_v2.json')
    parser.add_argument('--output-root', default='artifacts/latency/property_v1')
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    if (spec.get('status') != 'REGISTERED_VALIDATION_TIMING_BEFORE_TEST' or spec['partition'] != 'val'
            or spec['cpu_threads'] != 4 or spec['timing_repetitions'] != 3 or spec['accelerator'] != 'cpu'):
        raise ValueError('Unknown timing specification')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Launch this CPU-only benchmark with CUDA_VISIBLE_DEVICES empty')
    registry = load_registry(args.registry)
    pending, results = [], []
    for protocol, config in registry['protocols'].items():
        if args.protocol not in ['all', protocol]:
            continue
        for name in property_names(protocol, config):
            if ('_chemprop_' in name) != (args.kind == 'chemprop'):
                continue
            source = Path(config['runs_root']) / name / 'metadata.json'
            if not source.exists() or json.loads(source.read_text()).get('status') != 'completed':
                pending.append(name)
                continue
            destination = Path(args.output_root) / name
            if destination.exists():
                saved = json.loads((destination / 'run.json').read_text())
                if saved['status'] != 'COMPLETED_VALIDATION_CPU_TIMING':
                    raise ValueError('Incomplete previous timing; inspect before recovery')
                for path, expected in saved['inputs'].items():
                    if sha256_file(path) != expected:
                        raise ValueError('Completed benchmark input changed: ' + path)
                for path, expected in saved['outputs'].items():
                    if sha256_file(destination / path) != expected:
                        raise ValueError('Completed timing output changed: ' + path)
                result = saved
            else:
                result = benchmark(protocol, config, name, destination, spec, args.config, args.registry)
            results.append(name)
            print(json.dumps({'run': name, 'mean_amortized_cpu_seconds': np.mean([
                row['amortized_seconds_per_public_question'] for row in result['timings']]),
                'max_changed_decisions': max(row['changed_decisions_from_saved_validation'] for row in result['timings'])}), flush=True)
    print(json.dumps({'completed_or_verified': results, 'pending': pending, 'test_queries_read': False}))


if __name__ == '__main__':
    main()

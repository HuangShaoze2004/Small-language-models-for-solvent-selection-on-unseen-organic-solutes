#!/usr/bin/env python3
"""Summarize actual final-test evidence support and explicitly scoped timings.

No target files are opened and no partially completed inference is summarized.
CPU timing repetitions are not scientific learning-seed repetitions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from solvent_select.llm import read_jsonl, validate_query_manifest
from solvent_select.retrieval import MODES
from solvent_select.report import _run_identity
from solvent_select.study import load_registry, evidence_name, property_names
from solvent_select.validation_assessment import registered_runs
from solvent_select.views import sha256_file, write_json


def support_row(query, prediction):
    candidates = {str(c['id']) for c in query['candidates']}
    expected = {frozenset((a['id'], b['id'])) for i, a in enumerate(query['candidates'])
                for b in query['candidates'][i + 1:]}
    pairs = prediction['pairwise_evidence']
    if (len(pairs) != len(expected) or {frozenset((p['a'], p['b'])) for p in pairs} != expected
            or any(not isinstance(p['support'], int) or p['support'] < 0 for p in pairs)):
        raise ValueError('Incomplete or malformed candidate-pair evidence audit')
    support = sum(p['support'] > 0 for p in pairs) / len(pairs)
    if not np.isclose(support, prediction['pair_coverage'], rtol=0, atol=1e-12):
        raise ValueError('Saved support coverage disagrees with actual pairwise support')
    answer = prediction.get('prediction') or {}
    ranking = answer.get('ranking')
    choice = answer.get('choice')
    valid = (isinstance(choice, str) and choice in {'A', 'B', 'TIE'}) if query['task'] == 'pair' else (
        isinstance(ranking, list) and all(isinstance(v, str) for v in ranking)
        and len(ranking) == len(candidates) and set(ranking) == candidates)
    return {'query_id': query['query_id'], 'solute_id': query['solute_id'], 'condition_id': query['condition_id'],
            'task': query['task'], 'valid_output': float(valid), 'supported_pair_fraction': support,
            'all_candidate_pairs_supported': float(support == 1), 'no_candidate_pair_supported': float(support == 0),
            'unsupported_fallback_tie': (float(support == 0 and answer.get('choice') == 'TIE')
                                         if query['task'] == 'pair' else np.nan)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--latency-root', default='artifacts/latency/property_v1')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve earlier resource/support summaries')
    registry = load_registry()
    analysis = json.loads(Path(registry['analysis_plan']).read_text())
    sources = {'configs/study_registry_dualqa_v2.json': sha256_file('configs/study_registry_dualqa_v2.json'),
               __file__: sha256_file(__file__)}
    support, timing, pending = [], [], []
    def pin(path):
        sources[str(path)] = sha256_file(path)
    for protocol, config in registry['protocols'].items():
        qpath = Path(config['views']) / 'test_queries.jsonl'
        queries = read_jsonl(qpath)
        validate_query_manifest(queries, config['split'])
        if {q['split'] for q in queries} != {'test'}:
            raise ValueError('Only the registered frozen-test cohort is permitted')
        pin(qpath)
        pin(config['split'])
        by_id = {q['query_id']: q for q in queries}
        for mode in MODES:
            name = evidence_name(protocol, config, mode).removesuffix('_val') + '_test'
            path = Path(config['runs_root']) / name / 'predictions.jsonl'
            metadata_path = path.with_suffix(path.suffix + '.run.json')
            metadata = json.loads(metadata_path.read_text())
            if metadata['status'] != 'completed':
                raise ValueError('Support requires completed saved evidence predictions')
            predictions = read_jsonl(path)
            if len(predictions) != len(queries) or {p['query_id'] for p in predictions} != set(by_id):
                raise ValueError('Saved support and public query cohorts disagree')
            for prediction in predictions:
                support.append({'protocol': protocol, 'mode': mode, **support_row(by_id[prediction['query_id']], prediction)})
            seconds = [p['latency_seconds'] for p in predictions]
            if not all(np.isfinite(t) and t >= 0 for t in seconds):
                raise ValueError('Invalid saved direct-evidence inference times')
            timing.append({'protocol': protocol, 'run': name, 'method': 'fixed_evidence:' + mode, 'seed': None,
                           'accelerator': 'cpu', 'cpu_threads': None, 'repetition': 0,
                           'scope': 'saved_per_query_retrieval_and_decision_excludes_setup_and_persistence',
                           'seconds': sum(seconds), 'questions': len(queries),
                           'amortized_seconds_per_question': sum(seconds) / len(queries),
                           'thread_scope': 'Original evidence outputs did not record thread counts; no assumed match to new CPU4 benchmarks.'})
            pin(path)
            pin(metadata_path)
        for name in property_names(protocol, config):
            run=Path(config['runs_root'])/name
            path=run/'predictions_test.timing.json'
            metadata=json.loads(path.read_text())
            if metadata.get('partition')!='test' or metadata['records']!=10655:
                raise ValueError('Unexpected full-property timing cohort')
            seconds=metadata['elapsed_seconds']
            if not np.isfinite(seconds) or seconds<0:raise ValueError('Invalid property timing')
            is_graph='_chemprop_' in name
            timing.append({'protocol':protocol,'run':name,'method':'chemprop' if is_graph else ('nearest' if name.endswith('_nearest') else 'lightgbm'),
                'accelerator':'cuda' if is_graph else 'cpu','repetition':0,'seconds':seconds,'records':metadata['records'],
                'scope':'graph_prediction_and_persistence_excludes_graph_and_checkpoint_preparation' if is_graph else 'feature_prediction_and_persistence_model_resident',
                'amortized_seconds_per_record':seconds/metadata['records'],
                'thread_scope':'Recorded original full-test inference; distinct from CPU4 validation timing repetitions. No single-query or cross-device latency equivalence.'})
            pin(path)
    specifications=[(protocol,name.removesuffix('_val')+'_test' if name.endswith('_val') else name,folder.parent.with_name(name.removesuffix('_val')+'_test' if name.endswith('_val') else name)/'evaluation_test_view10000') for protocol,name,folder in registered_runs(registry,analysis)]
    for protocol, name, folder in specifications:
        if '_qwen_' not in name:
            continue
        path = folder.parent / 'predictions.jsonl'
        sidecar = path.with_suffix(path.suffix + '.run.json')
        if not sidecar.exists() or json.loads(sidecar.read_text()).get('status') != 'completed':
            pending.append({'run': name, 'kind': 'COMPLETE_LM_INFERENCE_TIMING'})
            continue
        metadata = json.loads(sidecar.read_text())
        predictions = read_jsonl(path)
        qpath = Path(registry['protocols'][protocol]['views']) / 'test_queries.jsonl'
        queries = read_jsonl(qpath)
        if (metadata['fingerprint']['query_sha256'] != sha256_file(qpath)
                or len(predictions) != len(queries) or {p['query_id'] for p in predictions} != {q['query_id'] for q in queries}):
            raise ValueError('LM timing must cover the complete exact final cohort')
        batches = {}
        for prediction in predictions:
            key = prediction['batch_id']
            value = prediction['batch_wall_seconds']
            if not np.isfinite(value) or value < 0 or (key in batches and batches[key] != value):
                raise ValueError('Invalid or inconsistent saved inference batch times')
            batches[key] = value
        identity, _, method, _ = _run_identity(path.parent, {'inputs': {str(path): sha256_file(path)}})
        if identity.get('_training_path'):
            pin(identity['_training_path'])
        if (path.parent/'experiment.json').exists():
            pin(path.parent/'experiment.json')
        pin('src/solvent_select/report.py')
        timing.append({'protocol': protocol, 'run': name, 'method': method,
                       'seed': metadata['fingerprint']['config']['seed'], 'accelerator': 'cuda',
                       'cpu_threads': metadata['fingerprint']['config']['cpu_threads'], 'repetition': 0,
                       'scope': 'saved_successful_batches_all_segments_excludes_setup_and_persistence',
                       'seconds': sum(batches.values()), 'questions': len(queries),
                       'amortized_seconds_per_question': sum(batches.values()) / len(queries),
                       'unique_batches': len(batches), 'final_batch_size': metadata['final_inference_batch_size'],
                       'already_complete_on_last_invocation': metadata['already_complete'],
                       'peak_allocated_bytes':metadata.get('peak_allocated_bytes'),'peak_reserved_bytes':metadata.get('peak_reserved_bytes'),
                       'shared_inference_source':metadata.get('shared_inference_source'),
                       'unique_inference_source':metadata.get('shared_inference_source',str(path)),
                       'group_scope': 'Per-run observations only; do not present an incomplete learning-seed group as its three-seed estimate.'})
        pin(path)
        pin(sidecar)
    output.mkdir(parents=True)
    frame = pd.DataFrame(support)
    columns = ['valid_output', 'supported_pair_fraction', 'all_candidate_pairs_supported',
               'no_candidate_pair_supported', 'unsupported_fallback_tie']
    per_condition = frame.groupby(['protocol', 'mode', 'solute_id', 'condition_id', 'task'])[columns].mean()
    per_solute = per_condition.groupby(['protocol', 'mode', 'solute_id', 'task']).mean().reset_index()
    summary = per_solute.groupby(['protocol', 'mode', 'task'])[columns].mean()
    summary['solute_count'] = per_solute.groupby(['protocol', 'mode', 'task']).size()
    summary.reset_index().to_csv(output / 'evidence_support.csv', index=False)
    per_solute.to_csv(output / 'evidence_support_per_solute.csv', index=False)
    pd.DataFrame(timing).to_csv(output / 'latency_observations.csv', index=False)
    write_json(output / 'manifest.json', {'status': 'FINAL_RESOURCE_SUMMARY_COMPLETE' if not pending else 'FINAL_RESOURCE_SUMMARY_INCOMPLETE',
        'pending': pending, 'sources': sources, 'target_files_read': [], 'test_queries_read': True,
        'support_aggregation': 'Query mean within condition, condition mean within solute, solute macro mean; pair/rank separate.',
        'interpretation': 'JSON-valid output and supported training candidate pairs are distinct. All original scientific scores retain all questions, including unsupported fallbacks. Timing scopes, devices, feature caches and CPU thread availability remain explicit; no pooled cross-scope latency or isolated-host claim. Full-test property timings cover10655measurement records; LM/evidence timings cover4546questions. CPU/GPU and preparation/persistence scopes differ; do not treat their quotients as controlled speedups. Shared zero-shot source timing is a reused observation, not additional compute.',
        'outputs': {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'support_rows': len(summary), 'timing_observations': len(timing), 'pending': len(pending)}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Build the experiment master table from wrappers AND child completion evidence."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json

import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get(path):
    return json.loads(Path(path).read_text())


def effective_status(wrapper, child):
    # A signal-saving Trainer may exit0 with status=interrupted.
    if child.get('status') in ('failed', 'interrupted'):
        return child['status']
    if wrapper.get('status') in ('failed', 'interrupted'):
        return wrapper['status']
    if not wrapper and child.get('status') == 'COMPLETED_VALIDATION_CPU_TIMING':
        return 'completed'
    if wrapper.get('status') == 'completed' and wrapper.get('exit_code') == 0:
        return 'completed' if child.get('status') == 'completed' else 'UNVERIFIED_CHILD_COMPLETION'
    return child.get('status', wrapper.get('status', 'unknown'))


def build(root, output):
    root, output = Path(root), Path(output)
    attempts = []
    ledger = root / 'artifacts/experiments.jsonl'
    if ledger.exists():
        with ledger.open() as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            attempts = [json.loads(line) for line in stream if line.strip()]
    registry_path = root / 'configs/study_registry_dualqa_v2.json'
    layouts = get(registry_path)['protocols'] if registry_path.exists() else {}
    rows, sources = [], {}
    roots=[p for p in (root / 'artifacts').glob('runs*') if p.is_dir()]
    run_directories={p.parent for directory in roots for name in
                     ['experiment.json','metadata.json','run.json','predictions.jsonl.run.json']
                     for p in directory.glob('*/'+name)}
    run_directories.update(p.parent for p in (root/'artifacts/consistency').rglob('run.json'))
    run_directories.update(p.parent for p in (root/'artifacts/latency').rglob('run.json'))
    for run in sorted(run_directories):
        wrapper_path = run/'experiment.json'
        wrapper = get(wrapper_path) if wrapper_path.exists() else {}
        child_path = next((run / name for name in ['metadata.json', 'run.json', 'predictions.jsonl.run.json']
                           if (run / name).exists()), None)
        child = get(child_path) if child_path else {}
        if wrapper_path.exists():
            sources[str(wrapper_path)] = digest(wrapper_path)
        if child_path:
            sources[str(child_path)] = digest(child_path)
        metrics, consistency = {}, {}
        # Validation-only inventory: test scoring remains a distinct frozen stage.
        for path in run.glob('evaluation*/metrics.json'):
            if 'test' in path.parent.name:
                continue
            data = get(path)
            if 'by_variant_task' in data:
                if data.get('partition')=='val':
                    sources[str(path)] = digest(path)
                    consistency[path.parent.name]=data
                continue
            if 'by_task' not in data:
                continue
            inputs = data.get('inputs', {})
            if any('test_queries' in key for key in inputs):
                continue
            sources[str(path)] = digest(path)
            metrics[path.parent.name] = {'queries': data.get('queries'), 'solutes': data.get('solutes'),
                                        'by_task': data['by_task'], 'inputs': inputs}
        config = child.get('config', child.get('fingerprint', {}).get('config', {}))
        fitted_metadata={}
        if child.get('status')=='COMPLETED_VALIDATION_CPU_TIMING' and child.get('run'):
            fitted_path=root/child['run']/'metadata.json'
            if fitted_path.exists():
                fitted_metadata=get(fitted_path)
                sources[str(fitted_path)]=digest(fitted_path)
        for key in ['model','checkpoint']:
            if not child.get(key):
                continue
            asset=Path(child[key])
            if not asset.is_absolute():asset=root/asset
            metadata=asset.parent/'metadata.json' if key=='model' else asset.parent.parent/'metadata.json'
            if metadata.exists():
                fitted_metadata=get(metadata)
                sources[str(metadata)]=digest(metadata)
        interrupted = []
        for path in list(run.glob('recovery*run.json'))+list(run.glob('recovery_segments/*/metadata.json')):
            record = get(path)
            if record.get('status') == 'interrupted':
                interrupted.append(str(path))
                sources[str(path)] = digest(path)
        profile = run / 'pilot_resource_report.json'
        profile_path = str(profile) if profile.exists() else None
        if profile.exists():
            sources[str(profile)] = digest(profile)
        kind=wrapper.get('kind') or ('validation_cpu_latency' if child.get('status')=='COMPLETED_VALIDATION_CPU_TIMING' else
               'validation_input_consistency' if consistency else
               child.get('fingerprint',{}).get('prediction_kind') or child.get('method'))
        namespace = [(p, c) for p, c in layouts.items()
                     if (run.parent.resolve() == (root / c['runs_root']).resolve()
                         and run.name.startswith(c['run_prefix'] + p + '_'))
                     or run.parent.resolve() == (root / c['consistency_runs_root'] / p).resolve()
                     or (child.get('status')=='COMPLETED_VALIDATION_CPU_TIMING' and child.get('run')
                         and (root/child['run']).parent.resolve()==(root/c['runs_root']).resolve()
                         and Path(child['run']).name.startswith(c['run_prefix']+p+'_'))]
        if len(namespace) > 1:
            raise ValueError('Ambiguous registered data namespace: ' + str(run))
        protocol=wrapper.get('protocol') or child.get('protocol') or (namespace[0][0] if namespace else None) or next((p for p in ['identity','scaffold']
                 if p in run.parts or run.name.startswith(p+'_') or run.name.startswith('sourceqa_'+p+'_')
                 or run.name.startswith('strictqa_'+p+'_')),None)
        split_path = child.get('splits', fitted_metadata.get('splits'))
        data_tag = namespace[0][1]['data_tag'] if namespace else (Path(split_path).parent.name if split_path else None)
        if data_tag == 'splits':
            data_tag = 'initial'
        row = {'run_id': run.name, 'kind': kind, 'protocol': protocol,
               'data_tag':data_tag,'study_scope':'REGISTERED_DATA_NAMESPACE' if namespace else 'HISTORICAL_OR_UNCLASSIFIED',
               'run_directory':str(run), 'runs_root':wrapper.get('runs_root',str(run.parent)),
               'input_fingerprints':json.dumps(child.get('inputs',child.get('fingerprint',{})),sort_keys=True),
               'data_code_model_provenance':json.dumps({k:v for k,v in child.items() if k in
                    ['code_sha256','model','model_sha256','checkpoint','checkpoint_sha256','scalers_sha256',
                     'measurements','measurement_sha256','splits','split_sha256','versions','environment']},sort_keys=True),
               'seed': wrapper.get('seed', child.get('seed',config.get('seed',fitted_metadata.get('seed')))), 'status': effective_status(wrapper, child),
               'referenced_fitted_model':json.dumps({k:v for k,v in fitted_metadata.items() if k in
                    ['method','seed','measurement_sha256','split_sha256','versions','code_sha256']},sort_keys=True),
               'wrapper_status': wrapper.get('status'), 'child_status': child.get('status'),
               'wrapper_pid': wrapper.get('wrapper_pid'), 'child_pid': child.get('pid', wrapper.get('pid')),
               'started_utc': wrapper.get('started_utc'), 'finished_utc': wrapper.get('finished_utc'),
               'wrapper_wall_seconds': wrapper.get('elapsed_seconds'),
               'training_or_prediction_segment_seconds': child.get('elapsed_seconds'),
               'total_training_segment_seconds': child.get('total_elapsed_seconds'),
               'peak_allocated_bytes': child.get('peak_allocated_bytes'), 'peak_reserved_bytes': child.get('peak_reserved_bytes'),
               'train_records': child.get('train_records'), 'train_examples': child.get('train_examples'),
               'global_step': child.get('global_step'), 'pilot_cumulative_resource_profile': profile_path,
               'ledger_attempts': sum(a.get('run_id') == run.name for a in attempts),
               'interrupted_segment_evidence': json.dumps(interrupted), 'config': json.dumps(config, sort_keys=True),
               'command': json.dumps(wrapper.get('command')), 'wrapper_evidence': str(wrapper_path) if wrapper_path.exists() else None,
               'child_evidence': str(child_path) if child_path else None, 'validation_metrics': json.dumps(metrics, sort_keys=True),
               'validation_consistency_metrics':json.dumps(consistency,sort_keys=True)}
        if child.get('status')=='COMPLETED_VALIDATION_CPU_TIMING':
            row['validation_cpu_timing']=json.dumps({key:child[key] for key in
                ['accelerator','cpu_threads','public_questions','unique_candidate_calls','candidate_appearances',
                 'model_loading_seconds','timings','implementation']},sort_keys=True)
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    manifest = {'status': 'VALIDATION_STAGE_INVENTORY', 'runs': len(rows),
                'counts_by_status': pd.Series([r['status'] for r in rows]).value_counts().to_dict(),
                'output': str(output), 'output_sha256': digest(output), 'sources': sources,
                'code_sha256': digest(__file__),
                'registry_sha256':digest(registry_path) if registry_path.exists() else None,
                'notes': ['Wrapped completion requires wrapper exit0 and child completed; standalone CPU evidence/consistency runs require their completed child metadata. Failed/interrupted status takes precedence.',
                          'Historical wrapper attempts remain unchanged in the JSONL ledger; archived interrupted child records are linked.',
                          'Pilot and resource probes are separate kinds, not formal scientific results. Test metrics are not included.',
                          'A live process must still be checked in /proc before starting or resuming; this table is a snapshot.']}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--output', default='artifacts/experiments.csv')
    args = parser.parse_args()
    result = build(args.root, args.output)
    print(json.dumps({key: result[key] for key in ['runs', 'counts_by_status', 'output']}))


if __name__ == '__main__':
    main()

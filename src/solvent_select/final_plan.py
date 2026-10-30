"""Derive the complete final task graph from verified, immutable releases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .freeze import option
from .llm import write_json
from .release_guard import verify_release
from .views import sha256_file


def set_option(command, key, value):
    command = command.copy()
    if key in command:
        command[command.index(key)+1] = str(value)
    else:
        command += [key, str(value)]
    return command


def reusable_zero_shot(first, second):
    """Same unadapted public prompts and decoding; split audit hashes may differ."""
    a, b = first['fingerprint'].copy(), second['fingerprint'].copy()
    if (a.get('prediction_kind') != 'llm' or b.get('prediction_kind') != 'llm'
            or a.get('adapter_sha256') or b.get('adapter_sha256')
            or a.get('config', {}).get('retrieval_mode') != 'none'
            or b.get('config', {}).get('retrieval_mode') != 'none'):
        return False
    a.pop('manifest_sha256', None)
    b.pop('manifest_sha256', None)
    return a == b


def validation_name(spec,protocol):
    fallback=f"sourceqa_{protocol}_evidence_{spec['retrieval_mode']}_val"
    return spec.get('validation_run_id') or option(spec['validation_command'],'--run-id',fallback)


def primary_evaluations(release,protocol,mode):
    result=[]
    for seed in [17,42,2026]:
        matches=[s for s in release['prediction_specifications'] if not s['consistency']
                 and s['prediction_kind']=='llm' and s['retrieval_mode']==mode
                 and s['fingerprint']['config']['seed']==seed
                 and (s.get('adaptation')=='rag_sft' or s.get('adaptation') is None and '_qwen_rag_' in str(s.get('adapter','')))]
        if len(matches)!=1:raise ValueError('Primary comparison does not have one mixed-RAG adapter per seed/mode')
        name=validation_name(matches[0],protocol).removesuffix('_val')+'_test'
        result.append(str(Path(release.get('runs_root','artifacts/runs_sourceqa'))/name/'evaluation_test_view10000'))
    return result


def assemble(releases):
    """Pure definition; no target files, model weights or outputs are opened."""
    tasks, seen_zero_shot = [], []
    def add(name, protocol, command, outputs, **kwargs):
        tasks.append({'id': name, 'protocol': protocol, 'action': 'command',
                      'command': command, 'resume_command': command.copy(), 'outputs': outputs, **kwargs})
    def evaluation(protocol, release, run, predictions, prediction_task, prop=False):
        output = run / 'evaluation_test_view10000'
        command = ['.venv/bin/python', '-m', 'solvent_select.evaluate', '--queries', release['query_path'],
                   '--targets', release['target_path'], '--predictions', str(predictions), '--output', str(output),
                   '--freeze-manifest', release['_path']] + (['--property-predictions'] if prop else [])
        add(run.name+'_evaluate', protocol, command,
            [{'path': str(output/f)} for f in ['metrics.json', 'per_query.parquet', 'per_solute.csv']],
            depends_on=[prediction_task])
        command = ['.venv/bin/python', '-m', 'solvent_select.error_analysis', '--queries', release['query_path'],
                   '--targets', release['target_path'], '--predictions', str(predictions),
                   '--per-query', str(output/'per_query.parquet'), '--measurements', release['_measurements'],
                   '--manifest', release['split_path'], '--output', str(run/'error_analysis_test_v1'),
                   '--freeze-manifest', release['_path']]
        add(run.name+'_error_analysis', protocol, command,
            [{'path': str(run/'error_analysis_test_v1/analysis.json'), 'status': 'FROZEN_TEST_DIAGNOSTICS'}],
            depends_on=[prediction_task,run.name+'_evaluate'])
    for protocol in ['identity', 'scaffold']:
        release = dict(releases[protocol])
        release['_measurements'] = release.get('measurement_path',f'data/processed/sourceqa/{"measurements" if protocol=="identity" else "measurements_scaffold"}.parquet')
        for model in release['property_models']:
            run = Path(model['run'])
            predictions = run/'predictions_test.parquet'
            interpreter = '.venv-chemprop/bin/python' if model['method']=='chemprop' else '.venv/bin/python'
            module = 'solvent_select.chemprop_runner' if model['method']=='chemprop' else 'solvent_select.baselines'
            command = [interpreter, '-m', module, 'predict', '--run', str(run), '--partition', 'test',
                       '--output', str(predictions), '--freeze-manifest', release['_path']]
            if model['method']=='chemprop':
                command += ['--accelerator', 'gpu']
            add(model['name']+'_predict_test', protocol, command,
                [{'path': str(predictions)}, {'path': str(predictions.with_suffix('.timing.json'))}],
                property_prediction={'path':str(predictions),'measurements':release['_measurements'],
                                     'manifest':release['split_path']})
            evaluation(protocol, release, run, predictions, model['name']+'_predict_test', prop=True)
        # Full-cohort inferences precede the transformed counterparts they score.
        specs = sorted(release['prediction_specifications'], key=lambda s: s['consistency'])
        full_predictions = {}
        for spec in specs:
            original = spec['validation_command']
            base_name = validation_name(spec,protocol).removesuffix('_val')
            name = base_name + ('_consistency_test' if spec['consistency'] else '_test')
            parent = Path(release.get('consistency_runs_root','artifacts/consistency/sourceqa'))/protocol if spec['consistency'] else Path(release.get('runs_root','artifacts/runs_sourceqa'))
            run = parent/name
            predictions = run/'predictions.jsonl'
            command = original[original.index('--')+1:] if '--' in original else original.copy()
            command = set_option(command, '--queries', spec['queries'])
            command = set_option(command, '--output', predictions)
            command = set_option(command, '--manifest', release['split_path'])
            command = set_option(command, '--freeze-manifest', release['_path'])
            command += ['--allow-test']
            prefix = ['.venv/bin/python', 'scripts/run_experiment.py', '--run-id', name, '--kind',
                      'frozen_'+spec['prediction_kind'], '--protocol', protocol, '--seed',
                      str(spec['fingerprint']['config']['seed']), '--runs-root', str(parent)]
            outputs = [{'path': str(predictions)}, {'path': str(predictions)+'.run.json', 'status': 'completed'}]
            task = {'id': name, 'protocol': protocol, 'action': 'command', 'command': prefix+['--']+command,
                    'resume_command': prefix+['--resume','--']+command+['--resume','latest'], 'outputs': outputs,
                    'query_source': spec['queries'], 'prediction_path': str(predictions),
                    'prediction_specification': spec}
            if spec['prediction_kind']=='evidence':
                task['env'] = {'CUDA_VISIBLE_DEVICES': ''}
            reusable = next((prior for prior in seen_zero_shot if reusable_zero_shot(prior['spec'],spec)), None)
            if reusable:
                task.update(action='reuse_zero_shot', source_prediction=reusable['prediction'],
                            source_specification=reusable['spec'], source_release=reusable['release'],
                            depends_on=[reusable['task_id']])
                task.pop('command'); task.pop('resume_command')
            elif spec['prediction_kind']=='llm' and not spec['adapter'] and spec['retrieval_mode']=='none':
                seen_zero_shot.append({'spec':spec, 'prediction':str(predictions), 'release':release['_path'],'task_id':name})
            tasks.append(task)
            key = json.dumps(original, sort_keys=True)
            if not spec['consistency']:
                full_predictions[key] = {'path':str(predictions),'task_id':name}
                evaluation(protocol, release, run, predictions, name)
            else:
                output = run/'evaluation'
                command = ['.venv/bin/python','-m','solvent_select.consistency','score',
                           '--folder', str(Path(release['consistency_manifest_path']).parent),
                           '--base-predictions', full_predictions[key]['path'], '--transformed-predictions',str(predictions),
                           '--output',str(output),'--freeze-manifest',release['_path']]
                add(name+'_score',protocol,command,[{'path':str(output/f)} for f in ['metrics.json','per_query.parquet','per_solute.csv']],
                    depends_on=[full_predictions[key]['task_id'],name])
        for control in release['cpu_consistency_controls']:
            name = control['id']+'_consistency_test'
            run = Path(release.get('consistency_runs_root','artifacts/consistency/sourceqa'))/protocol/name
            is_chemprop = control['kind']=='property' and any(m['run']==control['model_run'] and m['method']=='chemprop' for m in release['property_models'])
            interpreter = '.venv-chemprop/bin/python' if is_chemprop else '.venv/bin/python'
            command = [interpreter,'-m','solvent_select.final_controls','--release',release['_path'],
                       '--control-id',control['id'],'--output',str(run)]
            add(name,protocol,command,[{'path':str(run/'run.json'),'status':'completed'},
                  *[{'path':str(run/f)} for f in ['base_predictions.jsonl','transformed_predictions.jsonl',
                    'evaluation/metrics.json','evaluation/per_query.parquet','evaluation/per_solute.csv']]],
                env={'CUDA_VISIBLE_DEVICES':''})
    for protocol in ['identity','scaffold']:
        release = releases[protocol]
        output = f'artifacts/reports/frozen_dualqa/{protocol}_primary_condition_minus_structure.json'
        command = ['.venv/bin/python','-m','solvent_select.compare','--first',
                   *primary_evaluations(release,protocol,'condition'),
                   '--second',*primary_evaluations(release,protocol,'structure'),
                   '--freeze-manifest',release['_path'],'--output',output]
        add(protocol+'_primary_comparison',protocol,command,[{'path':output,'status':'FROZEN_TEST_COMPARISON'}],
            depends_on=[Path(path).parent.name+'_evaluate' for mode in ['condition','structure'] for path in primary_evaluations(release,protocol,mode)])
    ids=[t['id'] for t in tasks]
    if len(ids)!=len(set(ids)):
        raise ValueError('Final task IDs collide')
    earlier=set()
    for task in tasks:
        if not set(task.get('depends_on',[]))<=earlier:
            raise ValueError('Final dependency is missing or scheduled after its consumer')
        earlier.add(task['id'])
    return tasks


def prepare(release_paths, output):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Final execution plans are immutable')
    releases, hashes={},{}
    for path in release_paths:
        verified=verify_release(path)
        protocol=verified['protocol']
        if protocol in releases:
            raise ValueError('Duplicate protocol release')
        releases[protocol]={**json.loads(Path(path).read_text()),'_path':str(path)}
        hashes[str(path)]=sha256_file(path)
    if set(releases)!={'identity','scaffold'}:
        raise ValueError('Both protocol releases are required')
    if releases['identity']['query_sha256']!=releases['scaffold']['query_sha256']:
        raise ValueError('Protocols must share the global query cohort')
    tasks=assemble(releases)
    output.parent.mkdir(parents=True,exist_ok=True)
    plan={'status':'FROZEN_FINAL_EXECUTION_PLAN','partition':'test','release_sha256':hashes,
          'tasks':tasks,'code_sha256':sha256_file(__file__),
          'policy':'Every registered method and control retained; fixed order; no score-dependent branching; reused zero-shot runs require exact-input proof.'}
    write_json(output,plan)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',nargs=2,required=True)
    parser.add_argument('--output',default='configs/final_queue_dualqa_v2.json')
    args=parser.parse_args()
    plan=prepare(args.release,args.output)
    print(json.dumps({'status':plan['status'],'tasks':len(plan['tasks'])}))


if __name__=='__main__':
    main()

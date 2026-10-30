#!/usr/bin/env python3
"""Prepare the serial LM validation-consistency queue after development completes.

This does not execute a model, open targets or write a final-test release.
The source plan and its completed outputs become immutable queue inputs.
"""
import argparse
import json
from pathlib import Path

from solvent_select.freeze import option, queue_evidence
from solvent_select.views import sha256_file
from solvent_select.study import DEFAULT_REGISTRY,load_registry,queue_pairs


CORE_MODES = {'none', 'text', 'structure', 'condition'}


def replace_option(command, name, value):
    result = command.copy()
    result[result.index(name) + 1] = value
    return result


def assemble_tasks(source, output_root='artifacts/consistency/sourceqa',
                   query_root='data/consistency/sourceqa'):
    tasks, pinned = [], set()
    for task in source['tasks']:
        command = task['command']
        if ('solvent_select.llm' not in command or 'predict' not in command
                or option(command, '--retrieval-mode') not in CORE_MODES):
            continue
        protocol = option(command, '--protocol')
        name = task['id'].removesuffix('_val') + '_consistency_val'
        parent = Path(output_root) / protocol
        run = parent / name
        folder = Path(query_root) / f'{protocol}_val'
        inner = command[command.index('--') + 1:]
        base_predictions = option(inner, '--output')
        inner = replace_option(inner, '--queries', str(folder / 'transformed_queries.jsonl'))
        inner = replace_option(inner, '--output', str(run / 'predictions.jsonl'))
        prefix = ['.venv/bin/python', 'scripts/run_experiment.py', '--run-id', name,
                  '--kind', 'llm_consistency_validation', '--protocol', protocol,
                  '--seed', option(command, '--seed', '17'), '--runs-root', str(parent)]
        tasks.append({'id': name, 'command': prefix + ['--'] + inner,
                      'resume_command': prefix + ['--resume', '--'] + inner + ['--resume', 'latest'],
                      'outputs': [{'path': str(run / 'predictions.jsonl')},
                                  {'path': str(run / 'predictions.jsonl.run.json'), 'status': 'completed'}]})
        score_command = ['.venv/bin/python', '-m', 'solvent_select.consistency', 'score',
                         '--folder', str(folder), '--base-predictions', base_predictions,
                         '--transformed-predictions', str(run / 'predictions.jsonl'),
                         '--output', str(run / 'evaluation')]
        tasks.append({'id': name + '_score', 'command': score_command,
                      'resume_command': score_command.copy(),
                      'outputs': [{'path': str(run / 'evaluation' / file)} for file in
                                  ['metrics.json', 'per_query.parquet', 'per_solute.csv']]})
        pinned.update([base_predictions, base_predictions + '.run.json'])
        pinned.update(str(folder / file) for file in ['manifest.json', 'base_queries.jsonl',
                                                     'transformed_queries.jsonl', 'mapping.jsonl'])
        adapter = option(inner, '--adapter')
        if adapter:
            pinned.update(str(Path(adapter) / file) for file in ['adapter_model.safetensors', 'adapter_config.json'])
            pinned.add(str(Path(adapter).parent / 'run.json'))
    return tasks, pinned


def prepare(plan_path=None,state_path=None,output='configs/consistency_queue_dualqa_v2.json',registry_path=DEFAULT_REGISTRY):
    registry=load_registry(registry_path)
    sources=queue_pairs(registry) if plan_path is None else [(plan_path,state_path)]
    audits=[queue_evidence(*pair) for pair in sources]
    if any(audit['errors'] or audit['pending'] for audit in audits):
        raise ValueError('Source development queues are not complete and verified; no consistency queue written')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Never replace an existing queue')
    tasks,pinned=[],set()
    for (plan_path,state_path),audit in zip(sources,audits):
        source=json.loads(Path(plan_path).read_text())
        if source.get('partition')!='val' or not source.get('test_forbidden'):
            raise ValueError('Require validation-only source plans')
        for protocol,config in registry['protocols'].items():
            selected=[task for task in source['tasks'] if '--protocol' in task['command']
                      and task['command'][task['command'].index('--protocol')+1]==protocol]
            generated,inputs=assemble_tasks({'tasks':selected},config['consistency_runs_root'],config['consistency_data_root'])
            tasks.extend(generated);pinned.update(inputs)
        pinned.update(source['input_sha256'])
        pinned.update([str(plan_path),str(state_path)])
    if len(tasks) != 64:
        raise ValueError('Expected32registered core predictions and32scores')
    pinned.update([str(registry_path),__file__,'src/solvent_select/consistency.py'])
    data = {'status': 'VALIDATION_ONLY_REGISTERED', 'partition': 'val', 'test_forbidden': True,
            'dataset_version': 'dualqa_v2', 'source_completion': audits,
            'input_sha256': {str(p): sha256_file(p) for p in sorted(pinned)}, 'tasks': tasks}
    output.write_text(json.dumps(data, indent=2) + '\n')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    parser.add_argument('--state')
    parser.add_argument('--registry',default=DEFAULT_REGISTRY)
    parser.add_argument('--output', default='configs/consistency_queue_dualqa_v2.json')
    args = parser.parse_args()
    data = prepare(args.plan,args.state,args.output,args.registry)
    print(json.dumps({'tasks': len(data['tasks']), 'output': args.output}))


if __name__ == '__main__':
    main()

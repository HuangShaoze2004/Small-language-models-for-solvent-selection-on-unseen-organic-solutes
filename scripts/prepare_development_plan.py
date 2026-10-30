#!/usr/bin/env python3
"""Create the bounded formal validation plan, never touching final test labels."""
from pathlib import Path
import argparse
import hashlib
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [17, 42, 2026]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-tag', default='')
    parser.add_argument('--protocol',choices=['all','identity','scaffold'],default='all')
    parser.add_argument('--runs-root', default='artifacts/runs')
    parser.add_argument('--run-prefix', default='')
    parser.add_argument('--protocol-path', default='configs/development_protocol.json')
    parser.add_argument('--plan-output', default='configs/development_queue.json')
    args = parser.parse_args()
    protocols=['identity','scaffold'] if args.protocol=='all' else [args.protocol]
    tasks, pinned = [], set()
    def pin(*paths):
        pinned.update(str(p) for p in paths)
    def output(path, status=None):
        return {'path': path, **({'status': status} if status else {})}
    def wrapped(name, kind, protocol, seed, command, artifacts, resumable=False):
        prefix = ['.venv/bin/python', 'scripts/run_experiment.py', '--run-id', name, '--kind', kind,
                  '--protocol', protocol, '--seed', str(seed)]
        task = {'id': name, 'command': prefix + ['--'] + command, 'outputs': artifacts}
        if resumable:
            task['resume_command'] = prefix + ['--resume', '--'] + command + ['--resume', 'latest']
        tasks.append(task)
    def evaluate(name, protocol, predictions, property_predictions=False):
        path = f'artifacts/runs/{name}/evaluation_view10000'
        task = {'id': name + '_evaluate', 'command': ['.venv/bin/python', '-m', 'solvent_select.evaluate',
                      '--queries', f'data/views/{protocol}/val_queries.jsonl',
                      '--targets', f'data/views/{protocol}/val_targets.jsonl',
                      '--predictions', predictions, '--output', path] + (['--property-predictions'] if property_predictions else []),
                      'outputs': [output(path + '/metrics.json'), output(path + '/per_query.parquet'), output(path + '/per_solute.csv')]}
        task['resume_command'] = task['command'].copy()
        tasks.append(task)
    def predict(protocol, seed, mode, adapter_kind=None):
        label = adapter_kind or 'base'
        name = f'{protocol}_qwen_{label}_{seed}_{mode}_val'
        config = 'configs/llm_formal_rag.yaml' if adapter_kind == 'rag' else 'configs/llm_formal_sft.yaml'
        command = ['.venv/bin/python', '-u', '-m', 'solvent_select.llm', 'predict', '--config', config,
                   '--seed', str(seed), '--retrieval-mode', mode, '--manifest', f'data/splits/{protocol}.csv',
                   '--queries', f'data/views/{protocol}/val_queries.jsonl',
                   '--output', f'artifacts/runs/{name}/predictions.jsonl']
        if adapter_kind:
            command += ['--adapter', f'artifacts/runs/{protocol}_qwen_{adapter_kind}_{seed}/adapter']
        if mode != 'none':
            command += ['--measurements', 'data/processed/measurements.parquet' if protocol == 'identity'
                        else 'data/processed/measurements_scaffold.parquet']
        wrapped(name, 'llm_validation', protocol, seed, command,
                [output(f'artifacts/runs/{name}/predictions.jsonl.run.json', 'completed'),
                 output(f'artifacts/runs/{name}/predictions.jsonl')], resumable=True)
        evaluate(name, protocol, f'artifacts/runs/{name}/predictions.jsonl')
    for kind in ['sft', 'rag']:
        path = Path(f'configs/llm_formal_{kind}.yaml')
        data = yaml.safe_load(Path('configs/llm_sft.yaml' if kind == 'sft' else 'configs/llm_rag_sft.yaml').read_text())
        data.update(micro_batch_size=4 if kind == 'sft' else 1,
                    gradient_accumulation_steps=4 if kind == 'sft' else 16,
                    max_steps=625, eval_steps=125, save_steps=125,
                    max_train_examples=0, max_val_examples=384)
        text = '# Fixed validation-only budget:10000 examples, effective batch16, at most625 updates.\n' + yaml.safe_dump(data, sort_keys=False)
        if path.exists() and path.read_text() != text:
            raise FileExistsError(path)
        if not path.exists():path.write_text(text)
        pin(path)
    # Fast strong strict baselines precede lengthy language-model work.
    for seed in (SEEDS if 'scaffold' in protocols else []):
        name = f'scaffold_chemprop_{seed}'
        command = ['.venv-chemprop/bin/python', '-u', '-m', 'solvent_select.chemprop_runner', 'train',
                   '--config', 'configs/baselines_chemprop.yaml', '--data', 'data/processed/measurements_scaffold.parquet',
                   '--splits', 'data/splits/scaffold.csv', '--seed', str(seed), '--output', f'artifacts/runs/{name}',
                   '--purpose', 'formal', '--accelerator', 'gpu']
        wrapped(name, 'chemprop', 'scaffold', seed, command,
                [output(f'artifacts/runs/{name}/metadata.json', 'completed'),
                 output(f'artifacts/runs/{name}/predictions_val.parquet'),
                 output(f'artifacts/runs/{name}/checkpoints/last.ckpt')])
        # Chemprop uses boolean resume, so its resume command has no 'latest' value.
        tasks[-1]['resume_command'] = tasks[-1]['command'].copy()
        tasks[-1]['resume_command'].insert(tasks[-1]['resume_command'].index('--'), '--resume')
        tasks[-1]['resume_command'] += ['--resume']
        evaluate(name, 'scaffold', f'artifacts/runs/{name}/predictions_val.parquet', True)
    for protocol in protocols:
        for seed in SEEDS:
            for kind in ['sft', 'rag']:
                name = f'{protocol}_qwen_{kind}_{seed}'
                command = ['.venv/bin/python', '-u', '-m', 'solvent_select.llm', 'train',
                           '--config', f'configs/llm_formal_{kind}.yaml', '--seed', str(seed),
                           '--manifest', f'data/splits/{protocol}.csv',
                           '--train-queries', f'data/views/{protocol}/train_queries.jsonl',
                           '--train-targets', f'data/views/{protocol}/train_targets.jsonl',
                           '--val-queries', f'data/views/{protocol}/val_queries.jsonl',
                           '--val-targets', f'data/views/{protocol}/val_targets.jsonl',
                           '--output', f'artifacts/runs/{name}']
                if kind == 'rag':
                    command += ['--measurements', 'data/processed/measurements.parquet' if protocol == 'identity'
                                else 'data/processed/measurements_scaffold.parquet']
                wrapped(name, 'qlora_sft' if kind == 'sft' else 'qlora_rag_sft', protocol, seed, command,
                        [output(f'artifacts/runs/{name}/run.json', 'completed'),
                         output(f'artifacts/runs/{name}/adapter/adapter_model.safetensors')], resumable=True)
                modes = (['none', 'condition'] if protocol == 'identity' else ['none']) if kind == 'sft' else (
                    ['text', 'structure', 'condition', 'no_temperature', 'no_solvent'] if protocol == 'identity'
                    else ['structure', 'condition'])
                for mode in modes:
                    predict(protocol, seed, mode, kind)
        for mode in ['none', 'text', 'structure', 'condition']:
            # Base model is deterministic; do not mechanically repeat seeds.
            predict(protocol, 17, mode)
        pin(f'data/splits/{protocol}.csv', f'data/views/{protocol}/view_manifest.json')
        for split in ['train', 'val']:
            pin(f'data/views/{protocol}/{split}_queries.jsonl', f'data/views/{protocol}/{split}_targets.jsonl')
    pin('src/solvent_select/llm.py', 'src/solvent_select/retrieval.py', 'src/solvent_select/evaluate.py',
        'src/solvent_select/data.py', 'src/solvent_select/baselines.py', 'scripts/run_experiment.py',
        'src/solvent_select/chemprop_runner.py', 'scripts/run_development_queue.py', 'scripts/env.sh',
        'configs/baselines_chemprop.yaml', 'configs/development_protocol.json', 'requirements.lock', 'requirements-chemprop.lock')
    for protocol in protocols:
        pin('data/processed/measurements.parquet' if protocol=='identity' else 'data/processed/measurements_scaffold.parquet')
    def remap(token):
        if token == 'configs/development_protocol.json':
            return args.protocol_path
        if args.data_tag:
            for prefix in ['data/processed/', 'data/targets/', 'data/views/', 'data/splits/']:
                if token.startswith(prefix):
                    return prefix + args.data_tag + '/' + token[len(prefix):]
        if token.startswith('artifacts/runs/'):
            tail = token[len('artifacts/runs/'):]
            return args.runs_root.rstrip('/') + '/' + args.run_prefix + tail
        return token
    for task in tasks:
        task['id'] = args.run_prefix + task['id']
        for field in ['command', 'resume_command']:
            if field not in task:
                continue
            command = [remap(token) for token in task[field]]
            if '--run-id' in command:
                i = command.index('--run-id') + 1
                command[i] = args.run_prefix + command[i]
                if args.runs_root != 'artifacts/runs':
                    i = command.index('--')
                    command[i:i] = ['--runs-root', args.runs_root]
            task[field] = command
        for item in task['outputs']:
            item['path'] = remap(item['path'])
    pinned = {remap(p) for p in pinned}
    plan = {'status': 'VALIDATION_ONLY_REGISTERED', 'partition': 'val', 'test_forbidden': True,
            'dataset_version': args.data_tag or 'initial',
            'protocol': args.protocol_path, 'input_sha256': {p: digest(p) for p in sorted(pinned)},
            'tasks': tasks}
    path = Path(args.plan_output)
    if path.exists():
        raise FileExistsError('Never overwrite an existing queue plan')
    path.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'tasks': len(tasks), 'plan': str(path), 'sha256': digest(path)}))


if __name__ == '__main__':
    main()

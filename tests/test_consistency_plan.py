"""Validation consistency must keep exact base-model settings and no targets."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('consistency_plan', Path(__file__).parents[1] / 'scripts/prepare_consistency_plan.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_adapter_config_and_budget_are_reused_without_test_or_target():
    command = ['.venv/bin/python', 'scripts/run_experiment.py', '--protocol', 'identity', '--seed', '42', '--',
               '.venv/bin/python', '-m', 'solvent_select.llm', 'predict', '--config', 'same.yaml',
               '--retrieval-mode', 'condition', '--adapter', 'trained/adapter', '--manifest', 'splits.csv',
               '--queries', 'data/views/sourceqa/identity/val_queries.jsonl', '--output', 'run/predictions.jsonl']
    source = {'tasks': [{'id': 'sourceqa_identity_qwen_rag_42_condition_val', 'command': command}]}
    tasks, pins = module.assemble_tasks(source)
    assert len(tasks) == 2
    predict = tasks[0]['command']
    assert module.option(predict, '--config') == 'same.yaml'
    assert module.option(predict, '--adapter') == 'trained/adapter'
    assert module.option(predict, '--queries').endswith('identity_val/transformed_queries.jsonl')
    assert module.option(tasks[1]['command'], '--base-predictions') == 'run/predictions.jsonl'
    assert 'run/predictions.jsonl.run.json' in pins
    assert 'trained/adapter/adapter_model.safetensors' in pins
    assert not any('target' in x or 'test_' in x or '--allow-test' == x for x in predict)
    assert tasks[0]['resume_command'][-2:] == ['--resume', 'latest']


def test_all_core_modes_and_no_ablation_expansion():
    tasks = []
    for mode in ['none', 'text', 'structure', 'condition', 'no_temperature', 'no_solvent']:
        tasks.append({'id': 'model_' + mode + '_val', 'command': ['python', 'wrapper', '--protocol', 'scaffold', '--',
                     'python', '-m', 'solvent_select.llm', 'predict', '--retrieval-mode', mode,
                     '--queries', 'val_queries.jsonl', '--output', 'predictions.jsonl']})
    result, _ = module.assemble_tasks({'tasks': tasks})
    assert len(result) == 8
    assert all('test' not in ' '.join(t['command']) for t in result)

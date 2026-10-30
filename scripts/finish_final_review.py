#!/usr/bin/env python3
"""Continue independent CPU reviews after the unchanged final runner finishes.

Never starts models, edits the frozen queue, tunes methods or writes manuscripts.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from solvent_select.views import sha256_file, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def stages():
    result = []
    for protocol in ['identity', 'scaffold']:
        for restored in [False, True]:
            suffix = '_reproduced' if restored else ''
            output = f'artifacts/audit/dualqa/{protocol}_final_contrast_v1{suffix}.json'
            result.append((protocol + '_contrast' + suffix,
                ['.venv-reproduce/bin/python' if restored else '.venv/bin/python', 'scripts/audit_paired_final_contrast.py',
                 '--comparison', f'artifacts/reports/frozen_dualqa/{protocol}_primary_condition_minus_structure.json',
                 '--grade-audit', 'artifacts/audit/dualqa/independent_final_scores_v1.json', '--output', output], output))
    for name, script, options in [
        ('method_figures', 'plot_registered_final.py', ['--assessment', 'artifacts/audit/dualqa/independent_final_scores_v1.json']),
        ('consistency_figures', 'summarize_final_consistency.py', ['--audit', 'artifacts/audit/dualqa/independent_final_consistency_v1.json']),
        ('resources', 'summarize_final_resources.py', []),
    ]:
        output = f'artifacts/reports/dualqa/final_{name}_restored_v1'
        result.append((name + '_rebuild', ['.venv-reproduce/bin/python', 'scripts/' + script, *options, '--output', output], output + '/manifest.json'))
    output = 'artifacts/audit/dualqa/final_figure_arithmetic_v1_reproduced.json'
    result.append(('figure_arithmetic_replay', ['.venv-reproduce/bin/python', 'scripts/audit_final_figure.py',
        '--figure', 'artifacts/reports/dualqa/final_figures_v1', '--grade-audit',
        'artifacts/audit/dualqa/independent_final_scores_v1.json', '--output', output], output))
    return result


def verify_rebuilt_outputs():
    pairs = []
    for protocol in ['identity', 'scaffold']:
        pairs.append((f'artifacts/audit/dualqa/{protocol}_final_contrast_v1.json',
                      f'artifacts/audit/dualqa/{protocol}_final_contrast_v1_reproduced.json'))
    pairs.append(('artifacts/audit/dualqa/final_figure_arithmetic_v1.json',
                  'artifacts/audit/dualqa/final_figure_arithmetic_v1_reproduced.json'))
    for original, restored, names in [
        ('final_figures_v1', 'final_method_figures_restored_v1', ['summary.csv', 'identity_final.png', 'scaffold_final.png']),
        ('final_consistency_figures_v1', 'final_consistency_figures_restored_v1', ['summary.csv', 'identity_consistency.png', 'scaffold_consistency.png']),
        ('final_resources_v1', 'final_resources_restored_v1', ['evidence_support.csv', 'evidence_support_per_solute.csv', 'latency_observations.csv']),
    ]:
        for name in names:
            pairs.append((f'artifacts/reports/dualqa/{original}/{name}', f'artifacts/reports/dualqa/{restored}/{name}'))
    checked = []
    for first, second in pairs:
        if Path(first).read_bytes() != Path(second).read_bytes():
            raise ValueError('Restored output differs; preserve and investigate: ' + second)
        checked.append({'original': first, 'restored': second, 'sha256': sha256_file(first)})
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'review.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'created_utc': now(), 'stages': {}}
        plan = Path('configs/final_queue_dualqa_v2.json')
        digest = sha256_file(plan)
        if state.get('plan_sha256', digest) != digest:
            raise ValueError('Final plan changed')
        state.update(pid=os.getpid(), plan_sha256=digest, status='WAITING_FOR_FINAL_QUEUE', updated_utc=now(),
                     code_sha256=sha256_file(__file__))
        write_json(state_path, state)
        try:
            deadline = time.monotonic() + 6 * 3600
            while True:
                queue = json.loads(Path('artifacts/final_queue_dualqa_v2/state.json').read_text())
                if queue['plan_sha256'] != digest:
                    raise ValueError('Final queue plan changed')
                if queue['status'] == 'completed':
                    if len(queue['tasks']) != 294 or any(v['status'] != 'completed' for v in queue['tasks'].values()):
                        raise ValueError('Queue completion receipts incomplete')
                    break
                pid = queue.get('queue_pid')
                proc = Path(f'/proc/{pid}/cmdline')
                if queue['status'] != 'running' or not proc.exists() or b'solvent_select.final_runner' not in proc.read_bytes().split(b'\0'):
                    raise RuntimeError('Final runner stopped; preserve outputs and recover it before review')
                if time.monotonic() > deadline:
                    raise TimeoutError('Review wait limit reached; rerun monitor after inspecting actual runner')
                time.sleep(30)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
            for name, command, artifact in stages():
                previous = state['stages'].get(name)
                if previous:
                    if previous['status'] != 'completed' or sha256_file(artifact) != previous['output_sha256']:
                        raise ValueError('Incomplete/changed review stage; inspect before resuming: ' + name)
                    continue
                if Path(artifact).exists():
                    raise FileExistsError('Unregistered review output: ' + artifact)
                record = {'status': 'running', 'command': command, 'started_utc': now()}
                state['stages'][name] = record
                state.update(status='RUNNING_INDEPENDENT_REVIEW', current_stage=name, updated_utc=now())
                write_json(state_path, state)
                with (output / (name + '.log')).open('a') as log:
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    state['child_pid'] = child.pid
                    write_json(state_path, state)
                    code = child.wait()
                record.update(exit_code=code, finished_utc=now())
                if code:
                    record['status'] = 'failed'
                    raise RuntimeError('Independent review stage failed: ' + name)
                record.update(status='completed', output_sha256=sha256_file(artifact))
                state['child_pid'] = None
                write_json(state_path, state)
            rebuilt = verify_rebuilt_outputs()
            criteria = {protocol: json.loads(Path(f'artifacts/audit/dualqa/{protocol}_final_contrast_v1.json').read_text())['hypothesis_joint_criteria_supported']
                        for protocol in ['identity', 'scaffold']}
            write_json(output / 'result.json', {'status': 'FINAL_STATISTICAL_AND_REPRODUCTION_CHECKS_COMPLETE',
                'registered_joint_criteria_supported': criteria, 'byte_identical_rebuilds': rebuilt,
                'pdfs': 'Rebuilt successfully; date metadata can differ, so PDF byte equality is not required.',
                'manuscripts': 'DEFERRED_BY_USER', 'remaining': ['Scientific contribution/claim review', 'Final experimental documentation and reproducibility handoff'],
                'publication_readiness': 'IN_PROGRESS', 'models_invoked': False})
            state.update(status='completed', current_stage=None, child_pid=None, updated_utc=now(), result='result.json')
        except Exception as error:
            state.update(status='interrupted_or_failed', error=repr(error), updated_utc=now())
            raise
        finally:
            write_json(state_path, state)


if __name__ == '__main__':
    main()

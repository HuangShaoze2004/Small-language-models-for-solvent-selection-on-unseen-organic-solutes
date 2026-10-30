#!/usr/bin/env python3
"""Run a pinned, validation-only experiment plan serially with safe stop/reuse.

The queue owns a separate scheduling lock, never the inner model's GPU lock.
SIGTERM forwards to the active wrapper; training saves its full checkpoint.
An interrupted task requires --resume-task ID. Exit zero alone is insufficient:
all registered outputs and completion status are checked before advancing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def verify_inputs(plan):
    for path, expected in plan['input_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Pinned input changed: ' + path)


def outputs(task):
    hashes = {}
    for item in task['outputs']:
        path = item['path']
        if 'status' in item:
            actual = json.loads(Path(path).read_text()).get('status')
            if actual != item['status']:
                raise ValueError(f'{task["id"]}: {path} status={actual}, expected={item["status"]}')
        hashes[path] = digest(path)
    return hashes


def command_for(task, previous, resume_task):
    if previous and previous['status'] != 'completed':
        if task['id'] != resume_task or not task.get('resume_command'):
            raise ValueError('Explicit --resume-task and resumable command required: ' + task['id'])
        return task['resume_command']
    return task['command']


def active_task_processes(task):
    """Find an orphaned wrapper/model by exact argv or its registered output path."""
    command = task['command']
    output_path = command[command.index('--output') + 1] if '--output' in command else None
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [part.decode() for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
            same = argv == command or (len(argv) == len(command) and argv[1:] == command[1:])
            if output_path and '--output' in argv:
                same = same or (argv[argv.index('--output') + 1] == output_path
                                and Path(os.readlink(proc / 'cwd')).resolve() == ROOT.resolve())
            if same:
                found.append(int(proc.name))
        except (OSError, UnicodeDecodeError, ValueError, IndexError):
            continue
    return found


def execute(plan_path, state_path, resume_task=None):
    os.chdir(ROOT)
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('partition') != 'val' or not plan.get('test_forbidden'):
        raise ValueError('This queue only runs registered validation work')
    ids = [t['id'] for t in plan['tasks']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate task IDs')
    verify_inputs(plan)
    plan_hash = digest(plan_path)
    state_path = Path(state_path)
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'plan': str(plan_path), 'plan_sha256': plan_hash, 'tasks': {}, 'started_utc': now()}
    if state['plan_sha256'] != plan_hash:
        raise ValueError('Queue plan changed; use a distinct plan/state')
    stop = {'signal': None, 'child': None}
    def handler(number, _frame):
        stop['signal'] = number
        child = stop['child']
        if child is not None and child.poll() is None:
            child.send_signal(number)
    old = {number: signal.signal(number, handler) for number in (signal.SIGTERM, signal.SIGINT)}
    lock_path = ROOT / 'artifacts/development_queue.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owned_lock = False
    try:
        with lock_path.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owned_lock = True
            for task in plan['tasks']:
                active = active_task_processes(task)
                if active:
                    raise RuntimeError(f'Existing task processes {active}: {task["id"]}; attach instead of restarting')
            lock.seek(0); lock.truncate()
            lock.write(json.dumps({'pid': os.getpid(), 'plan': str(plan_path), 'time': now()})); lock.flush()
            state.update(status='running', queue_pid=os.getpid(), child_pid=None, updated_utc=now())
            save(state_path, state)
            for task in plan['tasks']:
                verify_inputs(plan)
                previous = state['tasks'].get(task['id'])
                if previous and previous['status'] == 'completed':
                    if outputs(task) != previous['output_sha256']:
                        raise ValueError('Completed task artifacts changed: ' + task['id'])
                    continue
                if stop['signal']:
                    break
                command = command_for(task, previous, resume_task)
                # An existing output without our completion receipt is never silently overwritten.
                if not previous and any(Path(item['path']).exists() for item in task['outputs']):
                    raise FileExistsError('Unregistered existing output: ' + task['id'])
                task_log = state_path.parent / (task['id'] + '.log')
                task_log.parent.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                record = {'status': 'running', 'started_utc': now(), 'command': command, 'log': str(task_log),
                          'previous_attempt': previous}
                state['tasks'][task['id']] = record
                state.update(current_task=task['id'], updated_utc=now())
                save(state_path, state)
                with task_log.open('a') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    stop['child'] = child
                    state['child_pid'] = child.pid
                    record['pid'] = child.pid
                    save(state_path, state)
                    print(json.dumps({'task': task['id'], 'pid': child.pid, 'time': now()}), flush=True)
                    code = child.wait()
                stop['child'] = None
                record.update(exit_code=code, elapsed_seconds=time.monotonic() - started, finished_utc=now())
                state.update(child_pid=None, updated_utc=now())
                if code != 0 or stop['signal']:
                    record['status'] = 'interrupted' if stop['signal'] else 'failed'
                    state['status'] = record['status']
                    save(state_path, state)
                    return 130 if stop['signal'] else 1
                try:
                    verify_inputs(plan)
                    record['output_sha256'] = outputs(task)
                except Exception as error:
                    record.update(status='failed', error=repr(error))
                    state['status'] = 'failed'
                    save(state_path, state)
                    raise
                record['status'] = 'completed'
                save(state_path, state)
                print(json.dumps({'task': task['id'], 'status': 'completed', 'time': now()}), flush=True)
            state.update(status='interrupted' if stop['signal'] else 'completed', current_task=None,
                         child_pid=None, updated_utc=now(), finished_utc=now())
            save(state_path, state)
            return 130 if stop['signal'] else 0
    except Exception as error:
        if owned_lock:
            state.update(status='failed', error=repr(error), updated_utc=now())
            save(state_path, state)
        raise
    finally:
        for number, previous in old.items():
            signal.signal(number, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--resume-task')
    args = parser.parse_args()
    return execute(args.plan, args.state, args.resume_task)


if __name__ == '__main__':
    sys.exit(main())

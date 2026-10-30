#!/usr/bin/env python3
"""Persist experiment lifecycle and exact commands; never infer success from launch."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def atomic_json(path, value):
    tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(tmp,path)

def git_state():
    def call(*args):
        x=subprocess.run(['git',*args],cwd=ROOT,text=True,capture_output=True)
        return x.stdout.strip() if x.returncode==0 else None
    diff=call('diff','HEAD') or call('diff') or ''
    return {'commit':call('rev-parse','HEAD'),'dirty_diff_sha256':hashlib.sha256(diff.encode()).hexdigest(),
            'status':call('status','--short')}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True)
    p.add_argument('--kind',required=True)
    p.add_argument('--protocol',default='identity')
    p.add_argument('--seed',type=int)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--runs-root',default='artifacts/runs')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command: p.error('command required after --')
    runs_root=(ROOT/a.runs_root).resolve()
    if not runs_root.is_relative_to(ROOT) or Path(a.run_id).name!=a.run_id:
        p.error('Experiment output must be a named run within this project')
    folder=runs_root/a.run_id
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'experiment.json'
    if path.exists() and not a.resume:
        raise SystemExit('Experiment exists; inspect it and explicitly resume, do not duplicate')
    metadata={'run_id':a.run_id,'kind':a.kind,'protocol':a.protocol,'seed':a.seed,
              'runs_root':str(runs_root),
              'command':command,'cwd':str(ROOT),'code':git_state(),'status':'starting',
              'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'wrapper_pid':os.getpid(),'log':str(folder/'console.log')}
    started=time.monotonic()
    atomic_json(path,metadata)
    with open(folder/'console.log','a',buffering=1) as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        metadata.update(status='running',pid=process.pid)
        atomic_json(path,metadata)
        def forward(signum,frame):
            metadata['stop_signal']=signum
            atomic_json(path,metadata)
            process.send_signal(signum)
        signal.signal(signal.SIGTERM,forward)
        signal.signal(signal.SIGINT,forward)
        status=process.wait()
    metadata.update(status='completed' if status==0 and 'stop_signal' not in metadata else 'interrupted' if 'stop_signal' in metadata else 'failed',
                    exit_code=status,elapsed_seconds=time.monotonic()-started,
                    finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic_json(path,metadata)
    with open(ROOT/'artifacts/experiments.jsonl','a') as ledger:
        fcntl.flock(ledger.fileno(),fcntl.LOCK_EX)
        ledger.write(json.dumps(metadata)+'\n')
        ledger.flush()
        fcntl.flock(ledger.fileno(),fcntl.LOCK_UN)
    print(json.dumps(metadata),flush=True)
    raise SystemExit(status)

if __name__=='__main__': main()

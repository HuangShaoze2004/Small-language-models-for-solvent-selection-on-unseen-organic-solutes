"""Serial validation scheduling that attaches to an existing live queue.

It never runs a final test or creates a release. A failed/interrupted queue is
not restarted without its explicit task ID. Waiting checks are60seconds apart.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .freeze import queue_evidence
from .llm import write_json
from .study import DEFAULT_REGISTRY,load_registry
from .views import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def now():return datetime.now(timezone.utc).isoformat()


def live_queue(pid,plan):
    if not pid:return False
    proc=Path('/proc')/str(pid)
    try:
        argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        return ('scripts/run_development_queue.py' in argv and '--plan' in argv
                and argv[argv.index('--plan')+1]==str(plan)
                and Path(os.readlink(proc/'cwd')).resolve()==ROOT.resolve())
    except (OSError,UnicodeDecodeError,IndexError):return False


def run(registry_path=DEFAULT_REGISTRY,state_path='artifacts/pipeline_dualqa_v2/state.json',resume_task=None):
    os.chdir(ROOT)
    registry=load_registry(registry_path);registered_hash=sha256_file(registry_path)
    state_path=Path(state_path)
    state={'status':'starting','pid':os.getpid(),'created_utc':now(),'registry':str(registry_path),
           'registry_sha256':registered_hash,'stage':None,'queue_pid':None,'history':[]}
    control={'signal':None,'child':None,'attached':None,'plan':None}
    def stop(number,_frame):
        control['signal']=number
        try:
            if control['child'] is not None and control['child'].poll() is None:
                control['child'].send_signal(number)
            elif live_queue(control['attached'],control['plan']):
                os.kill(control['attached'],number)
        except ProcessLookupError:pass
    handlers={n:signal.signal(n,stop) for n in [signal.SIGTERM,signal.SIGINT]}
    owned=False
    try:
        with (ROOT/'artifacts/pipeline_dualqa.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);owned=True
            if state_path.exists():
                previous=json.loads(state_path.read_text())
                if previous.get('registry_sha256')!=registered_hash:raise ValueError('Registry differs from previous pipeline')
                state['previous_invocation']=previous
            write_json(state_path,state)
            stages=[*registry['queues'],{'plan':registry['consistency_plan'],'state':registry['consistency_state'],'role':'lm_consistency'}]
            for stage in stages:
                if control['signal']:break
                if sha256_file(registry_path)!=registered_hash:raise ValueError('Registered study changed')
                if stage['role']=='lm_consistency' and not Path(stage['plan']).exists():
                    command=['.venv/bin/python','scripts/prepare_consistency_plan.py','--registry',str(registry_path),'--output',stage['plan']]
                    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
                    state['consistency_preparation']={'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
                    write_json(state_path,state)
                    if result.returncode:raise RuntimeError('Consistency preparation failed; inspect pipeline state')
                while not control['signal']:
                    if sha256_file(registry_path)!=registered_hash:raise ValueError('Registered study changed')
                    queue=json.loads(Path(stage['state']).read_text()) if Path(stage['state']).exists() else {'status':'not_started'}
                    if queue.get('status')=='completed':
                        audit=queue_evidence(stage['plan'],stage['state'])
                        if audit['errors'] or audit['pending']:raise ValueError('Completed queue artifacts no longer verify')
                        state['history'].append({'stage':stage['role'],'verified_utc':now(),'receipt':audit})
                        state.update(status='stage_completed',stage=stage['role'],queue_pid=None,updated_utc=now())
                        write_json(state_path,state);break
                    pid=queue.get('queue_pid')
                    if live_queue(pid,stage['plan']):
                        control.update(attached=pid,plan=stage['plan'])
                        state.update(status='waiting_existing_live_queue',stage=stage['role'],queue_pid=pid,updated_utc=now())
                        write_json(state_path,state)
                        time.sleep(60)
                        continue
                    if queue.get('status') in ['failed','interrupted'] and queue.get('current_task')!=resume_task:
                        raise ValueError('Queue needs explicit --resume-task '+str(queue.get('current_task')))
                    if queue.get('status')=='running':
                        raise ValueError('Recorded queue handle missing; inspect outputs before explicitly resuming')
                    audit=queue_evidence(stage['plan'],stage['state'])
                    if audit['errors']:raise ValueError('Queue inputs or artifacts changed: '+str(audit['errors']))
                    command=['.venv/bin/python','scripts/run_development_queue.py','--plan',stage['plan'],'--state',stage['state']]
                    if queue.get('current_task'):command+=['--resume-task',queue['current_task']]
                    log=state_path.parent/(stage['role']+'.log')
                    state.update(status='running_owned_queue',stage=stage['role'],command=command,updated_utc=now())
                    with log.open('a') as handle:
                        child=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
                        control.update(child=child,attached=None,plan=stage['plan'])
                        state['queue_pid']=child.pid;write_json(state_path,state)
                        if control['signal']:child.send_signal(control['signal'])
                        code=child.wait()
                    control['child']=None
                    if code and not control['signal']:raise RuntimeError('Validation queue stopped with exit '+str(code))
            state.update(status='interrupted' if control['signal'] else 'VALIDATION_QUEUES_COMPLETED',finished_utc=now(),queue_pid=None)
            write_json(state_path,state)
            return 130 if control['signal'] else 0
    except Exception as error:
        if owned:
            state.update(status='failed',error=repr(error),finished_utc=now());write_json(state_path,state)
        raise
    finally:
        for n,handler in handlers.items():signal.signal(n,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',default=DEFAULT_REGISTRY)
    parser.add_argument('--state',default='artifacts/pipeline_dualqa_v2/state.json')
    parser.add_argument('--resume-task')
    args=parser.parse_args()
    raise SystemExit(run(args.registry,args.state,args.resume_task))


if __name__=='__main__':main()

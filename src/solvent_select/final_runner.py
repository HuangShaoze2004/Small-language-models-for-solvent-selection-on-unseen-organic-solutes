"""Serial, resumable execution of a fully frozen final plan.

No model or outcome access is allowed before release verification. The global
queue lock is shared with development; model processes own the inner GPU lock.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .final_plan import assemble, reusable_zero_shot
from .llm import read_jsonl, write_json
from .release_guard import verify_release
from .views import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_plan(plan):
    if plan.get('status')!='FROZEN_FINAL_EXECUTION_PLAN' or plan.get('partition')!='test':
        raise ValueError('Only a frozen final execution plan is authorized')
    protocols=set()
    releases={}
    for path,expected in plan['release_sha256'].items():
        if sha256_file(path)!=expected:
            raise ValueError('Release changed after final plan registration')
        protocol=verify_release(path)['protocol']
        protocols.add(protocol)
        releases[protocol]={**json.loads(Path(path).read_text()),'_path':str(path)}
    if protocols!={'identity','scaffold'}:
        raise ValueError('Both verified protocol releases are required')
    ids=[t['id'] for t in plan['tasks']]
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('Empty or duplicate final task list')
    if plan.get('code_sha256')!=sha256_file(Path(__file__).with_name('final_plan.py')):
        raise ValueError('Final task-definition code changed')
    if plan['tasks']!=assemble(releases):
        raise ValueError('Final task graph differs from the registered frozen definitions')


def output_receipt(task):
    receipt={}
    for item in task['outputs']:
        path=Path(item['path'])
        if not path.is_file():
            raise ValueError('Missing task output: '+str(path))
        if 'status' in item and json.loads(path.read_text()).get('status')!=item['status']:
            raise ValueError('Child output has not reached required terminal status: '+str(path))
        receipt[str(path)]=sha256_file(path)
    if task.get('property_prediction'):
        import numpy as np
        import pandas as pd
        spec=task['property_prediction']
        manifest=pd.read_csv(spec['manifest'],dtype={'solute_id':str})
        ids=manifest.loc[manifest.split=='test','solute_id'].tolist()
        expected=pd.read_parquet(spec['measurements'],columns=['record_id'],filters=[('solute_id','in',ids)])
        predicted=pd.read_parquet(spec['path'],columns=['record_id','split','pred_logx'])
        if (predicted.record_id.duplicated().any() or set(predicted.record_id)!=set(expected.record_id)
                or set(predicted.split)!={'test'} or not np.isfinite(predicted.pred_logx).all()):
            raise ValueError('Property prediction output is not the complete registered partition')
    if task.get('query_source'):
        expected={q['query_id'] for q in read_jsonl(task['query_source'])}
        actual={q['query_id'] for q in read_jsonl(task['prediction_path'])}
        if expected!=actual:
            raise ValueError('Prediction IDs do not match the complete registered cohort')
        metadata=json.loads(Path(task['prediction_path']+'.run.json').read_text())
        if metadata.get('fingerprint_sha256')!=task['prediction_specification']['sha256']:
            raise ValueError('Completed prediction has the wrong configuration fingerprint')
    return receipt


def repair_unterminated_jsonl(path,archive):
    """Recover only an unfinished final write; preserve its original bytes."""
    path=Path(path)
    if not path.exists():
        return None
    raw=path.read_bytes()
    if not raw or raw.endswith(b'\n'):
        return None
    split=raw.rfind(b'\n')+1
    prefix,last=raw[:split],raw[split:]
    # Earlier records must already be intact; never discard arbitrary failures.
    for line in prefix.splitlines():
        if line.strip():json.loads(line)
    try:
        json.loads(last)
        repaired=raw+b'\n'
        action='append_missing_newline_to_complete_final_record'
    except (ValueError,UnicodeDecodeError):
        repaired=prefix
        action='quarantine_incomplete_final_record'
    archive=Path(archive)
    archive.mkdir(parents=True,exist_ok=True)
    backup=archive/(path.name+'.before_transport_repair')
    if backup.exists():raise FileExistsError('Never overwrite recovery evidence')
    backup.write_bytes(raw)
    path.write_bytes(repaired)
    evidence={'action':action,'original_sha256':sha256_file(backup),'repaired_sha256':sha256_file(path),
              'original_bytes':len(raw),'repaired_bytes':len(repaired),'backup':str(backup)}
    write_json(archive/'jsonl_transport_repair.json',evidence)
    return evidence


def reuse_zero_shot(task):
    if not reusable_zero_shot(task['source_specification'],task['prediction_specification']):
        raise ValueError('Zero-shot reuse lacks identical input/model/decoding proof')
    source=Path(task['source_prediction'])
    metadata_path=Path(str(source)+'.run.json')
    metadata=json.loads(metadata_path.read_text())
    if metadata.get('status')!='completed' or metadata.get('fingerprint_sha256')!=task['source_specification']['sha256']:
        raise ValueError('Source zero-shot inference is not complete with its registered fingerprint')
    if {r['query_id'] for r in read_jsonl(source)}!={q['query_id'] for q in read_jsonl(task['query_source'])}:
        raise ValueError('Zero-shot reuse question identities changed')
    destination=Path(task['prediction_path'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
    fingerprint=task['prediction_specification']
    copied={**metadata,'fingerprint':fingerprint['fingerprint'],'fingerprint_sha256':fingerprint['sha256'],
            'shared_inference_source':str(source),'shared_inference_source_sha256':sha256_file(source),
            'reuse_recorded_utc':now(),'reuse_scope':'Same public query bytes, unadapted weights, implementation and decoding; only the split audit hash differs. Timing fields describe the shared original inference, not an additional model call.'}
    write_json(str(destination)+'.run.json',copied)


def existing_processes(task):
    command=task.get('command',[])
    output=command[command.index('--output')+1] if '--output' in command else None
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():
            continue
        try:
            argv=[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
            if output and '--output' in argv and argv[argv.index('--output')+1]==output:
                if Path(os.readlink(proc/'cwd')).resolve()==ROOT.resolve():
                    found.append(int(proc.name))
        except (OSError,UnicodeDecodeError,ValueError,IndexError):
            continue
    return found


def execute(plan_path,state_path,resume_task=None):
    os.chdir(ROOT)
    plan=json.loads(Path(plan_path).read_text())
    verify_plan(plan)
    state_path=Path(state_path)
    digest=sha256_file(plan_path)
    state=json.loads(state_path.read_text()) if state_path.exists() else {
        'plan':str(plan_path),'plan_sha256':digest,'tasks':{},'created_utc':now()}
    if state['plan_sha256']!=digest:
        raise ValueError('Final plan changed; an explicit erratum and affected-method re-evaluation are required')
    stop={'signal':None,'child':None}
    tasks_by_id={task['id']:task for task in plan['tasks']}
    def interrupted(number,_frame):
        stop['signal']=number
        if stop['child'] is not None and stop['child'].poll() is None:
            stop['child'].send_signal(number)
    handlers={number:signal.signal(number,interrupted) for number in [signal.SIGINT,signal.SIGTERM]}
    lock_path=ROOT/'artifacts/development_queue.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    owned_lock=False
    try:
        with lock_path.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            owned_lock=True
            for task in plan['tasks']:
                active=existing_processes(task)
                if active:
                    raise RuntimeError('Existing task process; attach instead of restarting: '+str(active))
            state.update(status='running',queue_pid=os.getpid(),updated_utc=now(),current_task=None)
            write_json(state_path,state)
            for task in plan['tasks']:
                verify_plan(plan)
                for dependency in task.get('depends_on',[]):
                    saved=state['tasks'].get(dependency,{})
                    if saved.get('status')!='completed' or output_receipt(tasks_by_id[dependency])!=saved.get('output_sha256'):
                        raise ValueError('Task dependency is incomplete or changed: '+dependency)
                previous=state['tasks'].get(task['id'])
                if previous and previous['status']=='completed':
                    if output_receipt(task)!=previous['output_sha256']:
                        raise ValueError('A completed final output changed; preserve it and register an erratum')
                    continue
                if stop['signal']:
                    break
                if previous and (resume_task!=task['id'] or task.get('action')=='command' and not task.get('resume_command')):
                    raise ValueError('Explicit --resume-task ID is required for the interrupted/failed task')
                if previous and any('status' in item for item in task['outputs']):
                    # A signal may arrive after persistence but before the queue
                    # records success. Adopt only complete, validated artifacts.
                    try:receipt=output_receipt(task)
                    except (ValueError,OSError,KeyError):pass
                    else:
                        previous.update(status='completed',output_sha256=receipt,
                                        adopted_after_interruption=True,adopted_utc=now())
                        write_json(state_path,state)
                        continue
                if not previous and any(Path(item['path']).exists() for item in task['outputs']):
                    raise FileExistsError('Unregistered final output exists: '+task['id'])
                record={'status':'running','started_utc':now(),'previous_attempt':previous,
                        'command':task.get('resume_command') if previous else task.get('command'),
                        'action':task['action']}
                state['tasks'][task['id']]=record
                state.update(current_task=task['id'],child_pid=None,updated_utc=now())
                write_json(state_path,state)
                started=time.monotonic()
                if previous:
                    archive=state_path.parent/'recovery'/task['id']/str(time.time_ns())
                    archive.mkdir(parents=True)
                    write_json(archive/'previous_queue_record.json',previous)
                    for item in task['outputs']:
                        source=Path(item['path'])
                        if source.exists():
                            destination=archive/(sha256_file(source)[:12]+'_'+source.name)
                            shutil.copy2(source,destination)
                    if task.get('prediction_path'):
                        record['transport_repair']=repair_unterminated_jsonl(task['prediction_path'],archive)
                    record['recovery_archive']=str(archive)
                    write_json(state_path,state)
                if task['action']=='reuse_zero_shot':
                    reuse_zero_shot(task)
                    code=0
                elif task['action']=='command':
                    log_path=state_path.parent/(task['id']+'.log')
                    env={**os.environ,**task.get('env',{})}
                    with log_path.open('a') as log:
                        child=subprocess.Popen(record['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                        stop['child']=child
                        if stop['signal']:child.send_signal(stop['signal'])
                        record.update(pid=child.pid,log=str(log_path))
                        state['child_pid']=child.pid
                        write_json(state_path,state)
                        code=child.wait()
                    stop['child']=None
                else:
                    raise ValueError('Unregistered task action')
                record.update(exit_code=code,elapsed_seconds=time.monotonic()-started,finished_utc=now())
                state.update(child_pid=None,updated_utc=now())
                if code!=0 or stop['signal']:
                    record['status']='interrupted' if stop['signal'] else 'failed'
                    state['status']=record['status']
                    write_json(state_path,state)
                    return 130 if stop['signal'] else 1
                record['output_sha256']=output_receipt(task)
                record['status']='completed'
                write_json(state_path,state)
            if not stop['signal']:
                for task in plan['tasks']:
                    saved=state['tasks'].get(task['id'],{})
                    if saved.get('status')!='completed' or output_receipt(task)!=saved.get('output_sha256'):
                        raise ValueError('Final completion audit found incomplete or modified outputs')
            state.update(status='interrupted' if stop['signal'] else 'completed',updated_utc=now(),current_task=None)
            write_json(state_path,state)
            return 130 if stop['signal'] else 0
    except Exception as error:
        if owned_lock:
            current=state.get('current_task')
            if current and state['tasks'].get(current,{}).get('status')=='running':
                state['tasks'][current].update(status='failed',error=repr(error),finished_utc=now())
            state.update(status='failed',error=repr(error),updated_utc=now())
            write_json(state_path,state)
        raise
    finally:
        for number,handler in handlers.items():
            signal.signal(number,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default='configs/final_queue_dualqa_v2.json')
    parser.add_argument('--state',default='artifacts/final_queue_dualqa_v2/state.json')
    parser.add_argument('--resume-task')
    args=parser.parse_args()
    raise SystemExit(execute(args.plan,args.state,args.resume_task))


if __name__=='__main__':
    main()

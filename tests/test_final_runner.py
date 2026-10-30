"""Transport/lifecycle fixtures only, outside all scientific data directories."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

import solvent_select.final_runner as runner
from solvent_select.views import sha256_file


def fixture_plan(tmp_path,monkeypatch,code):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.chdir(tmp_path)
    # Exercise the executor only; independent tests exercise release rejection.
    # No real scientific release is fabricated or written by these fixtures.
    monkeypatch.setattr(runner,'verify_plan',lambda plan:None)
    output=tmp_path/'output.json'
    plan=tmp_path/'plan.json';state=tmp_path/'state.json'
    task={'id':'UNIT_fixture','protocol':'identity','action':'command',
          'command':[sys.executable,'-c',code], 'resume_command':[sys.executable,'-c',code],
          'outputs':[{'path':str(output),'status':'completed'}]}
    plan.write_text(json.dumps({'tasks':[task]}))
    return plan,state,output


def test_successful_child_is_verified_then_reused_and_modified_outputs_refused(tmp_path,monkeypatch):
    code="from pathlib import Path; Path('output.json').write_text('{\"status\":\"completed\"}'); p=Path('calls'); p.write_text(str(int(p.read_text())+1) if p.exists() else '1')"
    plan,state,output=fixture_plan(tmp_path,monkeypatch,code)
    assert runner.execute(plan,state)==0
    assert runner.execute(plan,state)==0
    assert (tmp_path/'calls').read_text()=='1'
    output.write_text('{"status":"completed","changed":true}')
    with pytest.raises(ValueError,match='completed final output changed'):
        runner.execute(plan,state)
    assert (tmp_path/'calls').read_text()=='1'


def test_exit_zero_without_child_completion_is_failed(tmp_path,monkeypatch):
    plan,state,_=fixture_plan(tmp_path,monkeypatch,"from pathlib import Path; Path('output.json').write_text('{\"status\":\"preparing\"}')")
    with pytest.raises(ValueError,match='terminal status'):
        runner.execute(plan,state)
    saved=json.loads(state.read_text())
    assert saved['status']=='failed' and saved['tasks']['UNIT_fixture']['status']=='failed'


def test_interruption_after_persistence_adopts_complete_output_without_recomputation(tmp_path,monkeypatch):
    plan,state,output=fixture_plan(tmp_path,monkeypatch,"raise AssertionError('must not run')")
    output.write_text('{"status":"completed"}')
    state.write_text(json.dumps({'plan_sha256':sha256_file(plan),'tasks':{'UNIT_fixture':{'status':'interrupted'}}}))
    with pytest.raises(ValueError,match='Explicit --resume-task'):
        runner.execute(plan,state)
    assert runner.execute(plan,state,resume_task='UNIT_fixture')==0
    assert json.loads(state.read_text())['tasks']['UNIT_fixture']['adopted_after_interruption']


def test_busy_global_lock_does_not_overwrite_existing_state(tmp_path,monkeypatch):
    plan,state,_=fixture_plan(tmp_path,monkeypatch,"raise AssertionError('must not run')")
    initial={'plan_sha256':sha256_file(plan),'tasks':{},'status':'running'}
    state.write_text(json.dumps(initial));before=state.read_bytes()
    lock_path=tmp_path/'artifacts/development_queue.lock';lock_path.parent.mkdir()
    with lock_path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runner.execute(plan,state)
    assert state.read_bytes()==before


def test_unfinished_final_json_record_is_archived_and_only_it_removed(tmp_path):
    path=tmp_path/'predictions.jsonl'
    original=b'{"query_id":"a"}\n{"query_id":"unfinished'
    path.write_bytes(original)
    record=runner.repair_unterminated_jsonl(path,tmp_path/'archive')
    assert path.read_bytes()==b'{"query_id":"a"}\n'
    assert Path(record['backup']).read_bytes()==original
    assert record['action']=='quarantine_incomplete_final_record'


def test_complete_final_record_without_newline_is_preserved(tmp_path):
    path=tmp_path/'predictions.jsonl';path.write_text('{"query_id":"a"}')
    result=runner.repair_unterminated_jsonl(path,tmp_path/'archive')
    assert path.read_text()=='{"query_id":"a"}\n'
    assert result['action']=='append_missing_newline_to_complete_final_record'


def test_corrupt_earlier_record_cannot_be_silently_removed(tmp_path):
    path=tmp_path/'predictions.jsonl';path.write_bytes(b'broken\n{"partial"')
    before=path.read_bytes()
    with pytest.raises(ValueError):
        runner.repair_unterminated_jsonl(path,tmp_path/'archive')
    assert path.read_bytes()==before


def test_property_receipt_checks_entire_partition_without_loading_experimental_labels(tmp_path,monkeypatch):
    import pandas as pd
    manifest=tmp_path/'split.csv';manifest.write_text('solute_id,split\ns1,test\ns2,train\n')
    measurements=tmp_path/'measurements.parquet'
    pd.DataFrame({'record_id':['r1','r2','r3'],'solute_id':['s1','s1','s2'],
                  'logx':[-1.,-2.,-3.]}).to_parquet(measurements)
    prediction=tmp_path/'prediction.parquet'
    pd.DataFrame({'record_id':['r1','r2'],'split':['test','test'],'pred_logx':[-.5,-1.5]}).to_parquet(prediction)
    original=pd.read_parquet
    def projected(path,*args,**kwargs):
        assert kwargs.get('columns') is not None
        assert 'logx' not in kwargs['columns']
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',projected)
    task={'outputs':[{'path':str(prediction)}],'property_prediction':{
        'path':str(prediction),'measurements':str(measurements),'manifest':str(manifest)}}
    runner.output_receipt(task)
    pd.DataFrame({'record_id':['r1'],'split':['test'],'pred_logx':[-.5]}).to_parquet(prediction)
    with pytest.raises(ValueError,match='complete registered partition'):
        runner.output_receipt(task)


def test_orphan_output_process_is_found_before_restarting(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    output=str(tmp_path/'predictions.jsonl')
    command=[sys.executable,'-c','import time; time.sleep(30)','--output',output]
    child=subprocess.Popen(command,cwd=tmp_path)
    try:
        assert child.pid in runner.existing_processes({'command':command})
    finally:
        child.terminate();child.wait(timeout=5)


def test_sigterm_forwarding_stops_next_task_even_if_child_exits_zero(tmp_path,monkeypatch):
    code="import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,lambda *args:exit(0)); Path('ready').write_text('yes'); time.sleep(30)"
    plan,state,_=fixture_plan(tmp_path,monkeypatch,code)
    timed_out=[]
    def signal_when_ready():
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if (tmp_path/'ready').exists():
                os.kill(os.getpid(),signal.SIGTERM);return
            time.sleep(.01)
        timed_out.append(True)
    thread=threading.Thread(target=signal_when_ready,daemon=True);thread.start()
    assert runner.execute(plan,state)==130
    thread.join(timeout=6)
    assert not timed_out
    saved=json.loads(state.read_text())
    assert saved['status']=='interrupted'
    assert saved['tasks']['UNIT_fixture']['exit_code']==0

"""Durable, locked, max-two queue; all capacity preflights finish before training."""
import os,sys,json,time,hashlib,subprocess,fcntl,traceback,csv
from pathlib import Path
R=Path('/root/autodl-tmp/moe_reproduction'); A=R/'artifacts'
M=A/'irstd_experts_fixed005_v1_matrix.json'; WORKER=A/'run_irstd_experts_fixed005_v1.py'
OUT=R/'runs/irstd_experts_fixed005_300'; STATE=A/'irstd_experts_queue_v1_status.json'
RESULT=A/'irstd_experts_results_v1.json'; CONTRACT=A/'irstd_experts_contract_v1.json'
EVENT=R/'logs/irstd_experts_queue_v1_events.jsonl'
def atomic(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def event(kind,**kw):
    with EVENT.open('a') as f:f.write(json.dumps(dict(event=kind,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))+'\n')
def verify():
    for rel,h in json.loads(CONTRACT.read_text())['sha256'].items():
        assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==h,('Contract changed',rel)
def workers():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0'); cmd=b' '.join(args).decode()
            if not args or 'python' not in args[0].decode().lower():continue
            if (p/'status').read_text().split('State:')[1].lstrip()[0]=='Z':continue
            if ('--mode run' in cmd or '--mode preflight' in cmd or '/train.py' in cmd or '/train_mask.py' in cmd or '/train_spar.py' in cmd):found.append((int(p.name),cmd))
        except (OSError,UnicodeDecodeError,IndexError):pass
    return found
def gpu():
    row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    return tuple(float(x) for x in row)
def gpu_pids():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    return {int(x.strip()) for x in out.splitlines() if x.strip().isdigit()}
def result_rows(jobs):
    rows=[]
    for j in jobs:
        p=OUT/j['run_id']; st=p/'status.json'; bm=p/'best_metrics.json'
        if not st.exists() or not bm.exists():continue
        s=json.loads(st.read_text());b=json.loads(bm.read_text())
        rows.append(dict(run_id=j['run_id'],dataset='IRSTD-1K',input_size=256,protocol='irstd_constant005_300',status=s['status'],best_epoch_one_based=b['epoch']+1,metrics={k:b[k] for k in ['IoU','nIoU','Pd','Fa']},nuaa_reference=j['nuaa_reference']))
    atomic(RESULT,rows)
    flat=[dict(run_id=x['run_id'],status=x['status'],best_epoch_one_based=x['best_epoch_one_based'],**x['metrics']) for x in rows]
    if flat:
        t=RESULT.with_suffix('.csv.tmp')
        with t.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
        t.replace(RESULT.with_suffix('.csv'))
def main():
    lock=open(A/'irstd_experts_queue_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(); jobs=json.loads(M.read_text())['runs']; assert len(jobs)==4
    assert not workers(),('Existing workers; refusing duplicate queue',workers())
    assert not gpu_pids(),('External GPU processes',gpu_pids())
    # An existing unfinished run needs explicit audited adoption/resume, never silent duplication.
    for j in jobs:
        p=OUT/j['run_id']
        assert not p.exists(),('Run output exists; inspect before any resume',str(p))
    atomic(RESULT,[]); event('START_CONTROLLER',pid=os.getpid())
    for j in jobs:
        verify();assert not workers() and not gpu_pids()
        atomic(STATE,dict(phase='preflight',controller_pid=os.getpid(),preflight=j['run_id'],active={},pending=[x['run_id'] for x in jobs],completed=[],failed=[],updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        path=A/'irstd_experts_preflight_v1'/(j['run_id']+'.json');path.parent.mkdir(exist_ok=True)
        with (path.with_suffix('.console.log')).open('a') as f:
            rc=subprocess.call([sys.executable,str(WORKER),'--mode','preflight','--matrix',str(M),'--job',j['run_id']],stdout=f,stderr=f,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        assert rc==0,('Capacity preflight failed',j['run_id'],rc)
        pf=json.loads(path.read_text());assert pf['status']=='passed' and pf['configuration']==j['configuration']
        event('PREFLIGHT_PASSED',run_id=j['run_id'],reserved_gib=pf['reserved_gib'],train_step_seconds=pf['train_step_seconds'])
    event('ALL_PREFLIGHTS_PASSED')
    active={}; pending=jobs.copy();done=[];failed=[]; audits=[]
    while pending or active:
        for name,(p,fh,j,need) in list(active.items()):
            code=p.poll()
            if code is not None:
                fh.close();st=json.loads((OUT/name/'status.json').read_text())
                if code==0 and st['status']=='complete' and st['epochs']==300:done.append(name);audits.append(name);event('COMPLETE',run_id=name)
                else:failed.append(name);event('FAILED',run_id=name,exit_code=code,status=st)
                del active[name]
        result_rows(jobs)
        # Completion auditing uses CPU and does not change any trained artifact.
        for name in list(audits):
            with (A/'irstd_completed_audits_v1'/f'{name}.audit_console.log').open('a') as f:
                rc=subprocess.call([sys.executable,str(A/'audit_irstd_experts_completed_v1.py'),name],stdout=f,stderr=f)
            event('AUDIT_PASSED' if rc==0 else 'AUDIT_FAILED',run_id=name)
            if rc!=0:failed.append(name+':audit')
            audits.remove(name)
        verify()
        actual={pid for pid,cmd in workers()}; expected={v[0].pid for v in active.values() if v[0].poll() is None}
        assert actual==expected,('Unexpected/orphan workers',actual,expected)
        total,free,util=gpu(); external=gpu_pids()-expected
        budget=total/1024-1.5; reserved=sum(v[3] for v in active.values())
        for j in list(pending):
            if failed or external or len(active)>=2:break
            pf=json.loads((A/'irstd_experts_preflight_v1'/(j['run_id']+'.json')).read_text())
            need=pf['reserved_gib']*1.1+.5
            if need>budget:raise RuntimeError(('Job exceeds GPU capacity',j['run_id'],need,budget))
            if reserved+need>budget or need+.5>free/1024:continue
            name=j['run_id']; pth=OUT/name;pth.mkdir(parents=True,exist_ok=False); fh=(pth/'console.log').open('a',buffering=1)
            p=subprocess.Popen([sys.executable,str(WORKER),'--mode','run','--matrix',str(M),'--job',name],stdout=fh,stderr=fh,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
            active[name]=(p,fh,j,need);pending.remove(j);reserved+=need;free-=need*1024
            event('START',run_id=name,pid=p.pid,reservation_gib=need)
        phase='failed' if failed else ('complete' if not pending and not active else 'training')
        atomic(STATE,dict(phase=phase,controller_pid=os.getpid(),active={k:dict(pid=v[0].pid,reservation_gib=v[3]) for k,v in active.items()},pending=[j['run_id'] for j in pending],completed=done,failed=failed,gpu_free_mib=free,gpu_utilization=util,external_gpu_pids=list(external),updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        if failed and not active:break
        if pending or active:time.sleep(10)
    event('QUEUE_FINISHED',completed=done,failed=failed)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        event('CONTROLLER_ERROR',error=repr(e),traceback=traceback.format_exc())
        state=json.loads(STATE.read_text()) if STATE.exists() else {}
        state.update(phase='failed',controller_error=repr(e),updated=time.strftime('%Y-%m-%d %H:%M:%S'));atomic(STATE,state)
        raise

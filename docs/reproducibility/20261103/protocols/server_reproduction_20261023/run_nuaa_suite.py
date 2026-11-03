"""NUAA-only explicit-config runner, preflight, metrics, and durable GPU queue."""
import argparse,os,sys,json,time,random,hashlib,signal,subprocess,traceback,csv
from pathlib import Path
ROOT=Path('/root/autodl-tmp/moe_reproduction')
sys.path[:0]=[str(ROOT/'artifacts'),str(ROOT/'code/experiment_runtime')]
import numpy as np
import torch
from skimage import measure
from model.utils import TrainSetLoader,TestSetLoader,weights_init_xavier
from model.load_param_data import load_dataset
from model.loss import spar_iou_loss
from ablation_variants import build_model,auxiliary_loss
torch.set_num_threads(4)

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def save_checkpoint(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def source_hashes():
    paths=list((ROOT/'code/experiment_runtime').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'artifacts/ablation_variants.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def loaders(c):
    assert c['dataset']=='NUAA-SIRST'
    train,_,test=load_dataset(str(ROOT/'data'),c['dataset'],'50_50');d=str(ROOT/'data'/c['dataset'])
    kw=dict(base_size=c['base_size'],crop_size=c['crop_size'],suffix='.png',dataset_name=c['dataset'])
    a=TrainSetLoader(d,img_id=train,**kw);b=TestSetLoader(d,img_id=test,**kw)
    return torch.utils.data.DataLoader(a,batch_size=c['train_batch_size'],shuffle=True,num_workers=0,drop_last=True),torch.utils.data.DataLoader(b,batch_size=c['test_batch_size'],shuffle=False,num_workers=0)

class Metrics:
    """IoU/nIoU per image; greedy centroid Pd, connected components with 8-connectivity."""
    def __init__(self):self.inter=self.union=self.n=self.niou=self.detected=self.targets=self.fa=self.legacy_fa=self.pixels=0
    def update(self,pred,truth):
        for p,g in zip(pred[:,0],truth[:,0]):
            inter=int((p&g).sum());union=int((p|g).sum());self.inter+=inter;self.union+=union;self.niou+=inter/max(union,1e-10);self.n+=1;self.pixels+=p.size
            regions=measure.regionprops(measure.label(p,connectivity=2));gt=measure.regionprops(measure.label(g,connectivity=2));self.targets+=len(gt);used=set();areas=[]
            for target in gt:
                for region in regions:
                    if region.label not in used and np.linalg.norm(np.asarray(region.centroid)-np.asarray(target.centroid))<3:
                        used.add(region.label);areas.append(region.area);self.detected+=1;break
            self.fa+=sum(r.area for r in regions if r.label not in used)
            self.legacy_fa+=sum(r.area for r in regions if r.area not in areas)
    def get(self):
        return dict(IoU=float(self.inter/max(self.union,1e-10)),nIoU=float(self.niou/max(self.n,1)),Pd=float(self.detected/max(self.targets,1)),Fa=float(self.fa/max(self.pixels,1)),Fa_legacy=float(self.legacy_fa/max(self.pixels,1)),images=self.n,targets=self.targets,threshold=.1)

def evaluate(net,dl):
    net.eval();m=Metrics();loss_sum=count=0
    with torch.no_grad():
        for x,y in dl:
            x=x.cuda();y=y.cuda();p=net(x);loss_sum+=float(spar_iou_loss(p,y))*len(x);count+=len(x)
            m.update((p[-1].sigmoid()>.1).cpu().numpy(),(y>.1).cpu().numpy())
    result=m.get();result['test_loss']=loss_sum/max(count,1);return result

def preflight(job):
    c=job['configuration'];seed_all(c['seed']);net,notes=build_model(c);net=net.cuda();net.apply(weights_init_xavier)
    opt=torch.optim.Adagrad((p for p in net.parameters() if p.requires_grad),lr=c['lr']);train,test=loaders(c);x,y=next(iter(train));x=x.cuda();y=y.cuda()
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    for iteration in range(4):
        if iteration==2:torch.cuda.synchronize();train_start=time.monotonic()
        net.train();opt.zero_grad();p=net(x);main=spar_iou_loss(p,y);aux=auxiliary_loss(net);loss=main+(c['aux_loss_weight']*aux if aux is not None else 0)
        assert torch.isfinite(loss),'Nonfinite preflight loss';loss.backward();opt.step()
    torch.cuda.synchronize();train_seconds=(time.monotonic()-train_start)/2;reserved=torch.cuda.max_memory_reserved()/1024**3
    net.eval()
    with torch.no_grad():
        p=net(x)[-1];changed=x.clone();changed[1:]=torch.randn_like(changed[1:]);q=net(changed)[-1]
        assert torch.isfinite(p).all() and p.shape==y.shape
        assert torch.allclose(p[0],q[0],atol=1e-4,rtol=1e-4),'Cross-sample influence during evaluation'
        pred=(p.sigmoid()>.1).cpu().numpy();truth=(y>.1).cpu().numpy();m=Metrics();m.update(pred,truth)
    # Regression check for the known equal-area false-alarm bug.
    a=np.zeros((1,1,40,40),dtype=bool);b=np.zeros_like(a);a[0,0,10,10]=a[0,0,30,30]=True;b[0,0,10,10]=True
    check=Metrics();check.update(a,b);assert check.fa==1 and check.legacy_fa==0
    return dict(status='passed',reserved_gib=reserved,train_step_seconds=train_seconds,seconds=time.monotonic()-start,total_parameters=sum(p.numel() for p in net.parameters()),trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),module_definitions=notes,eval_sample_independence=True,metric_equal_area_regression=True)

def run(job):
    c=job['configuration'];out=ROOT/'runs/nuaa_600_300'/job['run_id'];out.mkdir(parents=True,exist_ok=True);status=out/'status.json'
    cfg_hash=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest();seed_all(c['seed'])
    net,notes=build_model(c);net=net.cuda();net.apply(weights_init_xavier)
    opt=torch.optim.Adagrad((p for p in net.parameters() if p.requires_grad),lr=c['lr'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c['epochs'],eta_min=c['min_lr']);train,test=loaders(c)
    hashes=source_hashes();manifest=dict(job=job,config_sha256=cfg_hash,source_sha256=hashes,torch=torch.__version__,cuda=torch.version.cuda,definitions=notes,metrics='threshold 0.1; same checkpoint; corrected region-ID Fa with legacy-area Fa retained',started=time.strftime('%Y-%m-%d %H:%M:%S'))
    for split in ['train','test']:
        path=ROOT/'data/NUAA-SIRST/50_50'/(split+'.txt');manifest[split+'_ids_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    start_epoch=0;best=-1.;best_epoch=-1
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert state['config_sha256']==cfg_hash and state['source_sha256']==hashes,'Config/source changed; refusing incompatible resume'
        net.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        start_epoch=state['epoch']+1;best=state['best_iou'];best_epoch=state['best_epoch']
        random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy']);torch.set_rng_state(state['rng_torch']);torch.cuda.set_rng_state_all(state['rng_cuda'])
    else:atomic_json(out/'manifest.json',manifest)
    stop=[False]
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    durations=[]
    for epoch in range(start_epoch,c['epochs']):
        t=time.monotonic();lr=opt.param_groups[0]['lr'];net.train();loss_sum=main_sum=aux_sum=count=0
        atomic_json(status,dict(status='running',phase='train',epoch=epoch,epochs=c['epochs'],best_iou=best,best_epoch=best_epoch,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        for x,y in train:
            x=x.cuda();y=y.cuda();opt.zero_grad();p=net(x);main=spar_iou_loss(p,y);aux=auxiliary_loss(net);loss=main+(c['aux_loss_weight']*aux if aux is not None else 0)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();opt.step();n=len(x);count+=n;loss_sum+=float(loss.detach())*n;main_sum+=float(main.detach())*n;aux_sum+=float(aux.detach())*n if aux is not None else 0
        metrics=evaluate(net,test);sched.step()
        if metrics['IoU']>best:
            best=metrics['IoU'];best_epoch=epoch
            save_checkpoint(out/'best.pt',dict(model=net.state_dict(),configuration=c,epoch=epoch,metrics=metrics,source_sha256=hashes))
            atomic_json(out/'best_metrics.json',dict(epoch=epoch,configuration=c,**metrics))
        save_checkpoint(out/'last.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,best_iou=best,best_epoch=best_epoch,config_sha256=cfg_hash,source_sha256=hashes,rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()))
        seconds=time.monotonic()-t;durations.append(seconds)
        row=dict(epoch=epoch,epochs=c['epochs'],lr=lr,next_lr=opt.param_groups[0]['lr'],train_loss=loss_sum/count,main_loss=main_sum/count,aux_loss=aux_sum/count,seconds=seconds,best_iou=best,best_epoch=best_epoch,**metrics)
        with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        atomic_json(status,dict(status='running',phase='epoch_complete',epoch=epoch,epochs=c['epochs'],best_iou=best,best_epoch=best_epoch,mean_recent_epoch_seconds=float(np.mean(durations[-10:])),remaining_seconds=float(np.mean(durations[-10:])*(c['epochs']-epoch-1)),pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        if stop[0]:atomic_json(status,dict(status='paused',epoch=epoch,epochs=c['epochs'],best_iou=best));return
    atomic_json(status,dict(status='complete',epochs=c['epochs'],best_iou=best,best_epoch=best_epoch,finished=time.strftime('%Y-%m-%d %H:%M:%S')))

def queue(matrix):
    import fcntl
    lock=open(ROOT/'artifacts/nuaa_queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=json.loads(Path(matrix).read_text())['runs'];jobs=[j for j in jobs if j['configuration']['dataset']=='NUAA-SIRST']
    root=ROOT/'runs/nuaa_600_300';root.mkdir(parents=True,exist_ok=True);active={};done=[];failed=[];pending=jobs.copy()
    # A detached screen owns this process; child workers remain attached to the queue.
    def gpu():
        s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');return [float(v.strip()) for v in s]
    total,_,_=gpu();budget=total/1024-1.5
    while pending or active:
        for name,(p,fh,job,reserve) in list(active.items()):
            code=p.poll()
            if code is not None:
                fh.close();st=json.loads((root/name/'status.json').read_text())
                (done if code==0 and st['status']=='complete' else failed).append(name);del active[name]
        total,free,util=gpu();observed_free=free;reserved=sum(v[3] for v in active.values())
        for job in list(pending):
            name=job['run_id'];out=root/name;state=out/'status.json'
            if state.exists() and json.loads(state.read_text()).get('status')=='complete':pending.remove(job);done.append(name);continue
            pf=ROOT/'artifacts/nuaa_preflight'/(name+'.json')
            if not pf.exists():continue
            info=json.loads(pf.read_text())
            if info['status']!='passed':pending.remove(job);failed.append(name);continue
            need=info['reserved_gib']*1.1+.5
            if len(active)>=2 or reserved+need>budget or need+.5>free/1024:continue
            out.mkdir(parents=True,exist_ok=True);fh=open(out/'console.log','a',buffering=1)
            p=subprocess.Popen([sys.executable,__file__,'--mode','run','--matrix',str(matrix),'--job',name],stdout=fh,stderr=fh,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
            active[name]=(p,fh,job,need);pending.remove(job);reserved+=need;free-=need*1024
            print('START',name,'PID',p.pid,flush=True)
        rows=[]
        for job in jobs:
            p=root/job['run_id']/'best_metrics.json'
            if not p.exists():continue
            r=json.loads(p.read_text());st=json.loads((p.parent/'status.json').read_text());rows.append(dict(run_id=job['run_id'],group=job['group'],status=st['status'],target_epochs=job['configuration']['epochs'],best_epoch=r['epoch'],IoU_percent=100*r['IoU'],nIoU_percent=100*r['nIoU'],Pd_percent=100*r['Pd'],Fa_per_million=1e6*r['Fa'],Fa_legacy_per_million=1e6*r['Fa_legacy'],batch_size=job['configuration']['train_batch_size']))
        atomic_json(root/'results_so_far.json',rows)
        if rows:
            tmp=root/'results_so_far.csv.tmp'
            with tmp.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            tmp.replace(root/'results_so_far.csv')
        atomic_json(root/'queue_status.json',dict(active={k:dict(pid=v[0].pid,reservation_gib=v[3]) for k,v in active.items()},pending=[j['run_id'] for j in pending],completed=done,failed=failed,gpu_utilization=util,gpu_free_mib_observed=observed_free,updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        if pending or active:time.sleep(10)
    print('QUEUE_FINISHED',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['preflight','run','queue'],required=True);ap.add_argument('--matrix',required=True);ap.add_argument('--job');a=ap.parse_args()
    if a.mode=='queue':queue(a.matrix);return
    jobs=json.loads(Path(a.matrix).read_text())['runs'];job=next(j for j in jobs if j['run_id']==a.job)
    try:
        if a.mode=='preflight':
            result=preflight(job);atomic_json(ROOT/'artifacts/nuaa_preflight'/(a.job+'.json'),result);print(json.dumps(result),flush=True)
        else:run(job)
    except BaseException as e:
        error=dict(status='failed',error=repr(e),traceback=traceback.format_exc())
        if a.mode=='preflight':atomic_json(ROOT/'artifacts/nuaa_preflight'/(a.job+'.json'),error)
        else:atomic_json(ROOT/'runs/nuaa_600_300'/a.job/'status.json',error)
        raise
if __name__=='__main__':main()

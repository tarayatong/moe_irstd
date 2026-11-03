"""Strict completed-run checks, same highest-IoU checkpoint, CPU-only backup."""
from pathlib import Path
import argparse,datetime,hashlib,json,math,torch,zipfile
R=Path('/root/autodl-tmp/moe_reproduction');A=R/'artifacts'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_id');args=ap.parse_args()
    jobs=json.loads((A/'irstd_experts_fixed005_v1_matrix.json').read_text())['runs']
    job=next(j for j in jobs if j['run_id']==args.run_id);name=job['run_id'];p=R/'runs/irstd_experts_fixed005_300'/name;c=job['configuration'];torch.set_num_threads(2)
    def read(f):return json.loads((p/f).read_text())
    st=read('status.json');bm=read('best_metrics.json');m=read('manifest.json');rows=[json.loads(s) for s in (p/'epochs.jsonl').read_text().splitlines()]
    best=torch.load(p/'best.pt',map_location='cpu',weights_only=False);last=torch.load(p/'last.pt',map_location='cpu',weights_only=False)
    row=next(x for x in rows if x['epoch']==bm['epoch']);keys=['IoU','nIoU','Pd','Fa']
    original=next(j for j in json.loads((A/'nuaa_fixed005_v2_matrix.json').read_text())['runs'] if j['run_id']==job['nuaa_reference'])['configuration']
    checks=dict(dataset=c['dataset']=='IRSTD',input256=c['base_size']==c['crop_size']==256,sparse_input=c['input_routing']=='sparse',paired_configuration={k:v for k,v in c.items() if k!='dataset'}=={k:v for k,v in original.items() if k!='dataset'},complete=st['status']=='complete' and st['epochs']==300,all_epochs=len(rows)==300 and [x['epoch'] for x in rows]==list(range(300)),last_epoch=last['epoch']==299,same_epoch=bm['epoch']==best['epoch']==st['best_epoch']==last['best_epoch'],same_configuration=bm['configuration']==best['configuration']==m['job']['configuration']==c,four_metrics=all(bm[k]==best['metrics'][k]==row[k] for k in keys),best_iou=bm['IoU']==max(x['IoU'] for x in rows)==last['best_iou'],finite_rows=all(math.isfinite(x[k]) for x in rows for k in ['train_loss','main_loss','aux_loss','test_loss']+keys),constant_lr=all(x['lr']==x['next_lr']==.05 for x in rows),checkpoint_lr=last['optimizer']['param_groups'][0]['lr']==.05,config_hash=last['config_sha256']==m['config_sha256']==hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest(),source_hash=last['source_sha256']==m['source_sha256']==best['source_sha256'] and all(sha(R/f)==h for f,h in last['source_sha256'].items()),finite_weights=all(torch.isfinite(t).all().item() for checkpoint in [best,last] for t in checkpoint['model'].values() if torch.is_tensor(t)),split_hashes=all(m[s+'_ids_sha256']==sha(R/'data/IRSTD/80_20'/(s+'.txt')) for s in ['train','test']))
    assert all(checks.values()),checks
    record=dict(run_id=name,dataset='IRSTD-1K',input_size=256,protocol='irstd_constant005_300',input_routing='sparse',nuaa_reference=job['nuaa_reference'],group=job['group'],finished=st['finished'],verified_at=datetime.datetime.now().astimezone().isoformat(),best_epoch_one_based=bm['epoch']+1,configuration=c,metrics={k:bm[k] for k in keys},display_metrics={k:round(bm[k]*(1e6 if k=='Fa' else 100),2) for k in keys},checks=checks,best_pt_sha256=sha(p/'best.pt'),last_pt_sha256=sha(p/'last.pt'))
    folder=A/'irstd_completed_audits_v1';folder.mkdir(exist_ok=True);audit=folder/f'{name}.json';audit.write_text(json.dumps(record,indent=2,allow_nan=False))
    backup=A/f'{name}_completed_20261026.zip'
    with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f in [audit,A/'irstd_experts_fixed005_v1_matrix.json',A/'irstd_experts_contract_v1.json',A/'irstd_dataset_audit_v1.json',A/'irstd_experts_queue_v1_status.json',A/'irstd_experts_results_v1.json']:
            z.write(f,str(f.relative_to(R)))
        for fn in ['best.pt','last.pt','best_metrics.json','epochs.jsonl','manifest.json','status.json','console.log']:
            f=p/fn;z.write(f,str(f.relative_to(R)))
    receipt=dict(run_id=name,archive=backup.name,bytes=backup.stat().st_size,sha256=sha(backup),best_pt_sha256=record['best_pt_sha256'],last_pt_sha256=record['last_pt_sha256'])
    (folder/f'{name}_backup_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(dict(verified=record,backup=receipt)),flush=True)
if __name__=='__main__':main()

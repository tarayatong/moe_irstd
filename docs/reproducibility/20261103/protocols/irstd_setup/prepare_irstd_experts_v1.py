"""CPU-only dataset validation and immutable protocol/source contract."""
from pathlib import Path
import json,hashlib,collections,datetime,ast,os
from PIL import Image
R=Path('/root/autodl-tmp/moe_reproduction');A=R/'artifacts';D=R/'data/IRSTD'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=json.loads((A/'nuaa_fullroute_tail_v1_contract.json').read_text())['sha256']
    assert all(sha(R/f)==h for f,h in old.items()),'Original frozen NUAA sources changed'
    lists={s:(D/'80_20'/f'{s}.txt').read_text().split() for s in ['train','test']}
    assert len(lists['train'])==len(set(lists['train']))==800
    assert len(lists['test'])==len(set(lists['test']))==201
    assert not set(lists['train'])&set(lists['test'])
    sizes=collections.Counter();modes=collections.Counter();files={}; by_split={};mask_ranges=collections.Counter()
    for split,ids in lists.items():
        hashes=set()
        for i in ids:
            im=D/'images'/f'{i}.png';ma=D/'masks'/f'{i}.png'
            with Image.open(im) as x,Image.open(ma) as y:
                x.load();y.load();assert x.size==y.size,(i,x.size,y.size)
                sizes[str(x.size)]+=1;modes[str((x.mode,y.mode))]+=1
                mask_ranges[str(y.convert('L').getextrema())]+=1
                hashes.add(hashlib.sha256(x.convert('RGB').tobytes()).hexdigest())
            files[str(im.relative_to(R))]=sha(im);files[str(ma.relative_to(R))]=sha(ma)
        by_split[split]=hashes
    duplicates=by_split['train']&by_split['test']
    assert not duplicates,('Exact image-content overlap',len(duplicates))
    audit=dict(dataset='IRSTD-1K',train_images=800,test_images=201,overlap_ids=0,overlap_image_content=0,total_pairs=1001,image_sizes=dict(sizes),image_mask_modes=dict(modes),mask_extrema=dict(mask_ranges),split_sha256={s:sha(D/'80_20'/f'{s}.txt') for s in lists},data_sha256=files,checked_at=datetime.datetime.now().astimezone().isoformat(),input_size=256,normalization='Existing IRSTD loader mean87.4661865234375/std39.71953201293945; same transform pipeline as NUAA, dataset-specific constants retained')
    (A/'irstd_dataset_audit_v1.json').write_text(json.dumps(audit,indent=2))
    matrix=json.loads((A/'irstd_experts_fixed005_v1_matrix.json').read_text()); originals=json.loads((A/'nuaa_fixed005_v2_matrix.json').read_text())['runs']
    for j in matrix['runs']:
        c=j['configuration'];o=next(x for x in originals if x['run_id']==j['nuaa_reference'])['configuration']
        assert [k for k in c if c[k]!=o[k]]==['dataset']
        assert c['base_size']==c['crop_size']==256 and c['epochs']==300 and c['input_routing']=='sparse'
    # AST equivalence checks for all numerical training/evaluation operations.
    oldsrc=(A/'run_nuaa_fixed005_v2.py').read_text();newsrc=(A/'run_irstd_experts_fixed005_v1.py').read_text();ot=ast.parse(oldsrc);nt=ast.parse(newsrc)
    for name in ['seed_all','source_hashes','Metrics','evaluate','preflight']:
        assert ast.dump(next(n for n in ot.body if getattr(n,'name','')==name))==ast.dump(next(n for n in nt.body if getattr(n,'name','')==name))
    files_to_freeze=[R/f for f in old]+list((R/'code/experiment_runtime').rglob('*.py'))+[A/f for f in ['run_irstd_experts_fixed005_v1.py','run_irstd_experts_queue_v1.py','audit_irstd_experts_completed_v1.py','prepare_irstd_experts_v1.py','irstd_experts_fixed005_v1_matrix.json','irstd_dataset_audit_v1.json']]+[D/'80_20'/f'{s}.txt' for s in lists]
    contract=dict(protocol=matrix['protocol'],created=datetime.datetime.now().astimezone().isoformat(),sha256={str(p.relative_to(R)):sha(p) for p in files_to_freeze},data_sha256=files,original_nuaa_contract_preserved=True)
    assert not (A/'irstd_experts_contract_v1.json').exists(),'Contract exists; inspect instead of replacing'
    (A/'irstd_experts_contract_v1.json').write_text(json.dumps(contract,indent=2))
    (A/'irstd_completed_audits_v1').mkdir(exist_ok=True)
    print(json.dumps({k:v for k,v in audit.items() if k!='data_sha256'}));print('CONTRACT_FILES',len(contract['sha256']))
if __name__=='__main__':main()

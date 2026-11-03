import hashlib,json,shutil
from pathlib import Path
root=Path('C:/Users/tyf/Documents/Codex/2026-10-21/zhe/outputs')
out=Path('D:/workspace/paper/moe/work/handoff_20261103')
paths=[
 'server_reproduction_20261023/nuaa_frozen_runtime.tar.gz',
 'server_reproduction_20261023/local_python_snapshot.tar.gz',
 'irstd_experts_fixed005_20261026/irstd_setup_sources_20261026.zip',
 'nuaa_hf_joint_20261027/hf_joint_start_bundle.zip',
 'nuaa_hf_triple_product_20261027/start_bundle.zip',
 'nuaa_hf_triple_product_full_20261027/full_start_bundle.zip',
 'context_sparse_final_20261028/context_sparse_final_source.zip',
]
inventory=[]
for name in paths:
    p=root/name
    dest=out/'sources'/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dest)
    inventory.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
src=root/'irstd_experts_fixed005_20261026/setup/artifacts'
shutil.copytree(src,out/'protocols/irstd_setup',dirs_exist_ok=True)
for name in ['context_sparse_final_20261028/verified_weights/context_sparse_best.pt',
             'server_reproduction_20261023/expert_configuration_completed/experts_multi_low/best.pt']:
    p=root/name
    inventory.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(out/'artifact_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
print(json.dumps(inventory,indent=2))

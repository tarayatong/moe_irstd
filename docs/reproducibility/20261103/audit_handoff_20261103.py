"""Build a read-only evidence audit and a portable small-file handoff package."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('C:/Users/tyf/Documents/Codex/2026-10-21/zhe/outputs')
OUT = Path('D:/workspace/paper/moe/work/handoff_20261103')
OUT.mkdir(exist_ok=True)

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

rows = []
groups = [
    ('server_reproduction_20261023', read(ROOT/'server_reproduction_20261023/monitor_state.json')['completed_jobs'] + ['NUAA-SIRST_default']),
    ('irstd_experts_fixed005_20261026', read(ROOT/'server_reproduction_20261023/results_table/table_receipt.json')['irstd_run_ids']),
]
for folder in ['nuaa_hf_joint_20261027', 'nuaa_hf_triple_product_20261027', 'nuaa_hf_triple_product_full_20261027']:
    ids = sorted({p.parent.name for p in (ROOT/folder).rglob('manifest.json') if (p.parent/'best.pt').exists()})
    groups.append((folder, ids))
for folder, ids in groups:
    base = ROOT/folder
    manifests = list(base.rglob('manifest.json'))
    for run_id in ids:
        matches = []
        for p in manifests:
            m = read(p)
            job = m.get('job', {})
            rid = job.get('run_id', m.get('run_id', p.parent.name))
            if rid == run_id and (p.parent/'best.pt').exists():
                matches.append((p, m))
        if not matches:
            rows.append(dict(run_id=run_id, error='No local manifest+best.pt pair found'))
            continue
        p, m = sorted(matches, key=lambda item: str(item[0]))[0]
        cfg = m.get('job', {}).get('configuration', m.get('configuration', {}))
        dest = OUT/'evidence'/run_id
        dest.mkdir(parents=True, exist_ok=True)
        evidence = {}
        for name in ['manifest.json', 'best_metrics.json', 'verified_metrics.json', 'status.json', 'epochs.jsonl']:
            src = p.parent/name
            if src.exists():
                shutil.copy2(src, dest/name)
                evidence[name] = sha(src)
        checkpoint = p.parent/'best.pt'
        audits = list((base/'completed_audits').glob(run_id+'.json'))
        if run_id == 'NUAA-SIRST_default':
            audits += [base/'NUAA_main600_verified_metrics.json']
        if run_id == 'NUAA-SIRST_control300':
            audits += [base/'NUAA_fixed005_control300_verified_metrics.json']
        if (base/'completed_audit.json').exists():
            audits += [base/'completed_audit.json']
        if (base/(run_id+'_verified.json')).exists():
            audits += [base/(run_id+'_verified.json')]
        for a in audits:
            if a.exists():
                shutil.copy2(a, dest/'independent_audit.json')
        best = read(p.parent/'best_metrics.json') if (p.parent/'best_metrics.json').exists() else {}
        rows.append(dict(run_id=run_id, configuration=cfg, manifest=str(p),
                         checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                         source_hash_count=len(m.get('source_sha256', {})),
                         train_split_sha256=m.get('train_ids_sha256'),
                         test_split_sha256=m.get('test_ids_sha256'),
                         torch=m.get('torch'), cuda=m.get('cuda'),
                         best_metrics=best, independent_audit=bool(audits),
                         evidence_sha256=evidence))

for folder, names in {
    'server_reproduction_20261023': ['environment.lock.txt', 'requirements-reproduction.txt', 'python_version.txt', 'nvidia_smi.txt', 'nuaa_fixed005_matrix.json', 'nuaa_experiment_matrix.json', 'run_nuaa_fixed005.py', 'run_nuaa_suite.py', 'ablation_variants.py', 'local_source_sha256.json'],
    'context_sparse_final_20261028': ['project_sync_receipt.json', 'local_backup_receipt.json', 'final_source_manifest.json'],
}.items():
    dest = OUT/'protocols'/folder
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = ROOT/folder/name
        if src.exists():
            shutil.copy2(src, dest/name)

report = dict(audited_on='2026-11-03', scope='Archived NUAA/IRSTD runs plus later HF exploratory runs; file/configuration provenance audit only, no new training or metric evaluation',
              runs=rows, total=len(rows), local_checkpoint_pairs=sum('checkpoint' in r for r in rows),
              gaps=[r for r in rows if 'error' in r or not r.get('source_hash_count') or not r.get('train_split_sha256') or not r.get('test_split_sha256')])
(OUT/'experiment_index.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
with (OUT/'experiment_index.csv').open('w', encoding='utf-8-sig', newline='') as f:
    fields=['run_id','dataset','expert_variant','input_routing','epochs','lr','scheduler','seed','checkpoint_sha256','manifest','checkpoint','source_hash_count','train_split_sha256','test_split_sha256','error']
    w=csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        c=r.get('configuration', {})
        w.writerow({k:r.get(k,c.get(k,'')) for k in fields})
print(json.dumps({k:report[k] for k in ['total','local_checkpoint_pairs','gaps']}, ensure_ascii=False, indent=2))

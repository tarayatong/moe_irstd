import json
from pathlib import Path
import torch

root=Path('D:/workspace/paper/moe/work/handoff_20261103')
index=json.loads((root/'experiment_index.json').read_text(encoding='utf-8'))
checks=[]
for row in index['runs']:
    checkpoint=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
    best=row['best_metrics']
    metrics=checkpoint.get('metrics', checkpoint)
    comparisons={k:abs(float(metrics[k])-float(best[k]))<1e-12 for k in ['IoU','nIoU','Pd','Fa'] if k in metrics and k in best}
    comparisons['all_four_metrics_present']=all(k in comparisons for k in ['IoU','nIoU','Pd','Fa'])
    comparisons['configuration']=checkpoint.get('configuration')==row['configuration']
    comparisons['epoch']=checkpoint.get('epoch')==best.get('epoch')
    manifest=json.loads(Path(row['manifest']).read_text(encoding='utf-8-sig'))
    comparisons['source_hashes']=checkpoint.get('source_sha256')==manifest.get('source_sha256')
    checks.append(dict(run_id=row['run_id'],checks=comparisons,passed=all(comparisons.values())))
result=dict(scope='CPU deserialization and comparison with archived best_metrics and manifest; no inference or training',total=len(checks),passed=sum(c['passed'] for c in checks),runs=checks)
(root/'checkpoint_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='runs'}))
print(json.dumps([r for r in checks if not r['passed']],indent=2))

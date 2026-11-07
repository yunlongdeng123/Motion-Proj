"""私有取回核验：原A/B包足以重建C，记录新增数组hash，不复制观测两次。"""
from pathlib import Path
import json,hashlib
import numpy as np
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1/targets_edgeblend_v1')
m=json.loads((B/'target_manifest.json').read_text());records=[]
for row in m['rows']:
 p=B/row['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];z=np.load(p)
 records.append(dict(raw=row['raw'],file=row['file'],arrays={k:dict(sha256=hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest(),dtype=str(z[k].dtype),shape=list(z[k].shape)) for k in ['edgeblend_rgb','edgeblend_weight','edgeblend_ready']}))
out=dict(status='C_ADDED_ARRAY_BYTES_RECORDED',target_manifest_sha256=hashlib.sha256((B/'target_manifest.json').read_bytes()).hexdigest(),rows=records)
(B/'edgeblend_array_hashes.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(len(records))

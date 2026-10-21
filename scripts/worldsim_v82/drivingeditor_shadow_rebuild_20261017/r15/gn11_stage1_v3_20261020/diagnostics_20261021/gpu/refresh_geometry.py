"""最终所选像素LiDAR一致性闸门；只重算已有trace标签，不重光栅化。"""
import json
from pathlib import Path
from collections import Counter
from evidence import Evidence
from gpu_backend import CAMPAIGN,HERE,clean

config=json.loads((CAMPAIGN/'arms/full/run30k/effective_config.json').read_text());meta=json.loads(Path(config['v3_bundle']).read_text());e=Evidence(CAMPAIGN,meta,float(meta['meter_to_model']))
files=list(Path('/root/autodl-tmp/gsdiag_gpu_20261021_r1/runs/20261021T123228Z_682bee5a').glob('query_*.json'))
files+=list((HERE/'v1_export_r3/data/gpu').glob('*.js'))+list((HERE/'evidence_cases_r2/data/gpu').glob('*.js'))
out={};counts=Counter()
for path in files:
    text=path.read_text();v=json.loads(text.split(']=',1)[1].removesuffix(';')) if path.suffix=='.js' else json.loads(text)
    if v.get('status')!='NATIVE_BUFFER_CUDA_TRACE_VERIFIED':continue
    v=e.enrich_trace(v);key='/'.join(map(str,['gpu',v['arm'],v['raw'],*v['pixel'],v['mode']]))
    out[key]=dict(geometry_evidence=v['geometry_evidence'],by_id={r['id']:r['geometry_evidence'] for r in v['top']})
    counts.update(v['geometry_evidence']['counts'])
if len(out)!=180:raise RuntimeError(f'need all180, got{len(out)}')
target=HERE/'geometry_selected_lidar_r1.json'
with target.open('x') as f:json.dump(clean(out),f,separators=(',',':'),allow_nan=False)
print(json.dumps(dict(output=str(target),queries=len(out),counts=counts,optimizer_updates=0)))

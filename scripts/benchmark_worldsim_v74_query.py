"""一次固定GPU驻留资产的纯首交点计时；不读取真实距离，不重算质量。"""
import argparse,json,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.surface_readout import _select_first_triangle
parser=argparse.ArgumentParser();parser.add_argument('--decision',type=Path,required=True);parser.add_argument('--append-missing',action='store_true');args=parser.parse_args();torch.set_num_threads(2)
source=json.loads((args.decision/'paired_log_comparisons.json').read_text());cases={};records=[]
for folder in map(Path,source['sources']):
    for c in json.loads((folder/'manifest.json').read_text())['cases']:cases[(c['dataset'],c['case_id'])]=c
    records+=json.loads((folder/'per_actor.json').read_text())
out=args.decision/'query_timing';out.mkdir(exist_ok=args.append_missing)
rows=json.loads((out/'results.json').read_text())['per_actor'] if args.append_missing else []
existing={(r['dataset'],r['case_id'],r['method']) for r in rows}
for r in records:
    if r.get('contract_missing_input') or not r.get('surface_path'):continue
    if (r['dataset'],r['case_id'],r['method']) in existing:continue
    c=cases[(r['dataset'],r['case_id'])]
    with np.load(c['query_rays_file']) as z:o=torch.tensor(z['origins_actor_m'],device='cuda');d=torch.tensor(z['directions_actor'],device='cuda')
    with np.load(r['surface_path']) as z:v=torch.tensor(z['vertices_actor_m'],dtype=torch.float32,device='cuda');f=torch.tensor(z['faces'],dtype=torch.long,device='cuda')
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.no_grad():depth,face=_select_first_triangle(v,f,o,d,ray_chunk=256,face_chunk=2048)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    rows.append({'dataset':r['dataset'],'case_id':r['case_id'],'method':r['method'],'rays':len(o),'faces':len(f),'gpu_resident_query_seconds':elapsed if len(o) else None,'microseconds_per_ray':elapsed/len(o)*1e6 if len(o) else None,'order':len(rows)})
groups=defaultdict(list)
for r in rows:groups[(r['dataset'],r['method'])].append(r)
summary=[]
for (ds,m),rs in groups.items():
    sec=[r['gpu_resident_query_seconds'] for r in rs if r['gpu_resident_query_seconds'] is not None];ray=sum(r['rays'] for r in rs)
    summary.append({'dataset':ds,'method':m,'objects':len(rs),'query_rays':ray,'total_seconds':sum(sec),'object_median_seconds':float(np.median(sec)) if sec else None,'pooled_microseconds_per_ray':sum(sec)/ray*1e6 if ray else None})
(out/'results.json').write_text(json.dumps({'protocol':'one pass, no warmup/repetition sweep; same V73 opaque double-sided Moller-Trumbore first-hit selector, 256x2048 chunks, GPU resident inputs; no truth/nearest-point metric or asset-transfer time in timer; not hardware RT or statistical speed benchmark','source':str(args.decision),'summary':summary,'per_actor':rows},ensure_ascii=False,indent=2)+'\n')
print('Pure-query one-pass timing complete',len(rows),'assets')

"""由测量而非模型挑选可判别对照；额外冻结前向，优化0步。"""
import json
from pathlib import Path
import numpy as np
from evidence import Evidence,GATES
from gpu_backend import GPUAdapter,CAMPAIGN,HERE,clean,global_slot

def main():
    import torch
    from unittest.mock import patch
    config=json.loads((CAMPAIGN/'arms/full/run30k/effective_config.json').read_text());meta=json.loads(Path(config['v3_bundle']).read_text())
    e=Evidence(CAMPAIGN,meta,float(meta['meter_to_model']));f=e.frame(102);u=e.u
    valid=np.logical_and.reduce([u['gate_'+g][13] for g in GATES])&((u['base_weight']>0).sum(0)>=2)
    indices=np.flatnonzero(valid)
    cases=[]
    for target in ([365,280],[220,280]):
        ordered=indices[np.argsort(np.linalg.norm(f['uv'][indices]-target,axis=1))]
        for uid in ordered:
            pixel=np.rint(f['uv'][uid]).astype(int).tolist()
            if not(20<=pixel[0]<460 and 20<=pixel[1]<300):continue
            s=e.ray_support(102,pixel)
            if s['road']['supported'] and s['road']['current_visible'] and s['lidar_m'] and all(np.linalg.norm(np.array(pixel)-v['pixel'])>30 for v in cases):
                cases.append(dict(name='confirmed_visible_road',pixel=pixel,world_unit=int(uid)));break
    coords=np.argwhere(f['target']&(f['lidar']>0))
    for y,x in coords[np.linspace(0,max(0,len(coords)-1),min(100,len(coords))).astype(int)]:
        pixel=[int(x),int(y)];s=e.ray_support(102,pixel)
        if s['vehicle'] and all(np.linalg.norm(np.array(pixel)-v['pixel'])>25 for v in cases):cases.append(dict(name='measured_target_vehicle_surface',pixel=pixel))
        if len(cases)==4:break
    if len(cases)<3:raise RuntimeError('not enough measured controls; no synthetic substitute')
    out=HERE/'evidence_cases_r2';out.mkdir(exist_ok=False);records=[]
    with global_slot(),patch.object(torch.optim.Adam,'step',side_effect=RuntimeError('no optimization')),torch.no_grad():
        a=GPUAdapter()
        for arm in ('full','real_only','prior_only','ownership_only','gn11_v1'):
            for case in cases:
                x,y=case['pixel']
                for mode in ('deleted_five','joint'):
                    q=a.trace(arm,102,x,y,mode);key=f'gpu/{arm}/102/{x}/{y}/{mode}';path=f'data/gpu/{arm}_102_{x}_{y}_{mode}.js';dest=out/path;dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_text('window.GSDATA=window.GSDATA||{};window.GSDATA['+json.dumps(key)+']='+json.dumps(q,separators=(',',':'),allow_nan=False)+';')
                    records.append(dict(key=key,path=path,counts=q['geometry_evidence']['counts']));print(arm,case['pixel'],q['geometry_evidence']['counts'],flush=True)
    (out/'receipt.json').write_text(json.dumps(clean(dict(cases=cases,queries=records,selection='independent original nine-gate exposure and target-mask+LiDAR+OBB; no model residual selection',optimizer_updates=0)),indent=2))

if __name__=='__main__':main()

"""按证据/补全来源与原始束归属分析自由空间侵入，不改变模型或读出。"""
from pathlib import Path
import json
import sys

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.surface_readout import first_triangle_intersection

torch.set_num_threads(6)
base=Path('/root/autodl-tmp/runs/worldsim_v73/WS-V73-M2-PHYSICAL-SURFACE-01')
out=Path('/root/autodl-tmp/runs/worldsim_v73/WS-V73-M2-FAILURE-ANALYSIS-01/20260907T160000Z__physical-controls-r1')
out.mkdir(parents=True,exist_ok=False)
results={}
with torch.no_grad():
 for variant in ['joint-r2','pointwise-r1','lidar-only-r1','joint-no-free-r1']:
    root=base/('20260907T154500Z__physical-controls-'+variant)
    surface=torch.load(root/'final_surface.pt',weights_only=True)
    rays=torch.load(root/'raw_actor_rays.pt',weights_only=True)
    vertices=surface['vertices_actor_m'].float().cuda(); faces=surface['faces'].cuda()
    face_source=surface['source'].cuda()[faces[:,0]//9]
    all_rows=[]
    for role in ['build','heldout_time']:
        chosen=[r for r in rays if r['role']==role]
        origin=torch.cat([r['origins_actor_m'] for r in chosen]).cuda()
        direction=torch.cat([r['directions_actor'] for r in chosen]).cuda()
        ranges=torch.cat([r['observed_first_range_m'] for r in chosen]).cuda()
        positive=torch.cat([r['positive_actor'] for r in chosen]).cuda()
        depth,source=first_triangle_intersection(vertices,faces,origin,direction,face_source)
        early=torch.isfinite(depth)&(depth<ranges-.2)
        intrusion=torch.where(early,ranges-.2-depth,torch.zeros_like(depth))
        rows=[]
        for src,label in [(0,'evidence'),(1,'completion')]:
          for owned in [True,False]:
            take=early&(source==src)&(positive==owned)
            values=intrusion[take]
            rows.append({'source':label,'original_return_is_actor':owned,'early_rays':int(take.sum()),
                'intrusion_sum_m':values.sum().item(),'mean_given_early_m':values.mean().item() if len(values) else None,
                'maximum_intrusion_m':values.max().item() if len(values) else None})
        all_rows.append({'role':role,'rays':len(ranges),'early_rays':int(early.sum()),'groups':rows,
                         'intrusion_total_m':intrusion.sum().item()})
    results[variant]=all_rows
(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))

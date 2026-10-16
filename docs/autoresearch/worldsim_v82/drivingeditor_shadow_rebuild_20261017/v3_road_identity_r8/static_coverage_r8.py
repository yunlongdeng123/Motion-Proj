from pathlib import Path
import sys,json,time
import torch
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r8/shadow_rebuild_v3'))
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import seed_from_state,write
from shadow_rebuild.raster import cuda_raster
torch.set_num_threads(2);start=time.monotonic();s=torch.load(r/'E3_source_ownership_s0/latest.pt',map_location='cpu',weights_only=False)
if s['global_step']!=8000:raise ValueError('frozen8k source mismatch')
c=s['config'].copy();c['weights']=LossWeights(**c['weights']);c=Config(**c).validate()
m=SceneModel(seed_from_state(s['state']),c).cuda();m.load_state_dict(s['state']);m.requires_grad_(False);d=Dataset(r/'data_r4_road_identity','cuda',True)
records=[]
with torch.no_grad():
 for index in d.train_indices:
  f=d.frame(index);mask=f.road_certified
  g=m.field.evaluate(f.poses,f.valid,f.camera.center,{a:{'delete':True} for a in range(len(d.actor_ids))})
  z=(g['xyz']@f.camera.view[:3,:3].T+f.camera.view[:3,3])[:,2:3]
  feat=torch.cat((torch.ones_like(z),z,g['road'][:,None].float()),-1)
  image=cuda_raster(f.camera,g['xyz'],g['cov'],g['opacity'],feat)
  alpha=image[...,0];depth=image[...,1]/alpha.clamp_min(1e-6)
  full=m.field.evaluate(f.poses,f.valid,f.camera.center);dyn=(full['owner']>=0).float()[:,None]
  leak=cuda_raster(f.camera,full['xyz'],full['cov'],full['opacity'],dyn)[...,0]
  row=dict(frame_id=f.frame_id,pixels=int(mask.sum()),alpha_sum=float(alpha[mask].sum()),missing_alpha_lt_0p5=int((alpha[mask]<.5).sum()),
    covered_alpha_gt_0p9=int((alpha[mask]>.9).sum()),depth_abs_error_sum=float((depth[mask]-f.depth[mask]).abs().sum()),
    dynamic_visible_sum=float(leak[mask].sum()))
  near=mask&(f.depth<30);row['near30m_pixels']=int(near.sum());row['near30m_covered_gt_0p9']=int((alpha[near]>.9).sum())
  records.append(row)
 total=sum(x['pixels'] for x in records)
 summary=dict(global_step=8000,parameters_updated=False,certified_pixels=total,static_alpha_mean=sum(x['alpha_sum'] for x in records)/max(1,total),
   static_coverage_gt_0p9_fraction=sum(x['covered_alpha_gt_0p9'] for x in records)/max(1,total),
   missing_static_alpha_lt_0p5_fraction=sum(x['missing_alpha_lt_0p5'] for x in records)/max(1,total),
   near30m_coverage_gt_0p9_fraction=sum(x['near30m_covered_gt_0p9'] for x in records)/max(1,sum(x['near30m_pixels'] for x in records)),
   depth_mean_abs_error_m=sum(x['depth_abs_error_sum'] for x in records)/max(1,total),
   dynamic_alpha_mean=sum(x['dynamic_visible_sum'] for x in records)/max(1,total),
   frames=records,elapsed_seconds=time.monotonic()-start,
   scope='static isolated coverage on new train-only positive road certificate, not clean appearance GT or edit quality')
write(r/'static_coverage_008000_r8.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='frames'}),flush=True)

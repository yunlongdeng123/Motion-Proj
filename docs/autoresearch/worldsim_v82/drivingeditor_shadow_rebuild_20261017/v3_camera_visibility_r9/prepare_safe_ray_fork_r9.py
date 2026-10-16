from pathlib import Path
import sys,json,copy,os
from dataclasses import asdict
import numpy as np
import torch
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r8/shadow_rebuild_v3'))
from shadow_rebuild.data import Dataset
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.model import SceneModel
from shadow_rebuild.references import refresh
from shadow_rebuild.train import seed_from_state,write,save,preflight
torch.set_num_threads(2);out=r/'E5_ray_visible_from8k';out.mkdir(exist_ok=False)
parent=r/'E3_source_ownership_s0/latest.pt';s=torch.load(parent,map_location='cpu',weights_only=False)
if s['global_step']!=8000 or s['stage']!='geometry':raise ValueError('safe parent gate mismatch')
old=Dataset(r/'data_r3_ownership','cpu',True);d=Dataset(r/'data_r5_ray_visible_road','cpu',True)
if s['input_contract']!=old.manifest:raise ValueError('safe parent inputs differ')
expected=copy.deepcopy(old.manifest);expected['certified_road_identity']=d.manifest['certified_road_identity'];expected['ownership_road_evidence']['camera_occluded_added_pixels_rejected']=True
if expected!=d.manifest:raise ValueError('unexpected input changes')
for name in ['seed.npz','trajectory.npz']:
 if not os.path.samefile(old.root/name,d.root/name):raise ValueError('raw geometry inputs changed')
for row in old.rows:
 if not os.path.samefile(old.root/row['rgb'],d.root/row['rgb']):raise ValueError('RGB changed')
 with np.load(old.root/row['supervision']) as a,np.load(d.root/row['supervision']) as b:
  if set(b.files)!=set(a.files)|{'road_certified'}:raise ValueError('unexpected supervision keys')
  for key in a.files:
   if key=='road':
    if np.any(b[key]&~a[key]):raise ValueError('visibility rejection cannot invent road pixels')
   elif not np.array_equal(a[key],b[key]):raise ValueError('body/known/depth changed')
raw=s['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate()
if cfg.geometry_steps!=30000 or cfg.weights.road_identity!=0:raise ValueError('safe parent configuration unexpected')
cfg.geometry_steps=29000;cfg.weights.road_identity=.5
receipt=preflight(d,cfg);m=SceneModel(seed_from_state(s['state']),cfg);m.load_state_dict(s['state']);refs=refresh(m.field,d)
if refs['valid_reference_points']==0:raise ValueError('no camera-visible clean reference support')
s['state']=m.state_dict();s['config']=asdict(cfg);s['input_contract']=d.manifest
s['supervision_lineage']=dict(kind='safe_parent_camera_visible_road_identity',parent_checkpoint=str(parent),parent_global_step=8000,
 previous=s.get('supervision_lineage'),changes={'weights.road_identity':[0.,.5],'geometry_steps':[30000,29000]},
 learned_parameters_optimizer_RNG_preserved=True,render_unchanged_before_updates=True,reference_buffers_refreshed=refs,
 discarded_control=dict(run='E4_road_identity_s0',updates=1000,charged_to_geometry=True,
 reason='body/bumper contamination after road identity trial;6921road certificates potentially camera-occluded; retain all failed evidence, not a unique causal attribution'),
 budget=dict(working_geometry_target=29000,discarded_geometry_updates=1000,material=10000,joint=5000,
             final_working_model_global_step=44000,total_charged_updates=45000))
save(out/'latest.pt',s);write(out/'resolved_config.json',s['config']);write(out/'input_contract.json',d.manifest)
write(out/'safe_ray_visibility_fork.json',dict(**s['supervision_lineage'],preflight=receipt))
print(json.dumps(dict(status='SAFE_R9_FORK_READY',working_global_step=8000,charged_updates=9000,certified_pixels=receipt['certified_road_pixels'],references=refs)),flush=True)

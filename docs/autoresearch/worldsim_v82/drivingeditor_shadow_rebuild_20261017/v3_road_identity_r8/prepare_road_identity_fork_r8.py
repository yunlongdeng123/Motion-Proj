from pathlib import Path
import sys,json,copy,os,argparse
from dataclasses import asdict
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--output',default='E4_road_identity_s0');p.add_argument('--weight',type=float,default=.5);a=p.parse_args()
if not 0<=a.weight<=10:raise ValueError('explicit finite road identity weight0..10 required')
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r8/shadow_rebuild_v3'))
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.train import write,save,preflight
out=r/a.output;out.mkdir(exist_ok=False);parent=r/'E3_source_ownership_s0/latest.pt';s=torch.load(parent,map_location='cpu',weights_only=False)
if s['global_step']!=8000 or s['stage']!='geometry':raise ValueError('parent gate mismatch')
old=Dataset(r/'data_r3_ownership','cpu',True);d=Dataset(r/'data_r4_road_identity','cpu',True)
if s['input_contract']!=old.manifest:raise ValueError('parent inputs differ')
expected=copy.deepcopy(old.manifest);expected['certified_road_identity']=d.manifest['certified_road_identity']
if expected!=d.manifest:raise ValueError('only new certified road input may change')
for name in ['seed.npz','trajectory.npz']:
 if not os.path.samefile(old.root/name,d.root/name):raise ValueError('original measurements must remain identical')
for row in old.rows:
 if not os.path.samefile(old.root/row['rgb'],d.root/row['rgb']):raise ValueError('RGB changed')
 with np.load(old.root/row['supervision']) as a,np.load(d.root/row['supervision']) as b:
  if set(b.files)!=set(a.files)|{'road_certified'}:raise ValueError('unexpected supervision fields')
  for key in a.files:
   if not np.array_equal(a[key],b[key]):raise ValueError('existing supervision changed: '+key)
cfg=s['config'].copy();cfg['weights']=LossWeights(**cfg['weights']);cfg=Config(**cfg).validate()
if cfg.weights.road_identity!=0:raise ValueError('not a new road certificate intervention')
cfg.weights.road_identity=a.weight;receipt=preflight(d,cfg)
coverage=json.loads((r/'static_coverage_008000_r8.json').read_text())
if coverage['missing_static_alpha_lt_0p5_fraction']>.01:raise ValueError('restore measured static coverage before applying this loss')
s['config']=asdict(cfg);s['input_contract']=d.manifest
s['supervision_lineage']=dict(kind='independent_positive_road_identity',parent_checkpoint=str(parent),parent_global_step=8000,
 previous=s.get('supervision_lineage'),changes={'weights.road_identity':[0.,a.weight]},
 source='train-only actual LiDAR >=3worldtimes, own local plane normal, positive body3D4cm contact protection',
 unknown_pixels_without_certificate='no added loss, no background label',full_known_mask_exterior='unchanged',
 model_optimizer_RNG_preserved=True,not_same_step_causal_ablation=True)
save(out/'latest.pt',s);write(out/'resolved_config.json',s['config']);write(out/'input_contract.json',d.manifest)
write(out/'road_identity_fork.json',dict(**s['supervision_lineage'],preflight=receipt,static_coverage=coverage))
print(json.dumps(dict(status='FORK_READY_IDENTICAL_MODEL_AT8000',certificate_pixels=receipt['certified_road_pixels'],weight=a.weight,static_missing_fraction=coverage['missing_static_alpha_lt_0p5_fraction'])),flush=True)

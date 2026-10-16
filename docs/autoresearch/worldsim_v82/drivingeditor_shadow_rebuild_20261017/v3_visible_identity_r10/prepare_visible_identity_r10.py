from pathlib import Path
import json,sys,copy
from dataclasses import asdict
import torch
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r10/shadow_rebuild_v3'))
from shadow_rebuild.data import Dataset
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.train import save,write
out=r/'E6_visible_identity_s0';out.mkdir(exist_ok=False);parent=r/'E5_ray_visible_from8k/latest.pt';s=torch.load(parent,map_location='cpu',weights_only=False)
if s['global_step']!=9000 or s['stage']!='geometry':raise ValueError('safe9k source mismatch')
d=Dataset(r/'data_r5_ray_visible_road','cpu',True)
if s['input_contract']!=d.manifest:raise ValueError('data must remain identical')
raw=s['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate();previous=asdict(cfg)
if cfg.instance_balanced_visible_identity or cfg.geometry_steps!=29000 or cfg.weights.road_identity!=.5:raise ValueError('unexpected parent configuration')
cfg.instance_balanced_visible_identity=True
expected=copy.deepcopy(previous);expected['instance_balanced_visible_identity']=True
if asdict(cfg)!=expected:raise ValueError('only visible identity normalization may change')
s['config']=asdict(cfg);s['supervision_lineage']=dict(kind='visible_body_identity_normalization',parent_checkpoint=str(parent),parent_global_step=9000,
 previous=s.get('supervision_lineage'),changes={'instance_balanced_visible_identity':[False,True]},
 learned_parameters_optimizer_RNG_and_inputs_preserved=True,
 identity='inside per known positive Actor domain mean then mean Actors; original known-HWK identity outside component exactly unchanged, weight0.1',
 mask_inside='unchanged raw-alpha domain mean weight0.05',mask_outside='unchanged full-image per Actor visible-alpha mean sum weight0.5',
 road_identity='unchanged camera-visibility-guarded actual road, weight0.5',unknown='no body labels and no identity gradient without known mask')
save(out/'latest.pt',s);write(out/'resolved_config.json',s['config']);write(out/'input_contract.json',s['input_contract']);write(out/'visible_identity_fork.json',s['supervision_lineage'])
print(json.dumps(dict(status='R10_FORK_READY_IDENTICAL_MODEL',step=9000,charged_updates=10000,only_new_flag=True)),flush=True)

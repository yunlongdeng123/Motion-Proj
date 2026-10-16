"""General contour-plane split-only probe, no SGD or ownership change."""
from pathlib import Path
import sys,json,time
import numpy as np
import torch
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r7/shadow_rebuild_v3'))
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import seed_from_state,write,save
from shadow_rebuild.ownership import classify_evidence,split_plane
from shadow_rebuild.boundary_plane import propose_contour_planes
torch.set_num_threads(2);out=r/'E3_contour_split_s0';out.mkdir(exist_ok=False)
parent=r/'E2_ownership_s0/latest.pt';saved=torch.load(parent,map_location='cpu',weights_only=False)
if saved['global_step']!=7000:raise ValueError('source gate mismatch')
cfg=saved['config'].copy();cfg['weights']=LossWeights(**cfg['weights']);cfg=Config(**cfg).validate()
d=Dataset(r/'data_r3_ownership','cpu',True);model=SceneModel(seed_from_state(saved['state']),cfg);model.load_state_dict(saved['state'])
opt=model.optimizer('geometry');opt.load_state_dict(saved['optimizer']);audit=r/'ownership_007000_r6';e=np.load(audit/'per_gaussian.npz')
labels=classify_evidence(torch.from_numpy(e['evidence']),torch.from_numpy(e['frames'].astype(np.int32)),model.field.owner,[5,6,12,18,25,26,37])
indices=labels['mixed'].nonzero().flatten().numpy();start=time.monotonic()
rows,direction,cuts,provenance=propose_contour_planes(saved['state'],d,audit,r/'road_evidence_r5',indices)
receipt=split_plane(model.field,torch.from_numpy(rows),torch.from_numpy(direction),torch.from_numpy(cuts),opt)
receipt.update(candidate_mixed=len(indices),actual_splits=len(rows),parent_checkpoint=str(parent),parent_global_step=7000,
               training_updates=0,identity_changed=False,no_height_crop=True,
               limitation='split-only bounded topology probe; keep both children and audit, not asset cleanliness acceptance',
               provenance=provenance,elapsed_seconds=time.monotonic()-start)
saved['state']=model.state_dict();saved['optimizer']=opt.state_dict()
saved['supervision_lineage']=dict(kind='general_contour_plane_split',parent_checkpoint=str(parent),previous=saved.get('supervision_lineage'),receipt='contour_split.json')
save(out/'latest.pt',saved);write(out/'contour_split.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k!='provenance'}),flush=True)

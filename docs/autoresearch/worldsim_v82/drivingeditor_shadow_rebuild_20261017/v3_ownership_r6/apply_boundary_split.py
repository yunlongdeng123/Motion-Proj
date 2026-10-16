"""Split-only geometric intervention at global6000, before any further SGD."""
from pathlib import Path
import sys,json,copy,time,argparse
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--output',default='E2_split_s0');p.add_argument('--source-run',default='E1_balanced_s0');p.add_argument('--skip-split',action='store_true');a=p.parse_args()
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
sys.path.insert(0,str(r/'implementation_r5/shadow_rebuild_v3'))
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import seed_from_state,write,save
from shadow_rebuild.ownership import classify_evidence,split_boundary
from shadow_rebuild.boundary import propose_boundary_splits
torch.set_num_threads(2);out=r/a.output;out.mkdir(exist_ok=False)
parent=r/a.source_run/'latest.pt';saved=torch.load(parent,map_location='cpu',weights_only=False)
if saved['global_step']!=6000 or saved['stage']!='geometry':raise ValueError('source gate mismatch')
cfg=saved['config'].copy();cfg['weights']=LossWeights(**cfg['weights']);cfg=Config(**cfg).validate()
d=Dataset(r/'data_r2','cpu',True);model=SceneModel(seed_from_state(saved['state']),cfg);model.load_state_dict(saved['state'])
optimizer=model.optimizer('geometry');optimizer.load_state_dict(saved['optimizer'])
audit=r/'ownership_006000_r5b';e=np.load(audit/'per_gaussian.npz')
labels=classify_evidence(torch.from_numpy(e['evidence']),torch.from_numpy(e['frames'].astype(np.int32)),model.field.owner,[5,6,12,18,25,26,37])
indices=labels['mixed'].nonzero().flatten().numpy();start=time.monotonic()
rows,axes,cuts,provenance=propose_boundary_splits(saved['state'],d,audit,r/'road_evidence_r5',indices)
if a.skip_split:rows=rows[:0];axes=axes[:0];cuts=cuts[:0];provenance=[]
receipt=split_boundary(model.field,torch.from_numpy(rows),torch.from_numpy(axes),torch.from_numpy(cuts),optimizer)
receipt.update(candidate_mixed=len(indices),actual_splits=len(rows),parent_checkpoint=str(parent),parent_global_step=6000,
               split_enabled=not a.skip_split,
               training_updates=0,road_reassigned=0,scope='same config/RNG/input; only repeated projected body-road boundary split; keep both children, re-audit before reassignment',
               provenance=provenance,elapsed_seconds=time.monotonic()-start)
saved['state']=model.state_dict();saved['optimizer']=optimizer.state_dict()
saved['supervision_lineage']=dict(kind='boundary_split_repair',parent_checkpoint=str(parent),parent_global_step=6000,
                                previous=saved.get('supervision_lineage'),receipt='boundary_split.json',both_children_retained=True)
save(out/'latest.pt',saved);write(out/'boundary_split.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k!='provenance'}),flush=True)

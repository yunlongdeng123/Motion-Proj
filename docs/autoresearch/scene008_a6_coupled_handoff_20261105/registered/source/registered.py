"""Pair measured surface groups to their witnessed legacy support."""
from pathlib import Path
import sys
import numpy as np
import torch

RUN=Path(__file__).resolve().parent.parent
SOURCE=RUN.parent/'scene008_a6_coupled_handoff_20261105'
sys.path.insert(0,str(SOURCE/'source'))
import coupled as core
import fit_root
import fit_joint
import run_coupled

core.rescue.volume.RUN=RUN
core.rescue.volume.depth_order.takeover.calibrate.RUN=RUN
fit_root.RUN=RUN
fit_joint.RUN=RUN
run_coupled.RUN=RUN

from coupled import OBS,read,write,PRIMARY


class Registered(core.Coupled):
    def __init__(self,ledger):
        super().__init__(ledger)
        self.measurement_boxes=self.boxes.clone()
        self.boxes=self.zero.new_tensor(ledger['prism_bounds'][None])
        assert np.array_equal(ledger['native_row_id'],self.tr.native_ids.cpu().numpy())
        self.registered=self.zero.new_tensor(ledger['eligible'],dtype=torch.bool)

    def merged(self,raw,state,use_proxy=True):
        trace=super().merged(raw,state,use_proxy)
        count=len(self.actor_lineage[raw])
        keep=self.registered[self.actor_lineage[raw]]
        start=0 if state=='black' else len(self.tr.n.static.get_xyz)
        current=trace.kind[start:start+count]
        trace.kind[start:start+count]=torch.where(keep,current,0)
        return trace


def fixed_group_mask(e,raw):
    d=e.tr.raw(raw)
    with np.load(OBS/'frames'/f'{raw:03d}.npz',allow_pickle=False) as f:
        K=f['K'];inv=np.linalg.inv(f['world_to_camera']@f['world_from_local'][0])
    y,x=d['depth_valid'].nonzero(as_tuple=True)
    z=d['depth'][y,x]
    points=torch.stack(((x+.5-K[0,2])*z/K[0,0],(y+.5-K[1,2])*z/K[1,1],z),1)
    inv=z.new_tensor(inv)
    local=points@inv[:3,:3].T+inv[:3,3]
    point=(local-e.group_center)@e.group_basis
    lo,hi=e.measurement_boxes[0]
    good=((point>=lo)&(point<=hi)).all(1)
    result=torch.zeros_like(d['depth_valid'])
    result[y[good],x[good]]=True
    return result


fit_root.group_mask=fixed_group_mask
fit_joint.group_mask=fixed_group_mask
run_coupled.group_mask=fixed_group_mask

"""Measured-surface positive control for the zero-training rescue."""
import numpy as np
import torch

from rescue import RUN, Probe, build_anchors, measure_build, write


def panel(asset):
    xyz=asset['xyz']
    w=asset['box'][1]
    core=xyz[np.abs(xyz[:,1])<w/4]
    x=np.quantile(core[:,0],[.12,.68])+np.array([-.045,.045])
    z=np.quantile(core[:,2],[.12,.75])+np.array([-.035,.035])
    bounds=np.array([[x[0],-w/4,z[0]],[x[1],w/4,z[1]]],dtype=np.float32)
    return bounds


class Positive(Probe):
    def __init__(self):
        super().__init__()
        self.panel=panel(build_anchors())
        self.boxes=self.zero.new_tensor(self.panel[None])
        self.zero_volume=self.zero.new_zeros(1)
        self.retire=self.zero_volume+.60


def main():
    asset=build_anchors()
    e=Positive()
    inside=((asset['xyz']>=e.panel[0])&(asset['xyz']<=e.panel[1])).all(1)
    assert int(inside.sum())>=25
    for key in asset:
        if key!='box':
            asset[key]=asset[key][inside]
    write(RUN/'positive_contract.json',dict(box=e.panel.tolist(),retirement=.60,
          design='Controlled removal at a measured rear panel selected from build geometry alone.',
          eligible_anchors=int(inside.sum())))
    measure_build(e,asset,'positive')


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        main()

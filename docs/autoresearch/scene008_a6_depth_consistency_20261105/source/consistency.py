"""Full lower-body support crop with persistent, BUILD-registered row eligibility."""
from pathlib import Path
import sys
import numpy as np
import torch

RUN=Path(__file__).resolve().parent.parent
PREVIOUS=RUN.parent/'scene008_a6_heuristic_validation_20261105'
sys.path.insert(0,str(PREVIOUS/'source'))
import rescue
rescue.volume.RUN=RUN
rescue.volume.depth_order.takeover.calibrate.RUN=RUN
from rescue import Probe,BUILD,HELD,Trace,Window,RAWS,read,write,save
from interval_geometry import box_intervals,normal_interval_mass


class Consistency(Probe):
    def __init__(self):
        super().__init__()
        self.full_box=torch.stack((self.boxes[:,0].amin(0),self.boxes[:,1].amax(0)))[None]

    def frame(self,raw,state):
        trace=self.merged(raw,state,False)
        return RowWindow(self,trace,state)

    def reference(self,raw):
        rows=self.cache[raw][3]
        kind=torch.full((len(rows['features']),),3,device='cuda',dtype=torch.int32)
        out=Trace(self,raw,rows,kind,self.tr.black,2).base()
        ev=self.tr.evidence[raw]
        assert torch.allclose(out['alpha'][0],ev['lower_alpha'],atol=2e-6,rtol=0)
        return out['depth_sum'][0]/out['alpha'][0].clamp_min(1e-6)/self.tr.scale


class RowWindow(Window):
    def __init__(self,experiment,trace,state):
        super().__init__(experiment,trace,experiment.boxes)
        self.exp=experiment;self.state=state
        offset=0 if state=='black' else len(experiment.tr.n.static.get_xyz)
        row=self.ids[self.indices].long()-offset
        lineage=experiment.actor_lineage[trace.raw]
        assert bool(((row>=0)&(row<len(lineage))).all())
        self.owners=lineage[row].long()
        self.fraction=self.mass@self.mass.new_ones(len(experiment.boxes))
        row_ids=self.ids[self.indices].long()
        self.center_feature_maxabs_m=float(((trace.depth[row_ids]-trace.feature[row_ids,0])/experiment.tr.scale).abs().max()) if len(row_ids) else 0.
        assert self.center_feature_maxabs_m<2e-4,self.center_feature_maxabs_m

    def render(self,strength):
        t=self.trace
        alpha=self.alpha.clone()
        alpha[self.indices]*=1-(strength[self.owners]*self.fraction).clamp(0,1)
        alpha=torch.where(alpha>=1/255,alpha,0.)
        output=t.native.new_empty((len(t.pixels),9))
        k=t.experiment.kernel
        k.call(k.lib.compose_pass,len(t.pixels),t.offsets,self.identity,self.ids,alpha,t.color,
               t.feature,t.background,output)
        image=t.native.flatten(1).clone()
        image[:,t.pixels.long()]=output.T
        return t.package(image.reshape(9,t.h,t.w))

    def crop_modes(self,hit,rows,ray):
        """Camera-z mode after both inherited halfspace and physical crop."""
        t=self.trace;e=self.exp
        q=t.terms[rows].double();r=ray.double()
        x,y=r[:,0],r[:,1]
        precision=(q[:,0]*x*x+2*q[:,1]*x*y+2*q[:,2]*x+q[:,3]*y*y+2*q[:,4]*y+q[:,5]).clamp_min(1e-12)
        mode=(q[:,6]*x+q[:,7]*y+q[:,8])/precision;sigma=precision.rsqrt()
        origin,direction=Window.in_car(e,t,ray)
        lo,hi=box_intervals(origin,direction,e.full_box)
        lo,hi=lo[:,0].double(),hi[:,0].double()
        meta=t.feature[rows,6:].double()
        slope=(r*meta[:,10:13]).sum(1)
        clipped=(t.kind[rows]==1)&(slope.abs()*sigma>1e-10)
        threshold=meta[:,13]/torch.where(clipped,slope,torch.ones_like(slope))
        lo=torch.where(clipped&(slope>0),torch.maximum(lo,threshold),lo)
        hi=torch.where(clipped&(slope<0),torch.minimum(hi,threshold),hi)
        assert bool((hi>=lo-1e-6).all()),'nonempty crop expected for mass-positive hits'
        return mode.clamp(min=lo,max=hi)/e.tr.scale


def selected_hits(window,mask):
    """Joint parent prefixes, including contributors behind the native cutoff."""
    t=window.trace;counts=(t.offsets[1:]-t.offsets[:-1]).long();pixels=t.pixels.long()
    slots=torch.where(mask.flatten()[pixels]&(counts>0))[0]
    count=counts[slots];n=int(count.sum())
    bounds=torch.cat((count.new_zeros(1),count.cumsum(0)))
    ray=torch.arange(len(slots),device='cuda').repeat_interleave(count,output_size=n)
    hit=t.offsets[slots[ray]].long()+torch.arange(n,device='cuda')-bounds[ray]
    alpha=window.alpha[hit].double()
    logs=torch.cat((alpha.new_zeros(1),torch.log1p(-alpha).cumsum(0)))
    before=(logs[:-1]-logs[bounds[ray]]).exp()
    active=before*(1-alpha)>=.0001
    weight=before*alpha*active
    return dict(hit=hit,rows=window.ids[hit].long(),pixel=pixels[slots[ray]],
                slot=slots[ray],weight=weight)

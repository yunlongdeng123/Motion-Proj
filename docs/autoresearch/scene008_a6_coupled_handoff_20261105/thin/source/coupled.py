"""Measured surface groups and coupled old/new contribution transfer."""
from pathlib import Path
import os
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
import torch

RUN=Path(__file__).resolve().parent.parent
PARENT=RUN.parent/'scene008_a6_heuristic_validation_20261105'
sys.path.insert(0,str(PARENT/'source'))
import rescue
rescue.volume.RUN=RUN
rescue.volume.depth_order.takeover.calibrate.RUN=RUN
from rescue import Probe, BUILD, HELD, OBS, Trace, ObjectLayer, measured_surface, fuse
from depth_order import RAWS, read, write, save, merge_rows, select_rows, tag
from ray_support import padded
from volume import Window
from surface_groups import build_group
from heuristic_evaluate import panel_mask, local_metrics, seam
from street_metrics import compare

LEVELS=(.25,.5,.75,1.)
GAMMAS=(0.,.1,.2,.4,.6,.8,1.)
PRIMARY=.5
SIGMA_CAP=.035


def source_asset():
    with np.load(OBS/'a6.npz',allow_pickle=False) as f:
        keep=~np.isin(f['sample_raw_frame'],HELD)
        vals=[f['sample_'+k][keep] for k in ('xyz_local_m','rgb','rgb_valid','raw_frame','source_cloud_row')]
        a=fuse([vals])
        box=f['box_lwh_m'].copy()
    a['box_lwh_m']=box
    xyz,normal,sigma,rgb,_=measured_surface(a)
    eligible=a['rgb_valid']&(xyz[:,0]<-box[0]/4)&(xyz[:,2]<=xyz[:,2].min()+.55)
    indices=np.flatnonzero(eligible)
    count=[];minraw=[];maxraw=[]
    for k in indices:
        m=(a['sample_anchor']==k)&a['sample_rgb_valid']
        raw=np.unique(a['sample_raw_frame'][m])
        count.append(len(raw));minraw.append(raw.min());maxraw.append(raw.max())
        rgb[k]=np.median(a['sample_rgb'][m],axis=0)
    return dict(xyz=xyz[eligible],normal=normal[eligible],sigma=np.minimum(sigma[eligible],SIGMA_CAP),
                rgb=rgb[eligible],box=box,source=a['source_raw_frame'][eligible],
                rgb_frame_count=np.asarray(count),rgb_frame_min=np.asarray(minraw),rgb_frame_max=np.asarray(maxraw))


def jsonable(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,(np.integer,np.floating)):
        return value.item()
    if isinstance(value,dict):
        return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [jsonable(v) for v in value]
    return value


def create_group():
    out=build_group(source_asset())
    if out['status']!='PASS':
        write(RUN/'group_audit.json',jsonable(out))
        return None
    ledger=out.pop('ledger')
    ledger['center']=np.asarray(out['center'],np.float32)
    ledger['basis']=np.asarray(out['basis'],np.float32)
    ledger['bounds']=np.asarray(out['bounds'],np.float32)
    np.savez_compressed(RUN/'group_ledger.npz',**ledger)
    write(RUN/'group_audit.json',jsonable(out))
    return ledger


class ShellWindow(Window):
    @staticmethod
    def in_car(experiment,trace,camera_rays):
        o,d=Window.in_car(experiment,trace,camera_rays)
        return (o-experiment.group_center)@experiment.group_basis, d@experiment.group_basis


class Coupled(Probe):
    def __init__(self,ledger):
        super().__init__()
        self.group_center=self.zero.new_tensor(ledger['center'])
        self.group_basis=self.zero.new_tensor(ledger['basis'])
        self.boxes=self.zero.new_tensor(ledger['bounds'][None])
        self.zero_volume=self.zero.new_zeros(1)
        self.proxy=ObjectLayer(ledger['xyz'],ledger['normal'],ledger['sigma'],ledger['rgb'],
                               ledger['box'],generated=True).cuda()
        self.proxy_strength=self.zero.new_full((len(ledger['xyz']),),.98)
        self.bundles={}

    def merged(self,raw,state,use_proxy=True):
        n=self.tr.n
        actor=self.actor(raw,self.weight,self.extra)
        if state=='black':
            rows=merge_rows(actor,self.cache[raw][3])
            kind=torch.full((len(rows['features']),),3,device='cuda',dtype=torch.int32)
            kind[:len(actor['features'])]=torch.where(actor['features'][:,6]>.5,1,2)
            bg,env,channels=self.tr.black,None,3
        else:
            scene=self.assemble(raw,state=='delete')
            self.colourize(scene,self.road_colour)
            rows,_,env=scene
            first=len(n.static.get_xyz);count=len(self.cache[raw][0]['features'])
            tagged=tag(dict(actor,features=actor['features'][:,:2]),0)
            tagged['features'][:,6:]=actor['features'][:,6:]
            rows=merge_rows(merge_rows(select_rows(rows,slice(0,first)),tagged),
                            select_rows(rows,slice(first+count,None)))
            feat=rows['features']
            kind=torch.zeros(len(feat),device='cuda',dtype=torch.int32)
            kind[feat[:,2]==1]=2
            kind[(kind==2)&(feat[:,6]>.5)]=1
            kind[feat[:,5]==1]=3
            bg,channels=n.bg,4
        if use_proxy:
            d=self.tr.raw(raw)
            bands=1 if rows['shs'] is None else rows['shs'].shape[1]
            added=self.proxy.raster_rows(d['pose'],d['cam'],n.args,bands,rows['colors_precomp'] is not None)
            added['opacities']=self.proxy_strength[:,None]
            if state=='black':
                feat=added['features'].new_zeros((len(added['features']),5))
                feat[:,:2]=added['features'];feat[:,4]=1
                added['features']=padded(dict(added,features=feat))
            else:
                added=tag(added,0)
            rows=merge_rows(rows,added)
            kind=torch.cat((kind,kind.new_full((len(added['features']),),4)))
        return Trace(self,raw,rows,kind,bg,channels,env)

    def bundle(self,raw,state):
        trace=self.merged(raw,state)
        return Bundle(self,trace)


class Bundle:
    def __init__(self,experiment,trace):
        self.trace=trace
        self.window=ShellWindow(experiment,trace,experiment.boxes)
        self.proxy_hit=trace.kind[self.window.ids.long()]==4
        rows=torch.where(trace.kind==4)[0]
        lookup=trace.kind.new_full((len(trace.kind),),-1)
        lookup[rows]=torch.arange(len(rows),device='cuda',dtype=lookup.dtype)
        self.proxy_index=lookup[self.window.ids.long()]
        self.fraction=self.window.alpha.new_zeros(len(self.window.alpha))
        self.fraction[self.window.indices]=self.window.mass[:,0]
        self.cover_cache={}

    def alphas(self,retire,gamma):
        a=self.window.alpha.clone()
        a[self.window.indices]*=1-retire*self.window.mass[:,0]
        g=torch.as_tensor(gamma,device=a.device,dtype=a.dtype).flatten()
        scale=g if len(g)==1 else g[self.proxy_index[self.proxy_hit].long()]
        a[self.proxy_hit]*=scale
        return torch.where(a>=1/255,a,0.)

    def render(self,retire,gamma):
        t=self.trace;w=self.window
        a=self.alphas(retire,gamma)
        output=t.native.new_empty((len(t.pixels),9))
        k=t.experiment.kernel
        k.call(k.lib.compose_pass,len(t.pixels),t.offsets,w.identity,w.ids,a,t.color,
               t.feature,t.background,output)
        result=t.native.flatten(1).clone()
        result[:,t.pixels.long()]=output.T
        return t.package(result.reshape(9,t.h,t.w))

    def coverage(self,retire,gamma,mask):
        t=self.trace;w=self.window
        pixels=t.pixels.long()
        selected=mask.flatten()[pixels]
        counts=(t.offsets[1:]-t.offsets[:-1]).long()
        slots=torch.where(selected&(counts>0))[0]
        count=counts[slots]
        n=int(count.sum())
        image=t.native.new_zeros((t.h,t.w))
        if not n:
            return image,dict(native=0.,proxy=0.)
        bounds=torch.cat((count.new_zeros(1),count.cumsum(0)))
        ray=torch.arange(len(slots),device='cuda').repeat_interleave(count,output_size=n)
        hit=t.offsets[slots[ray]].long()+torch.arange(n,device='cuda')-bounds[ray]
        a=w.alpha[hit]*(1-retire*self.fraction[hit])
        g=torch.as_tensor(gamma,device=a.device,dtype=a.dtype).flatten()
        scale=g if len(g)==1 else g[self.proxy_index[hit].clamp_min(0).long()]
        a=torch.where(self.proxy_hit[hit],a*scale,a)
        a=torch.where(a>=1/255,a,0.).double()
        logs=torch.cat((a.new_zeros(1),torch.log1p(-a).cumsum(0)))
        before=(logs[:-1]-logs[bounds[ray]]).exp()
        active=before*(1-a)>=.0001
        # Remaining shell opacity and new-surface opacity are attributed in
        # the same actual hit order, with the native terminal condition.
        old=w.alpha[hit].double()*(1-retire)*self.fraction[hit].double()*(a>0)
        new=a*self.proxy_hit[hit]
        native=before*old*active
        proxy=before*new*active
        v=a.new_zeros(len(slots))
        v.index_add_(0,ray,native+proxy)
        image.flatten()[pixels[slots]]=v.float()
        return image,dict(native=float(native.sum()),proxy=float(proxy.sum()))


def group_mask(e,raw):
    d=e.tr.raw(raw)
    with np.load(OBS/'frames'/f'{raw:03d}.npz',allow_pickle=False) as f:
        K=f['K'];inv=np.linalg.inv(f['world_to_camera']@f['world_from_local'][0])
    y,x=d['depth_valid'].nonzero(as_tuple=True)
    z=d['depth'][y,x]
    points=torch.stack(((x+.5-K[0,2])*z/K[0,0],(y+.5-K[1,2])*z/K[1,1],z),1)
    inv=z.new_tensor(inv)
    local=points@inv[:3,:3].T+inv[:3,3]
    loc=(local-e.group_center)@e.group_basis
    lo,hi=e.boxes[0]
    good=((loc>=lo)&(loc<=hi)).all(1)
    result=torch.zeros_like(d['depth_valid'])
    result[y[good],x[good]]=True
    return result


if __name__=='__main__':
    create_group()

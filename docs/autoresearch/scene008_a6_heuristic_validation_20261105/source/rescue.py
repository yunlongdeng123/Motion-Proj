"""Design and build the measured-anchor rescue for the matched retirement probe."""
from pathlib import Path
import sys
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'

import numpy as np
import torch

RUN=Path(__file__).resolve().parent.parent
VOLUME=RUN.parent/'scene008_a6_volume_carve_20261105'
INSTANCES=RUN.parent/'scene008_instances_20261105_r1'
sys.path[:0]=[str(VOLUME/'source'),str(INSTANCES/'source')]
import volume
volume.RUN=RUN
volume.depth_order.takeover.calibrate.RUN=RUN
from volume import Experiment as Volume, Window
from depth_order import RAWS, PILOTS, Trace, read, write, save, merge_rows, select_rows, tag, metric
from object_layer import ObjectLayer, measured_surface
from object_observations import fuse

HELD=(89,95,101,105,113,121,125,130,137)
BUILD=tuple(r for r in RAWS if r not in HELD)
RETIRE=.25
MIN_LOSS=.005
MIN_RGB_DAMAGE=.002
Z_TOL=.12
MAX_ALPHA=.35
MIN_TRANSMISSION=.05
OBS=INSTANCES/'observations'


def build_anchors():
    with np.load(OBS/'a6.npz',allow_pickle=False) as f:
        raw=f['sample_raw_frame']
        keep=~np.isin(raw,HELD)
        vals=[f['sample_'+k][keep] for k in ('xyz_local_m','rgb','rgb_valid','raw_frame','source_cloud_row')]
        asset=fuse([vals])
        box=f['box_lwh_m'].copy()
    asset['box_lwh_m']=box
    xyz,normals,sigma,rgb,_=measured_surface(asset)
    eligible=asset['rgb_valid']&(xyz[:,0]<-box[0]/4)&(xyz[:,2]<=xyz[:,2].min()+.55)
    # Use the median of measured BUILD colors in each eligible anchor voxel.
    for k in np.flatnonzero(eligible):
        good=(asset['sample_anchor']==k)&asset['sample_rgb_valid']
        assert good.any()
        rgb[k]=np.median(asset['sample_rgb'][good],axis=0)
    assert len(xyz)>0 and eligible.any() and not np.isin(asset['source_raw_frame'],HELD).any()
    return dict(xyz=xyz[eligible],normal=normals[eligible],sigma=np.clip(.8*sigma[eligible],.014,.05),
                rgb=rgb[eligible],box=box,source=asset['source_raw_frame'][eligible])


class Probe(Volume):
    def __init__(self):
        super().__init__()
        self.proxy=None
        self.proxy_strength=None
        self.retire=self.zero_volume+RETIRE

    def load_proxy(self, ledger):
        self.proxy=None
        if len(ledger['xyz']):
            self.proxy=ObjectLayer(ledger['xyz'],ledger['normal'],ledger['sigma'],ledger['rgb'],
                                   ledger['box'],generated=True).cuda()
            self.proxy_strength=self.zero.new_tensor(ledger['opacity'])

    def merged(self, raw, state, use_proxy):
        n=self.tr.n
        actor=self.actor(raw,self.weight,self.extra)
        if state=='black':
            rows=merge_rows(actor,self.cache[raw][3])
            kind=torch.full((len(rows['features']),),3,device='cuda',dtype=torch.int32)
            kind[:len(actor['features'])]=torch.where(actor['features'][:,6]>.5,1,2)
            channels=2
            bg=self.tr.black
            env=None
        else:
            base=self.assemble(raw,state=='delete')
            self.colourize(base,self.road_colour)
            rows,_,env=base
            first=len(n.static.get_xyz)
            count=len(self.cache[raw][0]['features'])
            actor_tagged=tag(dict(actor,features=actor['features'][:,:2]),0)
            actor_tagged['features'][:,6:]=actor['features'][:,6:]
            rows=merge_rows(merge_rows(select_rows(rows,slice(0,first)),actor_tagged),
                            select_rows(rows,slice(first+count,None)))
            feat=rows['features']
            kind=torch.zeros(len(feat),device='cuda',dtype=torch.int32)
            kind[feat[:,2]==1]=2
            kind[(kind==2)&(feat[:,6]>.5)]=1
            kind[feat[:,5]==1]=3
            channels=4
            bg=n.bg
        if use_proxy and self.proxy is not None:
            d=self.tr.raw(raw)
            bands=1 if rows['shs'] is None else rows['shs'].shape[1]
            added=self.proxy.raster_rows(d['pose'],d['cam'],n.args,bands,rows['colors_precomp'] is not None)
            added['opacities']=self.proxy_strength[:,None]
            if state=='black':
                from ray_support import padded
                feat=added['features'].new_zeros((len(added['features']),4))
                feat[:,:2]=added['features']
                added['features']=padded(dict(added,features=feat))
            else:
                added=tag(added,0)
            rows=merge_rows(rows,added)
            # Measured rescue rows join depth sorting. They are outside the retirement operation.
            kind=torch.cat((kind,kind.new_zeros(len(added['features']))))
        return Trace(self,raw,rows,kind,bg,channels,env)

    def raster_pair(self, raw, state, use_proxy):
        trace=self.merged(raw,state,use_proxy)
        window=Window(self,trace,self.boxes)
        parent=trace.base()
        cut=window.render(self.retire)
        assert not any(self.parity(parent,window.render(self.zero_volume)).values())
        return parent,cut,trace,window


def front_response(trace, window, pixels, depth, strength):
    slots=torch.searchsorted(trace.pixels,pixels.to(trace.pixels.dtype))
    inside=(slots<len(trace.pixels))
    if inside.any():
        inside[inside.clone()]=trace.pixels[slots[inside]].long()==pixels[inside]
    transmission=depth.new_zeros(len(pixels))
    foreground=depth.new_zeros((len(pixels),3))
    if not inside.any():
        return transmission,foreground
    index=slots[inside]
    starts=trace.offsets[index].long()
    count=(trace.offsets[index+1]-trace.offsets[index]).long()
    n=int(count.sum())
    if not n:
        transmission[inside]=1
        return transmission,foreground
    ray=torch.arange(len(count),device='cuda').repeat_interleave(count,output_size=n)
    bounds=torch.cat((count.new_zeros(1),count.cumsum(0)))
    local=torch.arange(n,device='cuda')-bounds[ray]
    hit=starts[ray]+local
    rows=window.ids[hit].long()
    before=trace.depth[rows]<depth[inside][ray]
    alpha=window.alpha.clone()
    value=alpha[window.indices]*(1-(window.mass@strength).clamp(0,1))
    alpha[window.indices]=torch.where(value>=1/255,value,0.)
    alpha=alpha[hit].double()
    logs=torch.cat((alpha.new_zeros(1),torch.log1p(-alpha).cumsum(0)))
    t=(logs[:-1]-logs[bounds[ray]]).exp()
    accepted=(t*(1-alpha))>=.0001
    weight=t*alpha*accepted*before
    c=trace.color.reshape(-1,3)[rows].double()
    cf=c.new_zeros((len(count),3))
    cf.index_add_(0,ray,weight[:,None]*c)
    log_end=alpha.new_zeros(len(count))
    log_end.index_add_(0,ray,torch.where(before,torch.log1p(-alpha),0.))
    transmission[inside]=log_end.exp().float()
    foreground[inside]=cf.float()
    return transmission,foreground


def measure_build(experiment, asset, name='rescue'):
    xyz=experiment.zero.new_tensor(asset['xyz'])
    opacity=[]
    views=[]
    for raw in BUILD:
        base,cut,trace,window=experiment.raster_pair(raw,'black',False)
        with np.load(OBS/'frames'/f'{raw:03d}.npz',allow_pickle=False) as f:
            xyz2=f['world_to_camera']@f['world_from_local'][0]
            m=xyz.new_tensor(xyz2)
            K=xyz.new_tensor(f['K'])
        d=experiment.tr.raw(raw)
        cam=xyz@m[:3,:3].T+m[:3,3]
        z=cam[:,2]
        im=cam@K.T
        xy=(im[:,:2]/z.clamp_min(.01)[:,None]).floor().long()
        x,y=xy.unbind(1)
        valid=(x>=0)&(x<trace.w)&(y>=0)&(y<trace.h)&(z>.1)
        pix=(y.clamp(0,trace.h-1)*trace.w+x.clamp(0,trace.w-1)).long()
        valid&=d['obs'].flatten()[pix]&d['depth_valid'].flatten()[pix]
        valid&=(z-d['depth'].flatten()[pix]).abs()<=Z_TOL
        old_native=base['alpha'][0]-base['extra'][1]
        new_native=cut['alpha'][0]-cut['extra'][1]
        loss=(old_native-new_native).flatten()[pix]
        target=d['rgb'].permute(1,2,0).reshape(-1,3)[pix]
        old_rgb=base['rgb'].permute(1,2,0).reshape(-1,3)[pix]
        new_rgb=cut['rgb'].permute(1,2,0).reshape(-1,3)[pix]
        damage=(new_rgb-target).abs().mean(1)-(old_rgb-target).abs().mean(1)
        score=valid&((loss>MIN_LOSS)|(damage>MIN_RGB_DAMAGE))
        g=xyz.new_zeros(len(xyz))
        t,front=front_response(trace,window,pix[score],z[score]*experiment.tr.scale,experiment.retire)
        measured=xyz.new_tensor(asset['rgb'])[score]
        direction=t[:,None]*measured-(new_rgb[score]-front)
        residual=target[score]-new_rgb[score]
        value=((direction*residual).sum(1)/direction.square().sum(1).clamp_min(1e-8)).clamp(0,MAX_ALPHA)
        improves=(new_rgb[score]+value[:,None]*direction-target[score]).abs().mean(1)<(new_rgb[score]-target[score]).abs().mean(1)
        g[score]=torch.where((t>=MIN_TRANSMISSION)&improves,value,0.)
        opacity.append(g.cpu().numpy())
        views.append(dict(raw=raw,measured_matches=int(valid.sum()),loss_matches=int(score.sum()),
                          feasible=int((g>0).sum())))
    proposed=np.stack(opacity)
    count=(proposed>0).sum(0)
    use=count>=2
    ledger={k:(v[use] if k!='box' else v) for k,v in asset.items()}
    ledger['opacity']=np.asarray([np.median(v[v>0]) for v in proposed[:,use].T],dtype=np.float32)
    ledger['build_witness_count']=count[use].astype(np.int16)
    ledger['build_witness_raw']=np.asarray(BUILD,dtype=np.int16)
    ledger['build_witness_mask']=(proposed[:,use]>0)
    np.savez_compressed(RUN/f'{name}_ledger.npz',**ledger)
    write(RUN/('build_evidence.json' if name=='rescue' else 'positive_build.json'),dict(build=BUILD,held=HELD,anchors=len(xyz),selected=int(use.sum()),views=views,
          criterion='Build-only measured rear anchors with coverage or raw-RGB loss; closed-form observed-color alpha, prefix visibility, >=2 build views'))
    print('RESCUE_BUILD',name,len(xyz),int(use.sum()),sum(r['loss_matches'] for r in views),flush=True)
    return ledger


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        measure_build(Probe(),build_anchors())

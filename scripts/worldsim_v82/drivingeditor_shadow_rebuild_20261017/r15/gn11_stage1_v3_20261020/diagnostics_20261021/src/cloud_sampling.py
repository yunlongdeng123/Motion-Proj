"""冻结检查点的 CPU 几何采样；不创建优化器，不写原始模型。"""
import argparse, base64, gc, json, math, os, sys, zlib
from pathlib import Path
from collections import OrderedDict
import numpy as np
from backend import Adapter, CAMPAIGN, clean, digest

ACTORS=(5,26,18,6,37)
V1=CAMPAIGN.parent/'r15_baseline_owner5_20261018/runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k'

def pack(arrays):
    parts=[];records={};offset=0
    for key,value in arrays.items():
        value=np.ascontiguousarray(value);raw=value.tobytes()
        records[key]=dict(dtype=value.dtype.str,shape=value.shape,offset=offset,bytes=len(raw))
        parts.append(raw);offset+=len(raw)
    return dict(arrays=records,encoding='zlib+base64-little-endian',data=base64.b64encode(zlib.compress(b''.join(parts),4)).decode())

def rate(value):
    if isinstance(value,bool):raise ValueError('rate must be numeric')
    value=float(value)
    if not math.isfinite(value) or not 0<=value<=100:raise ValueError('sampling rate must be 0..100 percent')
    return value

def hashes(rows):
    x=np.asarray(rows,dtype=np.uint32).copy()
    x^=x>>np.uint32(16);x*=np.uint32(0x7feb352d);x^=x>>np.uint32(15)
    x*=np.uint32(0x846ca68b);x^=x>>np.uint32(16)
    return x

def ranked_sample(rows,percent):
    count=min(len(rows),math.ceil(len(rows)*percent/100))
    if count==0:return np.empty(0,np.int64)
    if count==len(rows):return rows.copy()
    order=np.lexsort((rows,hashes(rows)))
    return rows[order[:count]]

class Sampler:
    def __init__(self, adapter=None):
        self.adapter=adapter or Adapter();self.current=None;self.model=None;self.orders={};self.labels=OrderedDict()
        meta=self.adapter.meta
        engineering=Path(meta['original_engineering'])
        receipt=json.loads((engineering/'matched5_input_receipt.json').read_text())
        self.tracks=dict(zip(ACTORS,receipt['track_ids']))
        self.source=engineering.parent/'inputs/scene0008_089_137'
        self.metadata=self.make_metadata()

    def make_metadata(self):
        # bundle 中车框来自标注顺序，不能把其序号当 actor ID。
        A=self.adapter.A;scale=self.adapter.scale;frames={}
        for raw in range(89,138):
            fid=f'0008{raw:03d}';pose=np.loadtxt(self.source/'pose'/f'{fid}.txt')
            labels=json.loads((self.source/'annotations'/f'{fid}.json').read_text())['laser_labels']
            found={}
            for label in labels:
                actors=[a for a,t in self.tracks.items() if t==label['id']]
                if not actors:continue
                b=label['box'];theta=b['heading'];c,s=np.cos(theta),np.sin(theta)
                local=np.array([b['center_x'],b['center_y'],b['center_z']])
                world=pose[:3,:3]@local+pose[:3,3]
                found[actors[0]]=dict(actor=actors[0],center=(A[:3,:3]@world+A[:3,3]).tolist(),
                    half=(np.array([b['length'],b['width'],b['height']])/2*scale).tolist(),
                    axes=(A[:3,:3]@pose[:3,:3]@np.array([[c,-s,0],[s,c,0],[0,0,1]])/scale).tolist())
            boxes=[]
            for box in self.adapter.meta['frames'][fid]['selected_boxes_model']:
                actor=min(found,key=lambda a:np.linalg.norm(np.asarray(found[a]['center'])-box['center']))
                if np.linalg.norm(np.asarray(found[actor]['center'])-box['center'])>1e-4:raise ValueError('vehicle box identity mismatch')
                boxes.append(dict(box,actor=actor))
            if len(boxes)!=5 or len({b['actor'] for b in boxes})!=5:raise ValueError('five vehicle identities required')
            frames[str(raw)]=boxes
        return dict(version=1,actors=list(ACTORS),vehicles=frames,default_static_rate=2,default_dynamic_rate=2,default_vehicle_rate=10,
            single_vehicle_definition='current camera mask dilated5px, all depths; or 3D box expanded0.5m; spatial region, not permanent Gaussian ownership')

    def get_model(self,arm):
        if arm not in ('full','real_only','prior_only','ownership_only','gn11_v1'):raise ValueError('unknown arm')
        if self.current==arm:return self.model
        self.model=None;self.orders={};gc.collect()
        if arm!='gn11_v1':
            self.model=self.adapter.model(arm)
        else:
            self.adapter.models.clear()
            import torch
            config=json.loads((V1/'effective_config.json').read_text());branches=[]
            binding=json.loads(Path('/root/autodl-tmp/gsdiag_gpu_20261021_r2/v1_binding.json').read_text())
            for name,prefix in [('static','chkpnt'),('dynamic','dynamic_chkpnt')]:
                p=V1/f'{prefix}30000.pth';expected=binding['checkpoints'][name]
                if p.stat().st_size!=expected['bytes'] or digest(p)!=expected['sha256']:raise ValueError('V1 checkpoint binding mismatch')
                payload,step=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True)
                if step!=30000:raise ValueError('V1 must be terminal30K')
                xyz,dc,rest,logs,q,logit,t,logt,vel=[payload[i].detach().numpy() for i in range(1,10)]
                branches.append(dict(name=name,sha=expected['sha256'],xyz=xyz,dc=dc,scale=np.exp(logs),q=q,
                    opacity=1/(1+np.exp(-np.clip(logit[:,0],-80,80))),t=t[:,0],sigma_t=np.exp(logt[:,0]),vel=vel,
                    cycle=float(config['cycle']),road=np.full(len(xyz),-1,np.int64)))
            self.model=dict(arm=arm,branches=branches)
        self.current=arm
        return self.model

    def query(self,request):
        arm=request['arm'];raw=request.get('raw',102)
        if type(raw) is not int or not 89<=raw<=137:raise ValueError('raw outside89..137')
        sr=rate(request.get('static_rate',2));dr=rate(request.get('dynamic_rate',2));vr=rate(request.get('vehicle_rate',10))
        vehicle=request.get('vehicle','none');scope=request.get('vehicle_scope','projection')
        if vehicle!='none':
            if isinstance(vehicle,bool):raise ValueError('actor ID required')
            vehicle=int(vehicle)
            if vehicle not in ACTORS:raise ValueError('unknown actor')
        if scope not in ('projection','box'):raise ValueError('unknown vehicle scope')
        model=self.get_model(arm);cam=self.adapter.camera(raw);arrays={};branches=[];stats={}
        box=next((b for b in self.metadata['vehicles'][str(raw)] if b['actor']==vehicle),None)
        mask=None
        if box and scope=='projection':
            from PIL import Image
            from scipy.ndimage import binary_dilation
            if raw not in self.labels:
                self.labels[raw]=np.asarray(Image.open(self.adapter.meta['frames'][f'0008{raw:03d}']['label_path']))
                while len(self.labels)>4:self.labels.popitem(last=False)
            mask=binary_dilation(self.labels[raw]==ACTORS.index(vehicle)+1,iterations=5)
        for b in model['branches']:
            name=b['name'];n=len(b['xyz']);road=b['road']>=0;ordinary=np.flatnonzero(~road)
            percent=sr if name=='static' else dr
            if name not in self.orders:self.orders[name]=ordinary[np.lexsort((ordinary,hashes(ordinary)))]
            base=self.orders[name][:math.ceil(len(ordinary)*percent/100)]
            target=np.empty(0,np.int64);extra=np.empty(0,np.int64)
            if box:
                for start in range(0,n,100000):
                    end=min(n,start+100000);delta=cam['timestamp']-b['t'][start:end]
                    m=np.exp(-.5*(delta/b['sigma_t'][start:end])**2);active=(m>.05)&~road[start:end]
                    omega=2*np.pi/b['cycle'];xyz=b['xyz'][start:end]+b['vel'][start:end]*(np.sin(delta*omega)/omega)[:,None]
                    if scope=='box':
                        local=(xyz-box['center'])@np.asarray(box['axes'])
                        within=(np.abs(local)<=np.asarray(box['half'])+.5*self.adapter.scale).all(1)
                    else:
                        cp=xyz@cam['W'][:3,:3].T+cam['W'][:3,3]
                        uv=cp[:,:2]/np.maximum(cp[:,2,None],1e-8)*[cam['K'][0,0],cam['K'][1,1]]+[cam['K'][0,2],cam['K'][1,2]]
                        px=np.rint(uv).astype(np.int64);valid=(cp[:,2]>.02)&(px[:,0]>=0)&(px[:,0]<480)&(px[:,1]>=0)&(px[:,1]<320)
                        within=np.zeros(len(xyz),bool);within[valid]=mask[px[valid,1],px[valid,0]]
                    target=np.r_[target,np.flatnonzero(active&within)+start]
                extra=ranked_sample(target,max(vr,percent))
            indices=np.unique(np.r_[base,extra,np.flatnonzero(road)])
            prefix=name+'_';role=np.where(road[indices],1,0) if name=='static' else np.full(len(indices),2)
            for key,a in [('row',indices.astype(np.int32)),('xyz',b['xyz'][indices]),('sigma',b['scale'][indices]),('q',b['q'][indices]),
                ('velocity',b['vel'][indices]),('t',b['t'][indices]),('sigma_t',b['sigma_t'][indices]),('opacity',b['opacity'][indices]),
                ('color',np.maximum(.5+.28209479177387814*b['dc'][indices,0],0)),('role',role.astype(np.uint8)),('road_id',b['road'][indices].astype(np.int32))]:arrays[prefix+key]=a
            arrays[prefix+'cycle']=np.array([b['cycle']],np.float32)
            branches.append(dict(branch=name,total=n,displayed=len(indices),checkpoint_sha256=b['sha']))
            stats[name]=dict(total=n,ordinary_total=len(ordinary),road_total=int(road.sum()),base_sampled=len(base),sampled=len(indices),vehicle_candidates=len(target),vehicle_sampled=int(np.isin(target,indices).sum()))
        return dict(**pack(arrays),branches=branches,sampling=dict(static_rate=sr,dynamic_rate=dr,vehicle=vehicle,vehicle_rate=vr,vehicle_scope=scope,raw=raw,stats=stats,
            roads_all_retained=True,gpu_used=False,optimizer_updates=0,selection='nested deterministic uint32 rank; vehicle union never removes baseline rows'))

def main():
    sampler=Sampler()
    for line in sys.stdin:
        try:
            r=json.loads(line);v=sampler.metadata if r.get('action')=='metadata' else sampler.query(r)
            print(json.dumps(clean(v),separators=(',',':'),allow_nan=False),flush=True)
        except Exception as e:print(json.dumps(dict(error=type(e).__name__+': '+str(e),gpu_used=False,optimizer_updates=0)),flush=True)

if __name__=='__main__':main()

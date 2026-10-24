"""Road-only Gaussian teacher for deliberately occluded interiors; CPU reference.

The teacher infers an appearance prior. Its output is NOT observed RGB/depth GT.
No original/static checkpoint, target-car RGB or learned external model is used.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from cpu_plane_propagation import measured_maps,rays,intersect,project

CELL=.1
POLICY=dict(cell_m=CELL,methods=['nearest','isotropic','road_gaussian'],
    gaussian_sigma_m=[6.,.25],isotropic_sigma_m=1.5,
    max_longitudinal_m=18.,max_lateral_m=1.,
    source='same-frame measured small-triangle ground, outside all vehicle guards; per-frame/world-cell median',
    aggregation='equal weight per frame then temporal median, no target vehicle pixels',
    geometry='local weighted plane fitted from measured ground donors, never static rendered depth',
    pseudo_kind='inferred_road_prior_NOT_observed_ground_truth',
    calibration='spatial full-hole withdrawal from ALL frames including geometry and color; eroded interior scored separately',
    holes='centers x=12,20,28,36,44; y=-2,2,4,6; rectangle 8m longitudinal x 2m lateral; inner core eroded 1m/0.4m',
    split='x12/28/44 development, x20/36 validation; fixed before results, same scene not independent scene',
    inference_support='at least 8 donor cells within 18m longitudinal and 1m lateral, else isotropic fallback marked separately',
    photo_model='normalized anisotropic Gaussian mixture, before/after directional agreement gives uncertainty',
    disclaimer='not GaussianPro GPU reproduction; ray-conditioned road birth + prior targets; no optimizer updates')

def group_median(keys,values):
    unique,inv=np.unique(keys,axis=0,return_inverse=True)
    order=np.argsort(inv,kind='stable');start=np.r_[0,np.cumsum(np.bincount(inv))]
    out=np.array([np.median(values[order[start[i]:start[i+1]]],axis=0) for i in range(len(unique))])
    return unique,out,np.diff(start)

def build_atlas(inputs,out):
    """Every retained color is an observed unmasked road pixel, never a car pixel."""
    meta=json.loads((inputs/'manifest.json').read_text());plane=np.asarray(meta['ground_plane'])
    plane/=np.linalg.norm(plane[:3]);records=[];ids=[];nraw=0
    for i,fid in enumerate(meta['frame_order']):
        f=measured_maps(dict(np.load(inputs/f'{fid}.npz')),plane)
        yy,xx=np.where(f['measured_valid']);uv=np.c_[xx,yy]
        o,r=rays(f,uv);xyz=o+r*f['depth'][yy,xx,None]
        ok=(xyz[:,0]>=3)&(xyz[:,0]<=85)&(np.abs(xyz[:,1])<=8)
        xyz=xyz[ok];rgb=f['rgb'][yy[ok],xx[ok]];nraw+=len(xyz)
        keys=np.floor(xyz[:,:2]/CELL).astype(np.int32)
        kk,vv,cc=group_median(keys,np.c_[xyz[:,2],rgb])
        records.append(np.c_[kk,vv]);ids.append(np.full(len(kk),i,np.int16))
        print(json.dumps(dict(stage='atlas',frame=fid,cells=len(kk))),flush=True)
    rec=np.concatenate(records);fi=np.concatenate(ids)
    kk,vv,cc=group_median(rec[:,:2].astype(np.int32),rec[:,2:])
    # Robust dispersion includes illumination/view dependence and noisy source samples.
    lut={tuple(k):i for i,k in enumerate(kk)};idx=np.array([lut[tuple(k)] for k in rec[:,:2]])
    _,dev,_=group_median(rec[:,:2].astype(np.int32),np.abs(rec[:,2:]-vv[idx]))
    np.savez_compressed(out,keys=kk,xy=(kk+.5)*CELL,z=vv[:,0],rgb=vv[:,1:],
        source_frames=cc,rgb_mad=dev[:,1:],z_mad=dev[:,0],record_frame=fi,
        record_keys=rec[:,:2].astype(np.int32),record_z=rec[:,2],record_rgb=rec[:,3:],
        ground_plane=plane,raw_pixel_count=np.array(nraw))
    return dict(np.load(out))

class Teacher:
    def __init__(self,atlas,keep=None):
        if keep is None:keep=np.ones(len(atlas['xy']),bool)
        self.xy=atlas['xy'][keep];self.rgb=atlas['rgb'][keep];self.z=atlas['z'][keep]
        self.mad=atlas['rgb_mad'][keep].mean(1);self.count=atlas['source_frames'][keep]
        self.tree=cKDTree(self.xy);self.long_tree=cKDTree(self.xy/np.array([6.,.25]))

    def predict(self,q,method='road_gaussian'):
        q=np.asarray(q);N=len(q);rgb=np.empty((N,3));z=np.empty(N);unc=np.empty(N)
        support=np.empty(N,bool);distance=np.empty(N);normal=np.empty((N,3));directional=np.empty(N,bool)
        fallback=np.zeros(N,bool)
        for lo in range(0,N,1024):
            qq=q[lo:lo+1024];n=len(qq)
            if method=='road_gaussian':
                _,ii=self.long_tree.query(qq/np.array([6.,.25]),k=min(256,len(self.xy)))
            else:_,ii=self.tree.query(qq,k=min(128,len(self.xy)))
            if ii.ndim==1:ii=ii[:,None]
            dx=self.xy[ii]-qq[:,None];dist=np.linalg.norm(dx,axis=2)
            nearest,_=self.tree.query(qq)
            if method=='nearest':
                ww=np.zeros(ii.shape);ww[:,0]=1.;valid=np.ones(n,bool)
            else:
                scale=np.array([6.,.25]) if method=='road_gaussian' else np.array([1.5,1.5])
                ww=np.exp(-.5*np.sum((dx/scale)**2,2))
                eligible=(np.abs(dx[...,0])<=18)&(np.abs(dx[...,1])<=1) if method=='road_gaussian' else dist<=12
                ww*=eligible;valid=eligible.sum(1)>=8
                # Explicit fallback prevents "all uncertain means no interior supervision".
                bad=~valid
                if np.any(bad):
                    dd,jj=self.tree.query(qq[bad],k=ii.shape[1]);ii[bad]=jj;dx[bad]=self.xy[jj]-qq[bad,None]
                    ww[bad]=np.exp(-.5*(dd/3)**2);fallback[lo:lo+n]=bad
                ww*=np.minimum(self.count[ii],5)/5 # do not let view count dominate indefinitely
            ww/=np.maximum(ww.sum(1,keepdims=True),1e-30)
            color=np.sum(ww[...,None]*self.rgb[ii],1)
            # Fit a low curvature local surface from the same donors; ridge regularizes slopes.
            A=np.concatenate([np.ones((*ii.shape,1)),dx],2)
            lhs=np.einsum('nki,nk,nkj->nij',A,ww,A)+np.diag([1e-9,.1,.1])[None]
            rhs=np.einsum('nki,nk,nk->ni',A,ww,self.z[ii])
            beta=np.linalg.solve(lhs,rhs[...,None])[...,0]
            norm=np.c_[-beta[:,1:],np.ones(n)];norm/=np.linalg.norm(norm,axis=1)[:,None]
            side=[];sidegood=[]
            for sign in [-1,1]:
                sw=ww*(dx[...,0]*sign>0);ss=sw.sum(1);sidegood.append(ss>.1)
                side.append(np.sum(sw[...,None]*self.rgb[ii],1)/np.maximum(ss[:,None],1e-20))
            two=sidegood[0]&sidegood[1]
            disagreement=np.where(two,np.abs(side[0]-side[1]).mean(1),.08)
            dispersion=np.sum(ww*np.abs(self.rgb[ii]-color[:,None]).mean(2),1)
            uncertainty=disagreement+np.sum(ww*self.mad[ii],1)+.25*dispersion+.003*nearest
            sl=slice(lo,lo+n);rgb[sl]=color;z[sl]=beta[:,0];normal[sl]=norm
            unc[sl]=uncertainty;support[sl]=valid;distance[sl]=nearest;directional[sl]=two
        # Nonzero optimization weights are NOT probabilities or labels of known-correct pixels.
        confidence=np.clip(np.exp(-unc/.06)/(1+distance/5),.05,.9)
        return dict(rgb=rgb,z=z,normal=normal,uncertainty=unc,loss_weight=confidence,
            support=support,nearest_donor_m=distance,two_sided=directional,fallback=fallback)

def metric(pred,ref):
    if not len(ref):return None
    err=np.abs(pred-ref).mean(1);mse=np.mean((pred-ref)**2)
    return dict(n=len(ref),mae=float(err.mean()),psnr=float(-10*np.log10(max(mse,1e-12))),
        p90_abs=float(np.quantile(err,.9)),within_005=float(np.mean(err<=.05)))

def validate(atlas,out):
    rows=[];heldout=[];xy=atlas['xy']
    for cx in [12,20,28,36,44]:
        for cy in [-2,2,4,6]:
            # Withdraw all measurements (all temporal records aggregated into cells), no leakage.
            hole=(np.abs(xy[:,0]-cx)<4)&(np.abs(xy[:,1]-cy)<1)
            test=hole&(atlas['source_frames']>=2);core=test&(np.abs(xy[:,0]-cx)<3)&(np.abs(xy[:,1]-cy)<.6)
            if test.sum()<100 or core.sum()<40:continue
            teacher=Teacher(atlas,~hole);ids=np.flatnonzero(test);incore=core[ids]
            split='validation' if cx in [20,36] else 'development'
            row=dict(center=[cx,cy],split=split,withdrawn=int(hole.sum()),core_count=int(core.sum()),methods={})
            for method in POLICY['methods']:
                p=teacher.predict(xy[ids],method);item=dict(all=metric(p['rgb'],atlas['rgb'][ids]),
                    core=metric(p['rgb'][incore],atlas['rgb'][ids][incore]),
                    core_depth_mae=float(np.abs(p['z'][incore]-atlas['z'][ids][incore]).mean()),
                    uncertainty_mean=float(p['uncertainty'][incore].mean()))
                row['methods'][method]=item
                heldout.append(dict(split=split,method=method,pred=p['rgb'][incore],ref=atlas['rgb'][ids][incore],
                    uncertainty=p['uncertainty'][incore],hole=np.full(incore.sum(),len(rows))))
            rows.append(row);print(json.dumps(dict(stage='hole',**row)),flush=True)
    agg={}
    for split in ['development','validation']:
        agg[split]={}
        for method in POLICY['methods']:
            part=[x for x in heldout if x['split']==split and x['method']==method]
            pred=np.concatenate([x['pred'] for x in part]);ref=np.concatenate([x['ref'] for x in part]);unc=np.concatenate([x['uncertainty'] for x in part])
            agg[split][method]=metric(pred,ref)
            agg[split][method]['hole_macro_mae']=float(np.mean([r['methods'][method]['core']['mae'] for r in rows if r['split']==split]))
            np.savez_compressed(out/f'heldout_{split}_{method}.npz',pred=pred,ref=ref,uncertainty=unc)
    return dict(holes=rows,aggregate=agg,note='same-scene spatial withdrawal, not unseen-scene test; both color AND geometry inside every hole removed from all source frames')

def infer_frames(inputs,atlas,out):
    meta=json.loads((inputs/'manifest.json').read_text());plane=atlas['ground_plane'];teacher=Teacher(atlas)
    rows=[];critical=[];births=[];birthrgb=[];birthnormal=[]
    (out/'frames').mkdir()
    for fid in meta['frame_order']:
        f=dict(np.load(inputs/f'{fid}.npz'));yy,xx=np.where(f['actor_labels']>0);uv=np.c_[xx,yy];o,r=rays(f,uv)
        xyz,t=intersect(o,r,np.broadcast_to(-plane[3]*plane[:3],(len(uv),3)),np.broadcast_to(plane[:3],(len(uv),3)))
        domain=(t>.1)&(t<120)&(xyz[:,0]>=3)&(xyz[:,0]<=85)&(np.abs(xyz[:,1])<=8)
        ids=np.flatnonzero(domain);pts=xyz[ids]
        # Iterate surface/ray intersection. Predictions depend on measured road donors only.
        for _ in range(3):
            p=teacher.predict(pts[:,:2]);pp=np.c_[pts[:,:2],p['z']]
            pts,depth=intersect(o,r[ids],pp,p['normal'])
        p=teacher.predict(pts[:,:2]);valid=np.isfinite(pts).all(1)&(depth>.1)&(depth<120)&(pts[:,0]>=3)&(pts[:,0]<=85)&(np.abs(pts[:,1])<=8)
        ids=ids[valid];pts=pts[valid];depth=depth[valid];p={k:v[valid] for k,v in p.items()}
        image=np.zeros((320,480,3),np.float32);dep=np.zeros((320,480),np.float32);weight=np.zeros((320,480),np.float32)
        un=np.zeros((320,480),np.float32);fallback=np.zeros((320,480),bool);kind=np.zeros((320,480),np.uint8)
        y,x=yy[ids],xx[ids];image[y,x]=p['rgb'];dep[y,x]=depth;weight[y,x]=p['loss_weight'];un[y,x]=p['uncertainty'];fallback[y,x]=p['fallback'];kind[y,x]=2 # 2 explicitly inferred, never observed
        np.savez_compressed(out/'frames'/f'{fid}.npz',pseudo_rgb=image,pseudo_depth=dep,loss_weight=weight,
            uncertainty=un,source_kind=kind,fallback=fallback,actor_labels=f['actor_labels'],
            observed_reference_valid=f['deletion_valid'].astype(bool))
        for ai,actor in enumerate(meta['actors'],1):
            mask=f['actor_labels']==ai;core=mask&(distance_transform_edt(mask)>=5);mk=kind>0
            training=mk&~f['deletion_valid'].astype(bool)
            rows.append(dict(frame=fid,actor=actor,denominator=int(mask.sum()),core_denominator=int(core.sum()),
                pseudo_pixels=int((mask&mk).sum()),core_pseudo_pixels=int((core&mk).sum()),
                pseudo_train_pixels=int((mask&training).sum()),core_pseudo_train_pixels=int((core&training).sum()),
                core_fallback=int((core&fallback).sum()),core_weight_mean=float(weight[core&mk].mean()) if (core&mk).any() else 0))
        if fid=='0008102':
            for x,y in [(255,240),(256,219)]:
                j=np.flatnonzero((xx[ids]==x)&(yy[ids]==y))
                critical.append(dict(pixel=[x,y],pseudo_kind='inferred road prior, not observation',
                    xyz=pts[j[0]].tolist() if len(j) else None,rgb=image[y,x].tolist(),depth=float(dep[y,x]),
                    loss_weight=float(weight[y,x]),uncertainty=float(un[y,x]),
                    nearest_donor_m=float(p['nearest_donor_m'][j[0]]) if len(j) else None,
                    two_sided=bool(p['two_sided'][j[0]]) if len(j) else False,fallback=bool(fallback[y,x])))
        births.append(pts);birthrgb.append(p['rgb']);birthnormal.append(p['normal'])
        print(json.dumps(dict(stage='pseudo_targets',frame=fid,pixels=len(ids))),flush=True)
    xyz=np.concatenate(births);col=np.concatenate(birthrgb);norm=np.concatenate(birthnormal)
    keys=np.floor(xyz[:,:2]/CELL).astype(np.int32);kk,vv,cc=group_median(keys,np.c_[xyz[:,2],col,norm])
    np.savez_compressed(out/'road_birth_proposals.npz',local_xyz=np.c_[(kk+.5)*CELL,vv[:,0]],
        rgb=vv[:,1:4],normal_local=vv[:,4:],source_kind=np.full(len(kk),2,np.uint8),
        gaussian_sigma=np.tile([.075,.075,.015],(len(kk),1)),pixel_requests=cc)
    aggregate={}
    for actor in meta['actors']:
        part=[r for r in rows if r['actor']==actor]
        aggregate[str(actor)]={k:sum(r[k] for r in part) for k in ['denominator','core_denominator','pseudo_pixels','core_pseudo_pixels','pseudo_train_pixels','core_pseudo_train_pixels','core_fallback']}
    return dict(rows=rows,aggregate=aggregate,critical=critical,birth_count=len(kk))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--atlas',type=Path)
    a=ap.parse_args();a.output.mkdir(exist_ok=False,parents=True);start=time.time()
    (a.output/'protocol.json').write_text(json.dumps(POLICY,indent=2))
    atlas=dict(np.load(a.atlas)) if a.atlas else build_atlas(a.inputs,a.output/'observed_atlas.npz')
    validation=validate(atlas,a.output);(a.output/'validation.json').write_text(json.dumps(validation,indent=2))
    targets=infer_frames(a.inputs,atlas,a.output)
    summary=dict(validation=validation['aggregate'],targets=targets,elapsed_seconds=time.time()-start,
        GPU_used=False,optimizer_updates=0,quality_accepted=False,human_verdict=None,
        warning='Positive pseudo-label coverage is inferred prior support, NOT proof of accurate reconstruction or measured road coverage.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

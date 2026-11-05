"""Expand the measured road query grid to the near a5/a37 shadow receiver.

8cm road grid as in the successful local road branch. All 199 observations,
measured triangle tests and box guards are retained. Nearer ground returns are
reclassified from stored real nearest returns, never generated or interpolated
as photometric evidence. Original sensor assets and the B model are read only.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,'/root/autodl-tmp')
from fullclip_road_cpu_20261026 import loadmeta,Surface,transform,project,sample,measured_patch_evidence,ray_box_blocked

FULL=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_fullclip_cpu_20261026')
CELL=.08

def ground(f,plane):
    p=f['cloud'];good=(np.abs(p@plane[:3]+plane[3])<=.15)&(p[:,0]>=.5)&(p[:,0]<=90)&(np.abs(p[:,1])<=8)
    for c,h,a in zip(f['box_center'],f['box_half'],f['box_axes']):
        ids=np.flatnonzero(good);good[ids]&=~(np.abs((p[ids]-c)@a)<=h).all(1)
    return good

def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True);target=out/'evidence.npz'
    if target.exists():raise FileExistsError('Preserve expanded evidence')
    started=time.time();m,anchor,inv,plane=loadmeta();scale=m['meter_to_model'];T=np.asarray(m['world_to_model'])@anchor
    inverse=np.linalg.inv(T)
    boxes=[m['frames'][fid]['selected_boxes_model'][m['actors'].index(alias)] for fid in m['frame_order'] for alias in [5,37]]
    centers=np.array([b['center'] for b in boxes]);centers_local=transform(centers,inverse)
    xlo=max(.5,float(centers_local[:,0].min())-5);xhi=min(90.,float(centers_local[:,0].max())+5)
    ylo=max(-8.,float(centers_local[:,1].min())-5);yhi=min(8.,float(centers_local[:,1].max())+5)
    xx,yy=np.meshgrid(np.arange(int(np.floor(xlo/CELL)),int(np.ceil(xhi/CELL))+1),
        np.arange(int(np.floor(ylo/CELL)),int(np.ceil(yhi/CELL))+1),indexing='ij')
    keys=np.c_[xx.ravel(),yy.ravel()];xy=(keys+.5)*CELL
    # Keep a 0.8m geometric halo beyond the 2m receiver repair domain.
    flat=np.c_[xy,np.zeros(len(xy))];model=transform(flat,T);distance=np.full(len(model),np.inf)
    for b in boxes:
        q=(model-np.asarray(b['center']))@np.asarray(b['axes'])/scale
        gap=np.maximum(np.abs(q[:,:2])-np.asarray(b['half'])[:2]/scale,0)
        distance=np.minimum(distance,np.linalg.norm(gap,axis=1))
    keys=keys[distance<=2.8];xy=(keys+.5)*CELL
    chunks=[];near_counts=[]
    for raw in range(199):
        f=np.load(FULL/'frames'/('%03d.npz'%raw));g=ground(f,plane)
        chunks.extend([f['ground_local'],f['cloud'][g&(f['cloud'][:,0]<3)]])
        near_counts.append(int((g&(f['cloud'][:,0]<3)).sum()))
    pool=np.concatenate(chunks);_,unique=np.unique(np.floor(pool[:,:2]/.1).astype(int),axis=0,return_index=True);pool=pool[unique]
    surface=Surface(pool.astype(float),plane);height,supported=surface.height(xy);normal=surface.normal(xy);xyz=np.c_[xy,height]
    extraction=json.loads((FULL/'extraction.json').read_text());sharp=np.array([r['sharpness'] for r in extraction['frames']]);quality=np.minimum(sharp/max(np.median(sharp[89:138]),1e-9),1.)
    all_valid=[];all_rgb=[];rows=[]
    for raw in range(199):
        f=np.load(FULL/'frames'/('%03d.npz'%raw));uv,z=project(xyz,f['K'],f['w2l']);origin=np.linalg.inv(f['w2l'])[:3,3]
        ray=origin-xyz;length=np.linalg.norm(ray,axis=1);cosine=np.abs((ray/np.maximum(length[:,None],1e-9)*normal).sum(1))
        area=f['K'][0,0]*f['K'][1,1]*CELL**2*cosine/np.maximum(z*z,1e-8)
        # Keep source pixel footprint gate tied to the original 20cm evidence
        # patch, so denser receiver sampling does not reject distant geometry.
        area_evidence=area*(.2/CELL)**2
        candidate=supported&(z>.1)&(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)&(sample(f['guard'],uv)==0)&(cosine>=.12)&(area_evidence>=.25)&(quality[raw]>=.2)
        ids=np.flatnonzero(candidate);valid=np.zeros(len(xyz),bool)
        if len(ids):
            ok,dep,tri,conflict=measured_patch_evidence(f['cloud'],ground(f,plane),f['K'],f['w2l'],xyz[ids],uv[ids],normal[ids])
            bs=[dict(center=c,half=h,axes=ax) for c,h,ax in zip(f['box_center'],f['box_half'],f['box_axes'])]
            valid[ids]=ok&~ray_box_blocked(xyz[ids],origin,bs)
        rgb=np.asarray(Image.open(FULL/'rgb'/('%03d.png'%raw)),np.float32)/255.
        all_rgb.append(sample(rgb,uv).astype(np.float32));all_valid.append(valid)
        row=dict(raw=raw,qualified_geometric_sources=int(valid.sum()));rows.append(row)
        if raw%10==0 or raw==198:print(json.dumps(dict(raw=raw,cells=len(xyz),sources=int(valid.sum()),seconds=time.time()-started)),flush=True)
    valid=np.asarray(all_valid);rgb=np.asarray(all_rgb)
    normal_world=normal@anchor[:3,:3].T;world=transform(xyz,anchor)
    np.savez_compressed(target,keys=keys,local_xyz=xyz.astype(np.float32),world_xyz=world,
        model_xyz=transform(world,np.asarray(m['world_to_model'])),normal_local=normal.astype(np.float32),
        normal_world=normal_world.astype(np.float32),gaussian_sigma_m=np.tile([CELL*.65,CELL*.65,.015],(len(xyz),1)),
        surface_supported=supported,source_valid=valid,source_rgb=rgb,source_raw=np.arange(199,dtype=np.int16),cell_m=CELL)
    summary=dict(cell_m=CELL,cells=len(xyz),bounds_local=[xyz.min(0).tolist(),xyz.max(0).tolist()],
        source_frames=199,actual_measured_geometry=True,near_ground_return_counts=near_counts,
        geometry_supported=int(supported.sum()),source_records=rows,seconds=time.time()-started,
        original_source_assets_changed=False,optimizer_updates=0,GPU_used=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:summary[k] for k in ['cell_m','cells','bounds_local','geometry_supported','seconds']}),flush=True)

if __name__=='__main__':main()

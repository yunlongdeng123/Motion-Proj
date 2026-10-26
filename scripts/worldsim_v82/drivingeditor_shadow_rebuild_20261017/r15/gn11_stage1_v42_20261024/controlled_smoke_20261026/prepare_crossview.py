"""对目标轮廓每条射线逐像素核对额外时刻实测几何；不是邻格有色即当标签。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,warnings
import numpy as np
from PIL import Image
sys.path.insert(0,'/root/autodl-tmp')
from fullclip_road_cpu_20261026 import loadmeta,FixedSurface,measured_patch_evidence,ray_box_blocked,transform,project,sample
FULL=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_fullclip_cpu_20261026');OUT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026')
m,anchor,inv,plane=loadmeta();a=dict(np.load(FULL/'full199_initialization.npz'))
surface=FixedSurface(m,dict(keys=a['keys'],world=a['world_xyz'],surface_supported=a['surface_supported'],provenance=a['eligible'].astype(np.uint8),base_weight=a['source_valid'],normal_world=a['normal_world']))
dest=OUT/'crossview';dest.mkdir(exist_ok=True);summary=[]
for raw in [102,129,137]:
    if (dest/f'{raw}.npz').exists():continue
    t=np.load(FULL/'frames'/f'{raw:03d}.npz');r=surface.camera(t['K'],t['w2l']@inv)
    labels=np.load(f'/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023_dense_r7/frames/0008{raw:03d}.npz')['actor_labels']
    target=(labels==4)|(labels==5);ids=np.flatnonzero(target.ravel()&r['supported']);xyz=transform(r['world'][ids],inv);normals=a['normal_local'][np.maximum(r['unit_id'][ids],0)]
    colors=[];valids=[];sources=list(range(138,170))
    for src in sources:
        f=np.load(FULL/'frames'/f'{src:03d}.npz');uv,z=project(xyz,f['K'],f['w2l'])
        inside=(z>.1)&(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)
        ok,depth,tri,conflict=measured_patch_evidence(f['cloud'],f['ground'],f['K'],f['w2l'],xyz,uv,normals)
        boxes=[dict(center=c,half=h,axes=ax) for c,h,ax in zip(f['box_center'],f['box_half'],f['box_axes'])]
        ok&=inside&(sample(f['guard'],uv)==0)&~ray_box_blocked(xyz,np.linalg.inv(f['w2l'])[:3,3],boxes)
        colors.append(sample(np.asarray(Image.open(FULL/'rgb'/f'{src:03d}.png'))/255.,uv));valids.append(ok)
    colors=np.asarray(colors);valids=np.asarray(valids)
    with warnings.catch_warnings():warnings.simplefilter('ignore');median=np.nanmedian(np.where(valids[...,None],colors,np.nan),axis=0)
    inlier=valids&(np.abs(colors-median).mean(2)<=.05);count=inlier.sum(0)
    span=np.where(inlier,np.array(sources)[:,None],-1).max(0)-np.where(inlier,np.array(sources)[:,None],999).min(0)
    accepted=(count>=3)&(count>=.6*valids.sum(0))&(span>=3)
    reference=np.zeros((320*480,3),np.float32);reference[ids]=np.nan_to_num(median,nan=0)
    valid=np.zeros(320*480,bool);valid[ids]=accepted;counts=np.zeros(320*480,np.int16);counts[ids]=count
    np.savez_compressed(dest/f'{raw}.npz',rgb=reference.reshape(320,480,3),valid=valid.reshape(320,480),target=target,labels=labels,counts=counts.reshape(320,480),sources=np.array(sources),depth=r['depth'].reshape(320,480))
    result=dict(raw=raw,valid=int(valid.sum()),target=int(target.sum()),actors={str(x):dict(valid=int((valid.reshape(320,480)&(labels==k)).sum()),target=int((labels==k).sum())) for x,k in [(6,4),(37,5)]},critical={f'{x},{y}':dict(valid=bool(valid.reshape(320,480)[y,x]),rgb=reference.reshape(320,480,3)[y,x].tolist(),sources=int(counts.reshape(320,480)[y,x])) for x,y in [(255,240),(256,219)]})
    summary.append(result);print(json.dumps(result),flush=True)
(dest/'summary.json').write_text(json.dumps(summary,indent=2))

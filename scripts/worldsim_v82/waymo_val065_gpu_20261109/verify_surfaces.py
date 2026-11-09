"""分割完成后核验BUILD跨时刻稀疏表面；不读取任何重建或编辑结果。"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,argparse
import numpy as np
from PIL import Image
from inspect_background import project,transform,occludes
def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--prep',type=Path,required=True);p.add_argument('--actor-track',required=True);p.add_argument('--actor-id',type=int,required=True);a=p.parse_args();b=a.prep/'inputs';sel=read(a.prep/'selected_case.json')
    masks=a.prep/'masks_all_resolved_r2';sem=a.prep/'semantics'
    frames=read(b/'receipt.json')['build_raws'];refs=[r for r in frames if r%10==0 or r==198];queries=[30,60,120,150,198]
    obs={r:read(b/'observations'/f'{r:03d}.json') for r in set(refs+queries)}
    labs={r:np.array(Image.open(masks/f'{r:03d}/labels.png')) for r in obs};roads={r:np.array(Image.open(sem/f'{r:03d}.png').resize((480,320),Image.Resampling.NEAREST))==0 for r in refs}
    target=sel['target']['track_id'];actor=a.actor_track;aid=a.actor_id;pairs=[]
    for ref in refs:
        d=obs[ref];cloud=np.fromfile(b/d['lidar_path'],dtype='<f4').reshape(-1,6);world=transform(cloud[:,:3],np.array(d['ego_world']).reshape(4,4));uv,z,valid,origin=project(world,d)
        ids=np.flatnonzero(valid);pix=np.floor(uv[ids]/4).astype(int);nearest=np.full((320,480),np.inf);np.minimum.at(nearest,(pix[:,1],pix[:,0]),z[ids]);ids=ids[z[ids]<=nearest[pix[:,1],pix[:,0]]+.15];pix=np.floor(uv[ids]/4).astype(int)
        label=labs[ref][pix[:,1],pix[:,0]];road=roads[ref][pix[:,1],pix[:,0]]&(label==0)
        body=next((x for x in d['boxes'] if x['track']==actor),None)
        for role in ['road','vehicle']:
            if role=='vehicle' and body is None:continue
            selected=ids[road if role=='road' else label==aid];w=world[selected]
            if role=='vehicle':
                local=transform(w,np.linalg.inv(np.array(body['world_from_actor']).reshape(4,4)));inside=(abs(local)<=np.array(body['dimensions'])/2+.03).all(1)&(local[:,2]>-body['dimensions'][2]/2+.1);selected=selected[inside];local=local[inside];w=world[selected]
            for box in d['boxes']:
                if role=='vehicle' and box['track']==actor:continue
                good=~occludes(origin,w,box);selected=selected[good];w=w[good]
                if role=='vehicle':local=local[good]
            for q in queries:
                if abs(q-ref)<20:continue
                qd=obs[q];qtarget=next(x for x in qd['boxes'] if x['track']==target)
                if role=='vehicle':
                    qb=next((x for x in qd['boxes'] if x['track']==actor),None)
                    if qb is None:continue
                    qw=transform(local,np.array(qb['world_from_actor']).reshape(4,4))
                else:qw=w
                quv,qz,qvalid,qo=project(qw,qd);qp=np.floor(np.clip(quv/4,[0,0],[479,319])).astype(int)
                hit=qvalid&occludes(qo,qw,qtarget)&(labs[q][qp[:,1],qp[:,0]]==sel['target']['native_actor_id'])
                found=np.flatnonzero(hit);_,chosen=np.unique(qp[found]//2,axis=0,return_index=True);found=found[chosen]
                points=[dict(source_cloud_row=int(selected[i]),reference_uv_native=uv[selected[i]].tolist(),query_uv_native=quv[i].tolist(),query_depth_m=float(qz[i]),world_xyz_m=qw[i].tolist(),actor_local_xyz_m=local[i].tolist() if role=='vehicle' else None) for i in found]
                pairs.append(dict(role=role,reference_raw=ref,query_raw=q,returns=int(hit.sum()),cells_2px=len(found),target_mask_pixels=int((labs[q]==sel['target']['native_actor_id']).sum()),points=points))
        print('REFERENCE',ref,flush=True)
    pairs.sort(key=lambda x:(x['role'],-x['cells_2px'],-x['returns'],x['query_raw'],x['reference_raw']))
    result=dict(status='BUILD_MASK_AND_DEPTH_SUPPORTED_SPARSE_CORRESPONDENCES',roles={'vehicle':actor,'road':'world-fixed road-labelled LiDAR'},pairs=pairs,holdout_used=False,limits=['Same-frame semantic and SAM2 predictions are input estimates, not manual truth.','Other-time RGB is not dense deletion ground truth.','No within-scan actor compensation; pinhole/Brown alignment uncertainty retained.'],human_verdict=None)
    (a.prep/'verified_surface_correspondences.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({role:[{k:v for k,v in x.items() if k!='points'} for x in pairs if x['role']==role][:4] for role in ['road','vehicle']}))
if __name__=='__main__':main()

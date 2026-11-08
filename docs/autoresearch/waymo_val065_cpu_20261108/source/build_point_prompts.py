import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np

def transform(p,m):return p@m[:3,:3].T+m[:3,3]
def choose(uv,n=5):
    if len(uv)==0:return []
    center=np.median(uv,axis=0);chosen=[int(np.argmin(((uv-center)**2).sum(1)))]
    for _ in range(min(n,len(uv))-1):
        distance=np.min(((uv[:,None]-uv[chosen][None])**2).sum(2),axis=1)
        distance[chosen]=-1;chosen.append(int(distance.argmax()))
    return uv[chosen].round(3).tolist()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--raws',type=int,nargs='*');a=ap.parse_args()
    receipt=json.loads((a.bank/'receipt.json').read_text());raws=a.raws or receipt['build_raws'];assert set(raws)<=set(receipt['build_raws'])
    (a.bank/'point_prompts').mkdir(exist_ok=True)
    for raw in raws:
        obs=json.loads((a.bank/'observations'/f'{raw:03d}.json').read_text());xyz=np.fromfile(a.bank/obs['lidar_path'],dtype='<f4').reshape(-1,6)[:,:3]
        world=transform(xyz,np.array(obs['ego_world']).reshape(4,4));pose=np.array(obs['camera_pose_world']).reshape(4,4)@np.array(obs['camera_extrinsic']).reshape(4,4)
        cam=transform(world,np.linalg.inv(pose));dep=cam[:,0];x=-cam[:,1]/np.maximum(dep,.01);y=-cam[:,2]/np.maximum(dep,.01)
        fx,fy,cx,cy,k1,k2,p1,p2,k3=obs['camera_intrinsic_Brown'];rr=x*x+y*y;rad=1+k1*rr+k2*rr**2+k3*rr**3
        uv=np.column_stack((fx*(x*rad+2*p1*x*y+p2*(rr+2*x*x))+cx,fy*(y*rad+p1*(rr+2*y*y)+2*p2*x*y)+cy))
        valid=(dep>.1)&(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1280)&(abs(x)<1.1)&(abs(y)<1.1)
        ids=np.flatnonzero(valid);world=world[ids];uv=uv[ids];dep=dep[ids]
        bins=(uv[:,1].astype(int)//4)*480+uv[:,0].astype(int)//4;nearest=np.full(480*320,np.inf);np.minimum.at(nearest,bins,dep)
        visible=dep<=nearest[bins]+.15
        boxes=obs['boxes'];inside=[];body=[]
        for box in boxes:
            local=transform(world,np.linalg.inv(np.array(box['world_from_actor']).reshape(4,4)));half=np.array(box['dimensions'])/2
            member=(abs(local)<=half+.01).all(1);inside.append(member)
            body.append(member & (local[:,2]>-half[2]+.15) & visible)
        memberships=np.stack(inside) if inside else np.empty((0,len(world)),bool)
        outputs={}
        prompts=json.loads((a.bank/'prompts'/f'{raw:03d}.json').read_text())
        for prompt in prompts:
            k=next(i for i,b in enumerate(boxes) if b['track']==prompt['track_id'])
            candidates=body[k] & (memberships.sum(0)==1)
            origin=pose[:3,3]
            # Reject points whose camera ray meets another annotation box first.
            for j,other in enumerate(boxes):
                if j==k:continue
                inv=np.linalg.inv(np.array(other['world_from_actor']).reshape(4,4));o=transform(origin[None],inv)[0];p=transform(world,inv);d=p-o
                safe=np.where(abs(d)<1e-10,1e-10,d);half=np.array(other['dimensions'])/2
                t1=(-half-o)/safe;t2=(half-o)/safe;near=np.minimum(t1,t2).max(1);far=np.maximum(t1,t2).min(1)
                candidates &= ~((far>=np.maximum(near,0))&(near>0)&(near<.985))
            x0,y0,x1,y1=prompt['bbox_xyxy_native'];inbox=(uv[:,0]>=x0)&(uv[:,0]<=x1)&(uv[:,1]>=y0)&(uv[:,1]<=y1)
            candidates &= inbox
            negative=(memberships.sum(0)>0)&~inside[k]&visible&inbox
            outputs[prompt['track_id']]=dict(positive=choose(uv[candidates]),negative=choose(uv[negative]),positive_returns=int(candidates.sum()),policy='same-frame measured OBB-unique body returns, camera z-buffer and occluder-box exclusion')
        (a.bank/'point_prompts'/f'{raw:03d}.json').write_text(json.dumps(outputs))
        print(raw,sum(bool(v['positive']) for v in outputs.values()),len(outputs),flush=True)
if __name__=='__main__':main()

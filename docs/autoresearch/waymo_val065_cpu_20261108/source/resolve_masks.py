"""Resolve candidate overlaps by annotated ray-box entry depth, same rule for every track."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    receipt=json.loads((a.candidates/'receipt.json').read_text());rows=[];a.output.mkdir(exist_ok=False,parents=True)
    for item in receipt['frames']:
        raw=item['raw'];obs=json.loads((a.bank/'observations'/f'{raw:03d}.json').read_text());prompts=json.loads((a.bank/'prompts'/f'{raw:03d}.json').read_text())
        fx,fy,cx,cy,k1,k2,p1,p2,k3=obs['camera_intrinsic_Brown'];yy,xx=np.indices((320,480));xd=((xx+.5)*4-cx)/fx;yd=((yy+.5)*4-cy)/fy;x=xd.copy();y=yd.copy()
        for _ in range(8):
            rr=x*x+y*y;rad=1+k1*rr+k2*rr**2+k3*rr**3;x=(xd-2*p1*x*y-p2*(rr+2*x*x))/rad;y=(yd-p1*(rr+2*y*y)-2*p2*x*y)/rad
        rays=np.stack((np.ones_like(x),-x,-y),axis=-1).reshape(-1,3)
        world_from_camera=np.array(obs['camera_pose_world']).reshape(4,4)@np.array(obs['camera_extrinsic']).reshape(4,4)
        depth=np.full((320,480),np.inf);labels=np.zeros((320,480),np.uint16);unresolved=np.zeros((320,480),np.int16)
        for prompt in prompts:
            mask=np.array(Image.open(a.candidates/f'{raw:03d}'/f'{prompt["alias"]}.png').resize((480,320),Image.Resampling.NEAREST))>0
            unresolved+=mask
            local_from_camera=np.linalg.inv(np.array(prompt['world_from_actor']).reshape(4,4))@world_from_camera
            direction=rays@local_from_camera[:3,:3].T;o=local_from_camera[:3,3];half=np.array(prompt['dimensions'])/2
            safe=np.where(abs(direction)<1e-10,1e-10,direction);t1=(-half-o)/safe;t2=(half-o)/safe;enter=np.maximum(np.minimum(t1,t2).max(1),.1);leave=np.maximum(t1,t2).min(1)
            fallback=np.linalg.norm(local_from_camera[:3,3])
            distance=np.where(leave>=enter,enter,fallback).reshape(320,480)
            take=mask&(distance<depth);depth[take]=distance[take];labels[take]=int(prompt['alias'][1:])+1
        folder=a.output/f'{raw:03d}';folder.mkdir();Image.fromarray(labels).save(folder/'labels.png')
        for prompt in prompts:
            Image.fromarray(np.uint8(labels==int(prompt['alias'][1:])+1)*255).save(folder/f'{prompt["alias"]}.png')
        rows.append(dict(raw=raw,candidate_overlap_pixels=int((unresolved>1).sum()),resolved_overlap_pixels=0,labeled_pixels=int((labels>0).sum())))
    (a.output/'receipt.json').write_text(json.dumps(dict(status='RESOLVED_CANDIDATE_INPUTS',frames=rows,
        rule='nearest annotated OBB ray entry among SAM2 candidates; center distance fallback outside cuboid projection',
        resolution=[480,320],identity='native label = stable scene alias number + 1; zero is background',quality_accepted=None),indent=2))
    print(json.dumps(rows))
if __name__=='__main__':main()

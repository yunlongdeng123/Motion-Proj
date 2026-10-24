"""Inspect every plane hypothesis at the two user-selected pixels, not an argmin tie."""
import json,argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from cpu_plane_propagation import measured_maps,project,rays,intersect,source_witness,sample,refine_planes

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    meta=json.loads((a.inputs/'manifest.json').read_text());u=dict(np.load(a.inputs/'units.npz'));plane=np.asarray(meta['ground_plane']);plane/=np.linalg.norm(plane[:3])
    fs=[measured_maps(dict(np.load(a.inputs/f'{fid}.npz')),plane) for fid in meta['frame_order']]
    active=u['surface_supported']&((u['base_weight']>0).sum(0)>=2);seeds=u['local'][active];normals=u['normal_local'][active]
    uv=np.array([[255.,240.],[256.,219.]]);i=13;f=fs[i];o,r=rays(f,uv);base,t=intersect(o,r,np.broadcast_to(-plane[3]*plane[:3],(2,3)),np.broadcast_to(plane[:3],(2,3)))
    suv,sz=project(seeds,f);inside=(sz>.1)&(suv[:,0]>=0)&(suv[:,0]<480)&(suv[:,1]>=0)&(suv[:,1]<320);ss=np.flatnonzero(inside);sd,si=cKDTree(suv[ss]).query(uv,k=2);si=ss[si]
    md,mi=cKDTree(seeds[:,:2]).query(base[:,:2],k=2);fit=(np.arange(49)//5)%2==0;out=[]
    for method,inds in [('image_plane',si),('metric_plane',mi),('measured_refined',mi)]:
        for k in range(2):
            n=normals[inds[:,k]];x,tt=intersect(o,r,seeds[inds[:,k]],n)
            valid=(tt>.1)&(tt<120)&((sd[:,k]<=96) if method=='image_plane' else ((md[:,k]<=3)&(np.linalg.norm(x[:,:2]-base[:,:2],axis=1)<=1)))
            if method=='measured_refined':x=refine_planes(x,n,valid,i,uv,fs,fit)
            evidence=[[] for _ in range(2)]
            for j,g in enumerate(fs):
                if j==i:continue
                p,z=project(x,g);inside=(z>.1)&(p[:,0]>=0)&(p[:,0]<479)&(p[:,1]>=0)&(p[:,1]<319)
                clear=inside&(sample(~g['guard'],p)>=1-1e-6);patch=inside&(sample(g['patch_clear'],p)>=1-1e-6)
                measured=sample(g['measured_valid'],p)>=1-1e-6;dep=sample(g['depth'],p);close=measured&(np.abs(dep-z)<=.3)
                ok,_=source_witness(x,g,f,uv)
                for q in range(2):evidence[q].append(dict(frame=meta['frame_order'][j],inside=bool(inside[q]),mask_clear=bool(clear[q]),patch_clear=bool(patch[q]),measured_ground_patch=bool(measured[q]),depth_match=bool(close[q]),final_geometric=bool(ok[q] and valid[q])))
            for q in range(2):out.append(dict(method=method,hypothesis=k,pixel=uv[q].astype(int).tolist(),proposal_valid=bool(valid[q]),local_xyz=x[q].tolist(),counts={key:sum(e[key] for e in evidence[q]) for key in ['inside','mask_clear','patch_clear','measured_ground_patch','depth_match','final_geometric']},frames=evidence[q]))
    a.output.write_text(json.dumps(dict(hypotheses=out,scope='only these propagated planes and current49 measured maps; not an impossibility proof for arbitrary geometry or all possible matching'),indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='frames'} for r in out],indent=2))
if __name__=='__main__':main()

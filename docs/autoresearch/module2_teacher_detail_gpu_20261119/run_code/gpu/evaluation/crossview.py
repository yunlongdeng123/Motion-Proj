"""Diagnostics on actual rendered samples; never read a learned world texture."""
import numpy as np
from .material_evaluation import load,sha,photo,masked_hf

def sample(image,uv):
    x,y=uv[:,0],uv[:,1];xx=np.floor(x).astype(int);yy=np.floor(y).astype(int);dx=x-xx;dy=y-yy
    if image.ndim==3:return sum(image[:,yy+oy,xx+ox]*((dx if ox else 1-dx)*(dy if oy else 1-dy))[None]for oy in [0,1]for ox in [0,1])
    return sum(image[yy+oy,xx+ox]*((dx if ox else 1-dx)*(dy if oy else 1-dy))for oy in [0,1]for ox in [0,1])

def stats(a,b,mask):
    n=int(mask.sum())
    return dict(samples=n,MAE=float(np.abs(a[:,mask]-b[:,mask]).mean())if n else None,RMSE=float(np.sqrt(np.mean((a[:,mask]-b[:,mask])**2)))if n else None)

def evaluate_crossview_packets(packets,frozen,spec):
    rows=[]
    for item in spec['pairs']:
        assert sha(item['path'])==item['sha256'],'fixed crossview correspondence changed'
        m=load(item['path']);s,d=item['src'],item['dst'];sy,sx,uv=m['src_y'],m['src_x'],m['dst_uv'];n=len(sy)
        p=packets[s]['road_rgb'].astype(np.float64);q=packets[d]['road_rgb'].astype(np.float64)
        a=frozen[s]['baseline_road_alpha'];b=frozen[d]['baseline_road_alpha']
        ap=a[sy,sx];bq=sample(b,uv);left=p[:,sy,sx];right=sample(q,uv)
        visible=(m['src_joint_eligible_alphaT']>=.05)&(m['dst_joint_eligible_alphaT']>=.05)
        zero=(m['src_joint_eligible_alphaT']==0)|(m['dst_joint_eligible_alphaT']==0)
        groups={'all_fixed_matches':np.ones(n,bool),'actual_joint_eligible_alphaT_both_ge_.05':visible,'actual_joint_eligible_alphaT_either_zero':zero}
        row=dict(src=s,dst=d,fixed_correspondences=n,roles={},visibility_counts={k:int(v.sum())for k,v in groups.items()},
                 actual_render_premult={},highalpha_unpremult_diagnostic={},HF_1px_diagnostic={})
        hp,vp,_=masked_hf(p,frozen[s]['D'],1.);hq,vq,_=masked_hf(q,frozen[d]['D'],1.)
        hf_ok=vp[sy,sx]&(sample(vq.astype(np.float64),uv)>=1-1e-12)
        row['HF_fixed_support']=dict(qualified_correspondences=int(hf_ok.sum()),unsupported_boundary_correspondences=int((~hf_ok).sum()),localmass_min=.9)
        for key,mask in groups.items():
            row['actual_render_premult'][key]=stats(left,right,mask)
            row['highalpha_unpremult_diagnostic'][key]=stats(left/ap[None],right/bq[None],mask)
            row['HF_1px_diagnostic'][key]=stats(hp[:,sy,sx],sample(hq,uv),mask&hf_ok)
        for key,bit in [('A',1),('C',2),('S',4)]:
            role=(m['source_roles']&bit)>0
            row['roles'][key]=dict(all=stats(left/ap[None],right/bq[None],role),joint_visible=stats(left/ap[None],right/bq[None],role&visible))
        row['not_independent_truth']='Actual rendered reprojection, not field equality; high-alpha division is diagnostic only, not training/detail metric. Camera exposure, footprint filtering and occlusion differ.'
        rows.append(row)
    return dict(policy=spec,rows=rows)

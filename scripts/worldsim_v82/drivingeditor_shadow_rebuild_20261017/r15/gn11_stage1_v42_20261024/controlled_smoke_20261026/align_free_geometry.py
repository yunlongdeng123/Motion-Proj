"""Fit metric Sim(3) on visible nonvehicle LiDAR, spatial-block holdout audit.
Never fit generated hidden roads to the desired road plane; no insertion if alignment fails.
"""
import os,json
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026');F=C.parent/'r15_gn11_fullclip_cpu_20261026'
def similarity(x,y):
    mx=x.mean(0);my=y.mean(0);xx=x-mx;yy=y-my;u,s,vt=np.linalg.svd(yy.T@xx/len(x));D=np.eye(3);D[2,2]=np.linalg.det(u@vt);R=u@D@vt;scale=np.trace(np.diag(s)@D)/(xx**2).sum(1).mean();t=my-scale*R@mx;return scale,R,t
results=[]
critical=json.loads((F/'coverage.json').read_text())['critical']
for actor in ['a37','a6']:
    folder=C/f'freeinpaint_pass3/{actor}/object-removal';camera=np.load(folder/'predicted_cameras.npz');samples=[]
    for i,raw in enumerate([102,109,116]):
        f=np.load(F/'frames'/f'{raw:03d}.npz');X=f['cloud'];cam=X@f['w2l'][:3,:3].T+f['w2l'][:3,3];ph=cam@f['K'].T;uv=ph[:,:2]/np.maximum(cam[:,2:],1e-8)-.5
        inside=(cam[:,2]>.5)&(cam[:,2]<50)&(uv[:,0]>2)&(uv[:,0]<477)&(uv[:,1]>2)&(uv[:,1]<317);idx=np.flatnonzero(inside);ij=np.floor(uv[idx]+.5).astype(int);order=np.argsort(cam[idx,2],kind='stable');_,first=np.unique(ij[order,1]*480+ij[order,0],return_index=True);idx=idx[order[first]];ij=np.floor(uv[idx]+.5).astype(int)
        idx=idx[f['guard'][ij[:,1],ij[:,0]]==0];uv=uv[idx];newuv=(uv+.5)*1.05-.5;depth=np.load(folder/f'{i}_depth.npy').squeeze();z=map_coordinates(depth,[newuv[:,1],newuv[:,0]],order=1,mode='nearest');K=camera['intrinsics'][i];cp=camera['c2ws'][i];xp=(np.c_[newuv+.5,np.ones(len(z))]@np.linalg.inv(K).T)*z[:,None];xp=xp@cp[:3,:3].T+cp[:3,3]
        block=((np.floor(uv[:,0]/24)+np.floor(uv[:,1]/24)).astype(int)%2)==0
        ok=(z>.01)&np.isfinite(xp).all(1);samples.append((xp[ok],X[idx][ok],block[ok],f['ground'][idx][ok]))
    x=np.concatenate([s[0] for s in samples]);y=np.concatenate([s[1] for s in samples]);fit=np.concatenate([s[2] for s in samples]);ground=np.concatenate([s[3] for s in samples]);keep=fit.copy()
    for _ in range(6):
        scale,R,t=similarity(x[keep],y[keep]);error=np.linalg.norm(x@(scale*R).T+t-y,axis=1);keep=fit&(error<=np.quantile(error[fit],.8))
    error=np.linalg.norm(x@(scale*R).T+t-y,axis=1);held=~fit;gh=held&ground
    record=dict(actor=actor,scale=float(scale),rotation=R.tolist(),translation=t.tolist(),fit_points=int(fit.sum()),fit_inliers=int(keep.sum()),holdout_points=int(held.sum()),holdout_3d_median_m=float(np.median(error[held])),holdout_3d_p90_m=float(np.quantile(error[held],.9)),road_holdout_points=int(gh.sum()),road_holdout_median_m=float(np.median(error[gh])),road_holdout_p90_m=float(np.quantile(error[gh],.9)),critical=[])
    depth=np.load(folder/'0_depth.npy').squeeze();cp=camera['c2ws'][0];K=camera['intrinsics'][0]
    for point in critical:
        uv=(np.array(point['pixel'])+.5)*1.05-.5;z=float(map_coordinates(depth,[[uv[1]],[uv[0]]],order=1)[0]);xp=np.r_[uv+.5,1.]@np.linalg.inv(K).T*z;world=(cp[:3,:3]@xp+cp[:3,3])@(scale*R).T+t;observed=np.array(point['variants']['full199']['local_xyz']);record['critical'].append(dict(pixel=point['pixel'],generated_local_world=world.tolist(),observed_local_world=observed.tolist(),difference_m=float(np.linalg.norm(world-observed)),height_difference_m=float(world[2]-observed[2])))
    # Diagnostic admission contract declared before this script runs. Fit is not clean geometry GT.
    requested_mask=np.array(Image.open(C/'inputs'/actor/'masks/102.png'))>0
    for point in record['critical']:
        x0,y0=point['pixel'];point['inside_requested_actor_mask']=bool(requested_mask[y0,x0]);point['road_reference_applicable_to_generated_removal']=bool(requested_mask[y0,x0])
    record['metric_alignment_admitted']=record['holdout_3d_median_m']<=.3 and record['road_holdout_median_m']<=.15 and record['road_holdout_p90_m']<=.3
    record['inserted_into_edit']=False;results.append(record)
(C/'free_geometry_alignment.json').write_text(json.dumps(dict(policy='visible nonvehicle nearest LiDAR; alternating24px spatial blocks fit/check, six80%trimmed Sim3 fits; admit median heldout<=0.3m, road median<=0.15m,p90<=0.3m. These are engineering gates, not scene generalization.',generated_regions_never_used_for_fitting=True,results=results,training_updates=0,quality_accepted=False,human_verdict=None),indent=2));print(json.dumps(results,indent=2))

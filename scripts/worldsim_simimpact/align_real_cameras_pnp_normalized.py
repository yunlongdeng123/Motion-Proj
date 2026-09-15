"""通过真实RGB与原生渲染的特征对应，检查发布global transform的局部标定。

PnP使用已重建GS深度，属于额外参考诊断；不能据此宣称独立几何真值。
只处理冻结BUILD图像，不加载QUERY。
"""
import sys,os,json
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from scipy.ndimage import map_coordinates,maximum_filter,minimum_filter
from scipy.spatial.transform import Rotation
B=Path('/root/autodl-tmp/external/worldsim_simimpact')
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
sys.path[:0]=[str(B/'HUGSIM'),str(B/'HUGSIM/sim')];os.chdir(B/'HUGSIM')
from omegaconf import OmegaConf
from hugsim_env.envs.hug_sim import HUGSimEnv
from scene.cameras import Camera
torch.set_num_threads(4);torch.manual_seed(20260915);cv2.setNumThreads(4);cv2.setRNGSeed(20260915)
sift=cv2.SIFT_create(nfeatures=6000);matcher=cv2.BFMatcher(cv2.NORM_L2)
all_rows=[]
for name in ['scene-0013','scene-0038','scene-0041']:
    out=R/'alignment_pnp_r2'/name;out.mkdir(parents=True,exist_ok=True)
    cfg=OmegaConf.load(R/'rollouts'/name/'native_r1/resolved_config.yaml')
    env=HUGSimEnv(cfg,str(out));env.reset(seed=20260915)
    meta=json.loads((Path(cfg.model_path)/'meta_data.json').read_text());global_T=np.array(meta['inv_pose'])
    views=json.loads((R/'inputs'/f'{name}.json').read_text())['views']
    def render(T,K):
        camera=Camera(K=K,c2w=T,width=800,height=450,image=np.zeros((450,800,3)),image_name='alignment_BUILD')
        with torch.no_grad():pkg=env.render_fn(viewpoint=camera,prev_viewpoint=None,**env.render_kwargs)
        rgb=(pkg['render'].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
        alpha=pkg['alphas'][0,...,0].cpu().numpy()
        normalized=pkg['depth'][0].cpu().numpy()/np.maximum(alpha,1e-6)
        normalized[alpha<.8]=np.nan
        return rgb,normalized
    for vi,v in enumerate(views):
        row={'scene':name,'view_index':vi,'camera':v['camera'],'sample_index':v['sample_index'],'input':v['image'],'extra_reference':'Published GS RGB and alpha-normalized expected depth (alpha>=0.8) used by PnP; not physical GT','iterations':[]}
        original=global_T@np.array(v['world_from_camera']);T=original.copy();K=np.array(v['intrinsics_original'])*.5;K[2,2]=1
        real=np.array(Image.open(v['image']).convert('RGB').resize((800,450),Image.Resampling.LANCZOS));rk,rd=sift.detectAndCompute(cv2.cvtColor(real,cv2.COLOR_RGB2GRAY),None)
        for iteration in range(2):
            rgb,depth=render(T,K);sk,sd=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None)
            if iteration==0:Image.fromarray(rgb).save(out/f'{vi:02d}_render_before.jpg',quality=95)
            if sd is None or rd is None:row['error']='No SIFT descriptors';break
            pairs=matcher.knnMatch(sd,rd,k=2);reverse=matcher.match(rd,sd);back={m.queryIdx:m.trainIdx for m in reverse}
            matches=[a for a,b in pairs if a.distance<.75*b.distance and back.get(a.trainIdx)==a.queryIdx]
            if len(matches)<30:row['error']=f'Only {len(matches)} mutual matches';break
            uv=np.array([sk[m.queryIdx].pt for m in matches]);target=np.array([rk[m.trainIdx].pt for m in matches])
            z=map_coordinates(depth,uv[:,[1,0]].T,order=1,mode='nearest')
            span=maximum_filter(depth,size=5)-minimum_filter(depth,size=5)
            gradient=map_coordinates(span,uv[:,[1,0]].T,order=1,mode='nearest')
            ok=(z>1)&(z<100)&np.isfinite(z)&(gradient<np.maximum(.3,.03*z))
            uv=uv[ok];target=target[ok];z=z[ok]
            points=(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T)*z[:,None];points=points@T[:3,:3].T+T[:3,3]
            hold=np.arange(len(points))%5==0;fit=~hold
            if fit.sum()<20:row['error']='Too few continuous-surface matches';break
            w2c=np.linalg.inv(T);rvec=cv2.Rodrigues(w2c[:3,:3])[0];tvec=w2c[:3,3].reshape(3,1).copy()
            success,rvec,tvec,inliers=cv2.solvePnPRansac(np.ascontiguousarray(points[fit]),np.ascontiguousarray(target[fit]),K,None,rvec,tvec,True,iterationsCount=1000,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
            if not success or inliers is None or len(inliers)<20:row['error']='PnP lacks 20 inliers';break
            inds=inliers.ravel();rvec,tvec=cv2.solvePnPRefineLM(points[fit][inds],target[fit][inds],K,None,rvec,tvec)
            projected=cv2.projectPoints(points,rvec,tvec,K,None)[0][:,0];errors=np.linalg.norm(projected-target,axis=1)
            new_w2c=np.eye(4);new_w2c[:3,:3]=cv2.Rodrigues(rvec)[0];new_w2c[:3,3]=tvec.ravel();updated=np.linalg.inv(new_w2c)
            delta=np.linalg.norm(updated[:3,3]-T[:3,3])
            stat={'mutual_matches':len(matches),'continuous_surface_matches':len(points),'fit_inliers':len(inds),'held_match_median_reprojection_px':float(np.median(errors[hold])),
                'held_fraction_within_2px':float(np.mean(errors[hold]<2)),'pose_update_m':float(delta),'pose_update_deg':float(Rotation.from_matrix(T[:3,:3].T@updated[:3,:3]).magnitude()*180/np.pi)}
            row['iterations'].append(stat)
            if delta>3:row['error']='PnP translation >3m, retained original as unresolved';break
            T=updated
        final_rgb,final_depth=render(T,K);Image.fromarray(real).save(out/f'{vi:02d}_real.jpg',quality=95);Image.fromarray(final_rgb).save(out/f'{vi:02d}_render_after.jpg',quality=95)
        row['world_from_camera_initial']=original.tolist();row['world_from_camera_pnp']=T.tolist();row['translation_change_m']=float(np.linalg.norm(T[:3,3]-original[:3,3]));row['status']='MATCHED' if len(row['iterations'])==2 and 'error' not in row else 'UNRESOLVED'
        np.save(out/f'{vi:02d}_depth_after.npy',final_depth)
        (out/f'{vi:02d}.json').write_text(json.dumps(row,indent=2));all_rows.append(row);print(name,vi,row['status'],row['translation_change_m'],row['iterations'][-1] if row['iterations'] else row.get('error'),flush=True)
    del env;torch.cuda.empty_cache()
(R/'alignment_pnp_r2.json').write_text(json.dumps(all_rows,indent=2))

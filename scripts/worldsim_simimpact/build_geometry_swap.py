"""固定原生GS点数/语义，按官方前馈深度移动可见中心，仅接入碰撞通道。

这是带已知标定、原生可见性/语义与可选BUILD LiDAR锚点的诊断适配器，
不是官方前馈端到端系统；QUERY文件不在这里加载。
"""
import sys,json,os
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import map_coordinates
B=Path('/root/autodl-tmp/external/worldsim_simimpact')
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
sys.path[:0]=[str(B/'HUGSIM'),str(B/'HUGSIM/sim'),str(Path(__file__).parent)]
os.chdir(B/'HUGSIM')
from omegaconf import OmegaConf
from hugsim_env.envs.hug_sim import HUGSimEnv
from scene.cameras import Camera
from geometry_contract import outputs
torch.set_num_threads(4)
torch.manual_seed(20260915)
rows=[]
for name in ['scene-0013','scene-0038','scene-0041']:
    out=R/'geometry_swap'/name;out.mkdir(parents=True,exist_ok=True)
    cfg=OmegaConf.load(R/'rollouts'/name/'native_r1/resolved_config.yaml')
    env=HUGSimEnv(cfg,str(out));env.reset(seed=20260915)
    points=env.points.detach().cpu().numpy();n=len(points)
    model_dir=Path(cfg.model_path);meta=json.loads((model_dir/'meta_data.json').read_text());inv=np.array(meta['inv_pose'])
    inputs=json.loads((R/'inputs'/f'{name}.json').read_text())['views'][:6]
    best=np.full(n,np.inf);view_index=np.full(n,-1,dtype=int);uv=np.zeros((n,2));zold=np.zeros(n)
    native_cameras=[]
    for i,v in enumerate(inputs):
        c2w=inv@np.array(v['world_from_camera']);K=np.array(v['intrinsics_original'])*.5;K[2,2]=1
        camera=Camera(K=K,c2w=c2w,width=800,height=450,image=np.zeros((450,800,3)),image_name='BUILD')
        with torch.no_grad():pkg=env.render_fn(viewpoint=camera,prev_viewpoint=None,**env.render_kwargs)
        dep=pkg['depth'][0].cpu().numpy()
        np.save(out/f'native_depth_{i}.npy',dep)
        cp=(points-c2w[:3,3])@c2w[:3,:3];proj=cp@K.T
        xy=proj[:,:2]/np.clip(proj[:,2:],1e-8,None)
        valid=(cp[:,2]>1)&(cp[:,2]<80)&(xy[:,0]>=0)&(xy[:,0]<799)&(xy[:,1]>=0)&(xy[:,1]<449)
        native_z=map_coordinates(dep,xy[:,[1,0]].T,order=1,mode='constant',cval=0)
        err=np.abs(native_z-cp[:,2]);valid&=err<=np.maximum(.15,.02*cp[:,2])
        score=err/np.maximum(cp[:,2],1)+.001*np.linalg.norm((xy-[400,225])/[400,225],axis=1)
        take=valid&(score<best);best[take]=score[take];view_index[take]=i;uv[take]=xy[take];zold[take]=cp[take,2]
        native_cameras.append(c2w)
    visible=view_index>=0
    np.savez_compressed(out/'identity.npz',points=points,visible=visible,view_index=view_index)
    # Only the first BUILD LiDAR sweep calibrates a common scalar for the first six outputs.
    lidar_world=np.load(R/'reference'/name/'BUILD_00.npz')['points_world']
    for method in ['dvgt1','vggt','omega512','pi3x']:
        for variant in ['six','twelve']:
            predpath=R/'predictions'/method/name/variant
            rec,views,cam,conf,kg,base,cv=outputs(predpath);h,w=rec['network_hw']
            ratios=[]
            for i,v in enumerate(views[:6]):
                T=np.array(v['world_from_camera']);cp=(lidar_world-T[:3,3])@T[:3,:3];projected=cp@kg[i].T
                xy=projected[:,:2]/np.clip(projected[:,2:],1e-8,None)
                valid=(cp[:,2]>1)&(cp[:,2]<80)&(xy[:,0]>=0)&(xy[:,0]<w-1)&(xy[:,1]>=0)&(xy[:,1]<h-1)
                z=map_coordinates(cam[i,...,2],xy[:,[1,0]].T,order=1,mode='constant',cval=0)
                valid&=np.isfinite(z)&(z>.2)
                ratios.extend((cp[valid,2]/z[valid]).tolist())
            scale=float(np.median(ratios)) if ratios else None
            for protocol,factor in [('native_metric_or_baseline',1.),('build_lidar_scalar',scale)]:
                if factor is None:continue
                changed=points.copy();valid_count=0;displacements=[]
                for i in range(6):
                    inds=np.flatnonzero(view_index==i)
                    coords=uv[inds]*[w/800,h/450]+[(w/800-1)/2,(h/450-1)/2]
                    predz=map_coordinates(cam[i,...,2],coords[:,[1,0]].T,order=1,mode='constant',cval=0)*factor
                    valid=np.isfinite(predz)&(predz>.2)&(predz<200)
                    inds=inds[valid];predz=predz[valid];origin=native_cameras[i][:3,3]
                    changed[inds]=origin+(points[inds]-origin)*(predz/zold[inds])[:,None]
                    displacements.extend(np.linalg.norm(changed[inds]-points[inds],axis=1).tolist());valid_count+=len(inds)
                key=f'{method}_{variant}_{protocol}'
                np.savez_compressed(out/f'{key}.npz',points=changed)
                row={'scene':name,'method':method,'variant':variant,'protocol':protocol,'total_centers':n,
                    'eligible_visible_centers':int(visible.sum()),'changed_centers':valid_count,'preserved_center_count':True,
                    'base_scale':base,'additional_scalar':factor,'build_anchor_points':len(ratios),
                    'displacement_quantiles_m':np.quantile(displacements,[0,.1,.5,.9,1]).tolist(),
                    'extra_information':['known calibration','native GS semantic/opacity selection','native visible-surface support and point sampling']+(['BUILD LiDAR global scale'] if protocol=='build_lidar_scalar' else []),
                    'scope':'Collision geometry only. Retains unseen native centers. Not an official end-to-end feed-forward result.'}
                rows.append(row);print(name,key,row['changed_centers'],row['displacement_quantiles_m'][2],flush=True)
    del env
    torch.cuda.empty_cache()
(R/'geometry_swap_protocol.json').write_text(json.dumps(rows,indent=2))

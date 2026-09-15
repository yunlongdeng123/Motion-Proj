import os
"""真实扫描方向上的重建首交，供官方RGB+LiDAR策略消费。

同一时刻的发现诊断，非独立确认。普通BUILD LiDAR尺度控制另列。
只使用首6输出图，避免12图条件因输出面集合增加而天然增加命中。
"""
import os,json,time
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.ndimage import map_coordinates
from geometry_contract import outputs
R=Path(os.environ.get('SIMIMPACT_RUN_ROOT','/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1'))
O=R/'lidar_policy';O.mkdir(exist_ok=True)
(O/'registration.json').write_text(json.dumps({'task':'WS-SIM-LIDAR-01','parent':'WS-SIM-IMPACT-01','stage':'same-time discovery',
    'scenes':sorted(p.stem for p in (R/'inputs').glob('*.json')),'methods':['vggt','omega512','dvgt1','pi3x'],
    'inputs':'Existing frozen six/twelve RGB predictions; first six output maps in both conditions',
    'sensor_reference':'First real BUILD LiDAR scan: fixed measured ray directions and return targets; no raydrop/intensity model',
    'controls':['known calibration and poses','native metric / camera-baseline scale','one median BUILD LiDAR scalar','early-only causal hybrid in downstream evaluation'],
    'claim_limit':'Adapter + policy probe, not yet a pose-responsive closed-loop LiDAR simulator; same-time reference not independent confirmation',
    'human_verdict':None},indent=2))
rows=[]
for name in sorted(p.stem for p in (R/'inputs').glob('*.json')):
    inp=json.loads((R/'inputs'/f'{name}.json').read_text());ego=np.array(inp['views'][0]['world_from_ego_camera'])
    ref=np.load(R/'reference'/name/'BUILD_00.npz');gtworld=ref['points_world'];originworld=ref['lidar_origin_world']
    gt=(gtworld-ego[:3,3])@ego[:3,:3];origin=(originworld-ego[:3,3])@ego[:3,:3]
    ranges=np.linalg.norm(gt-origin,axis=1);good=(ranges>1)&(ranges<80);gt=gt[good];ranges=ranges[good];gtworld=gtworld[good]
    dirs=(gt-origin)/ranges[:,None];rays=np.c_[np.broadcast_to(origin,gt.shape),dirs].astype(np.float32)
    dest=O/'scans'/name;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'real.npz',points=gt,origin=origin,ranges=ranges,directions=dirs,world_from_ego=ego)
    for method in ['dvgt1','vggt','omega512','pi3x']:
        for variant in ['six','twelve']:
            rec,views,cam,conf,K,base,cv=outputs(R/'predictions'/method/name/variant);h,w=rec['network_hw'];ratios=[]
            for i,v in enumerate(views[:6]):
                T=np.array(v['world_from_camera']);cp=(gtworld-T[:3,3])@T[:3,:3];p=cp@K[i].T;uv=p[:,:2]/np.clip(p[:,2:],1e-6,None)
                valid=(cp[:,2]>1)&(cp[:,2]<80)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
                z=map_coordinates(cam[i,...,2],uv[:,[1,0]].T,order=1,mode='constant',cval=0);valid&=(z>.2)&np.isfinite(z)
                ratios.extend((cp[valid,2]/z[valid]).tolist())
            scalar=float(np.median(ratios));yy,xx=np.mgrid[:h,:w];pix=np.stack([xx,yy,np.ones_like(xx)],axis=-1)
            grid=np.arange(h*w).reshape(h,w)
            template=np.concatenate([np.stack([grid[:-1,:-1],grid[:-1,1:],grid[1:,:-1]],-1).reshape(-1,3),
                np.stack([grid[1:,1:],grid[1:,:-1],grid[:-1,1:]],-1).reshape(-1,3)])
            for protocol,factor in [('native_scale',1.),('build_scale',scalar)]:
                verts=[];faces=[];offset=0
                for i,v in enumerate(views[:6]):
                    cp=(pix@np.linalg.inv(K[i]).T)*cam[i,...,2:3]*factor
                    valid=np.isfinite(cp).all(-1)&(cp[...,2]>.2)&(cp[...,2]<100)
                    fs=template[valid.ravel()[template].all(-1)];p=cp.reshape(-1,3)[fs]
                    edges=np.maximum.reduce([np.linalg.norm(p[:,0]-p[:,1],axis=1),np.linalg.norm(p[:,1]-p[:,2],axis=1),np.linalg.norm(p[:,2]-p[:,0],axis=1)])
                    fs=fs[edges<=np.maximum(.15,.05*p[...,2].min(1))]
                    used,index=np.unique(fs,return_inverse=True);T=np.linalg.inv(ego)@np.array(v['world_from_camera'])
                    ve=cp.reshape(-1,3)[used]@T[:3,:3].T+T[:3,3];verts.append(ve);faces.append(index.reshape(-1,3)+offset);offset+=len(ve)
                V=np.concatenate(verts).astype(np.float32);F=np.concatenate(faces).astype(np.uint32)
                scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V),o3d.core.Tensor(F)))
                hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=4);distance=hit['t_hit'].numpy();finite=np.isfinite(distance);delta=distance-ranges
                points=origin+dirs[finite]*distance[finite,None]
                key=f'{method}_{variant}_{protocol}'
                np.savez_compressed(dest/f'{key}.npz',points=points,first_range=distance,gt_range=ranges,origin=origin,directions=dirs,
                    first_triangle=hit['primitive_ids'].numpy(),vertices=V,faces=F)
                row={'scene':name,'method':method,'variant':variant,'protocol':protocol,'rays':len(ranges),'returned':int(finite.sum()),
                    'early_0p2':int((delta<-.2).sum()),'early_0p5':int((delta<-.5).sum()),'early_1m':int((delta<-1).sum()),
                    'hit_0p2':int((abs(delta)<=.2).sum()),'late_0p2':int((finite&(delta>.2)).sum()),'miss':int((~finite).sum()),
                    'median_abs_range_error_m':float(np.median(abs(delta[finite]))),'base_scale':base,'additional_scalar':factor,
                    'extra_information':'Known intrinsics/poses and measured sensor ray directions'+('; same BUILD LiDAR global scalar' if protocol=='build_scale' else ''),
                    'ground_to_above_0p2m_count':int(((gt[finite,2]<=.2)&(points[:,2]>.2)).sum()),
                    'artifact':str(dest/f'{key}.npz'),'status':'DONE'}
                rows.append(row);print(json.dumps({k:row[k] for k in ['scene','method','variant','protocol','early_1m','miss','ground_to_above_0p2m_count']}),flush=True)
                del scene,V,F,verts,faces
(O/'raycast_summary.json').write_text(json.dumps(rows,indent=2))

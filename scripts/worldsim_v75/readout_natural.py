"""真实DVGT点图到有先验的cuboid位置；独立LiDAR检查与普通控制。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from prepare_argoverse import ROOT, LOG, CORNERS, poses, QUAT, POS
from prepare_natural import OUT, BASE

def bbox(center, rotation, dimensions, K):
    points=(CORNERS*dimensions)@rotation.T+center
    uv=points@K.T;uv=uv[:,:2]/uv[:,2:]
    return np.r_[uv.min(0),uv.max(0)]

def ray_faces(rays, center, rotation, dimensions):
    origin=-center@rotation
    direction=rays@rotation
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=(-dimensions/2-origin)/direction;t2=(dimensions/2-origin)/direction
        near=np.minimum(t1,t2);far=np.maximum(t1,t2)
    entry=near.max(1);exit=far.min(1);axis=near.argmax(1)
    sign=-np.sign(direction[np.arange(len(rays)),axis])
    valid=(exit>=entry)&(entry>0)&np.isfinite(entry)
    normal=rotation[:,axis].T
    denominator=(rays*normal).sum(1)
    a=(center*normal).sum(1)/denominator
    b=sign*dimensions[axis]/2/denominator
    return valid,entry,a,b,axis,sign

def read_center(rays, depth, center, rotation, dimensions):
    hit,entry,a,b,axis,sign=ray_faces(rays,center,rotation,dimensions)
    valid=hit&np.isfinite(depth)&(depth>0.2)&np.isfinite(a)&(abs(a)>1e-6)
    if not valid.any(): return None
    scales=(depth[valid]-b[valid])/a[valid]
    scale=float(np.median(scales));estimated=center*scale
    new_hit,_,_,_,new_axis,new_sign=ray_faces(rays,estimated,rotation,dimensions)
    same=new_hit & (new_axis==axis) & (new_sign==sign)
    return {'center_camera':estimated.tolist(),'radial_scale':scale,'n':int(valid.sum()),
            'scale_mad':float(np.median(abs(scales-scale))),
            'face_retention':float(same[valid].mean()),'median_observed_depth_m':float(np.median(depth[valid])),
            'ordinary_surface_depth_m':float(np.median(entry[valid]))}

def main():
    global OUT, BASE
    parser=argparse.ArgumentParser();parser.add_argument('--calibrated-rays',action='store_true')
    parser.add_argument('--run-dir',type=Path,default=OUT)
    args=parser.parse_args();control=args.calibrated_rays;OUT=args.run_dir
    suffix='_ray_control' if control else ''
    path=OUT/f'readout{suffix}_result.json';assert not path.exists(),'拒绝覆盖读出终态'
    p=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'inference_result.json').read_text())['status']=='complete'
    BASE=Path(p['base_run'])
    base=json.loads((BASE/'input_manifest.json').read_text());raw=ROOT/base['log_id']
    traj=np.load(BASE/'trajectory.npz');scene=json.loads((BASE/'scene.json').read_text())
    target=next(t for t in scene['tracks'] if t['id']==p['target'])
    camera=np.array(p['views'][0]['camera_world']);K=np.array(p['views'][0]['K_network'])
    dimensions=np.array(p['target_dimensions_oracle'])
    rotation=camera[:3,:3].T@np.array(p['target_rotation_world_oracle'])
    observed=np.array(p['bbox_network']);cx,cy=(observed[:2]+observed[2:])/2
    depth_guess=K[1,1]*dimensions[2]/(observed[3]-observed[1])
    ray=np.linalg.inv(K)@np.array([cx,cy,1])
    fit=least_squares(lambda c:bbox(c,rotation,dimensions,K)-observed,ray*depth_guess,
                      bounds=([-200,-200,2],[200,200,200]),loss='soft_l1',f_scale=1)
    ordinary=fit.x
    yy,xx=np.mgrid[:512,:512];pixels=np.stack([xx,yy,np.ones_like(xx)],-1)
    bcenter=(observed[:2]+observed[2:])/2;half=(observed[2:]-observed[:2])*0.3
    central=(xx>=bcenter[0]-half[0])&(xx<=bcenter[0]+half[0])&(yy>=bcenter[1]-half[1])&(yy<=bcenter[1]+half[1])
    native=np.load(OUT/'native_outputs.npz')['points'][0,0]
    rdf_to_flu=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    ego=np.array(p['views'][0]['ego_world'])
    world=(native/0.1)@rdf_to_flu.T@ego[:3,:3].T+ego[:3,3]
    depths=[];projection_checks=[]
    for i,v in enumerate(p['views']):
        cw=np.array(v['camera_world']);points=(world[i]-cw[:3,3])@cw[:3,:3]
        # 官方ray depth是ego原点距离。普通校准控制将该距离放回已知相机像素射线，
        # 而不是继续把未对齐的点图朝向当作已知rig朝向；求射线与ego球面的正交点。
        if control:
            rays_world=(pixels@np.linalg.inv(np.array(v['K_network'])).T)@cw[:3,:3].T
            offset=cw[:3,3]-ego[:3,3]
            radius=np.linalg.norm(native[i]/.1,axis=-1)
            aa=(rays_world*rays_world).sum(-1);bb=(rays_world*offset).sum(-1)
            disc=bb*bb+aa*(radius*radius-offset@offset)
            calibrated=(-bb+np.sqrt(np.maximum(disc,0)))/aa
            calibrated[disc<0]=np.nan
            depths.append(calibrated)
        else:
            depths.append(points[...,2])
        uv=points@np.array(v['K_network']).T;uv=uv[...,:2]/uv[...,2:]
        valid=np.isfinite(uv).all(-1)&(points[...,2]>2)&(points[...,2]<80)
        if v['pad_top']:
            valid[:v['pad_top']]=False;valid[512-v['pad_top']:]=False
        errors=np.linalg.norm(uv-pixels[...,:2],axis=-1)
        projection_checks.append({'camera':v['camera'],'valid_pixels':int(valid.sum()),
                                  'median_reprojection_error_px':float(np.median(errors[valid])),
                                  'positive_depth_fraction':float((points[...,2]>.2).mean())})
    depths=np.array(depths)
    rays=pixels[central]@np.linalg.inv(K).T
    raw_read=read_center(rays,depths[0][central],ordinary,rotation,dimensions)
    # 每点时间限制，文件起点在cutoff前不代表整个扫描都可用。
    lidar_file=Path(p['lidar_anchor_file']);lidar=pd.read_feather(lidar_file)
    stamps=int(lidar_file.stem)+lidar.offset_ns.to_numpy(np.int64)
    allowed=stamps<=p['cutoff_ns']
    ego_df=pd.read_feather(raw/'city_SE3_egovehicle.feather')
    lidar_ego=poses(ego_df,[int(lidar_file.stem)])[0];lidar_ego[:3,3]-=np.array(base['city_origin'])
    points=lidar[['x','y','z']].to_numpy(float)[allowed]
    points=points@lidar_ego[:3,:3].T+lidar_ego[:3,3]
    actors=pd.read_feather(raw/'annotations.feather');actors=actors[actors.timestamp_ns==int(lidar_file.stem)]
    background=np.ones(len(points),bool);target_inside=np.zeros(len(points),bool)
    for _,row in actors.iterrows():
        rot=lidar_ego[:3,:3]@Rotation.from_quat(row[QUAT].to_numpy(float)).as_matrix()
        center=lidar_ego[:3,:3]@row[POS].to_numpy(float)+lidar_ego[:3,3]
        dims=row[['length_m','width_m','height_m']].to_numpy(float)
        q=(points-center)@rot
        background &= ~np.all(abs(q)<=dims/2+0.5,axis=-1)
        if row.track_uuid==p['target']:
            target_inside=np.all(abs(q)<=dims/2+0.15,axis=-1)
    anchor_gt=[];anchor_pred=[];anchor_rows=[]
    for i,v in enumerate(p['views']):
        cw=np.array(v['camera_world']);pc=(points-cw[:3,3])@cw[:3,:3]
        uv=pc@np.array(v['K_network']).T
        uv=uv[:,:2]/np.where(abs(uv[:,2:])>1e-8,uv[:,2:],np.nan)
        xy=np.rint(np.nan_to_num(uv,nan=-1)).astype(int)
        good=(pc[:,2]>5)&(pc[:,2]<60)&background&(xy[:,0]>=2)&(xy[:,0]<510)&(xy[:,1]>=2+v['pad_top'])&(xy[:,1]<510-v['pad_top'])
        ids=np.flatnonzero(good);xy=xy[ids];gt=pc[ids,2]
        # 每个网络像素只使用最近的合法背景回波。
        index=xy[:,1]*512+xy[:,0];order=np.lexsort((gt,index));index=index[order]
        keep=np.r_[True,np.diff(index)!=0];sel=order[keep];xy=xy[sel];gt=gt[sel]
        pdz=depths[i,xy[:,1],xy[:,0]]
        neighborhood=np.array([depths[i,xy[:,1]+dy,xy[:,0]+dx] for dy,dx in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]])
        stable=(np.isfinite(neighborhood).all(0))&(pdz>.2)&(np.ptp(neighborhood,axis=0)<=0.1*pdz)
        anchor_gt.extend(gt[stable]);anchor_pred.extend(pdz[stable])
        anchor_rows.append({'camera':v['camera'],'anchor_pixels':int(stable.sum())})
    anchor_gt=np.array(anchor_gt);anchor_pred=np.array(anchor_pred)
    scale=float(np.median(anchor_gt/anchor_pred)) if len(anchor_gt) else None
    scaled_read=read_center(rays,depths[0][central]*scale,ordinary,rotation,dimensions) if scale else None
    pc=(points-camera[:3,3])@camera[:3,:3]
    uv=pc@K.T;uv=uv[:,:2]/np.where(abs(uv[:,2:])>1e-8,uv[:,2:],np.nan)
    in_core=np.all(uv>=bcenter-half,1)&np.all(uv<=bcenter+half,1)&target_inside&(pc[:,2]>.2)
    lrays=np.c_[uv[in_core],np.ones(in_core.sum())]@np.linalg.inv(K).T
    lidar_read=read_center(lrays,pc[in_core,2],ordinary,rotation,dimensions) if in_core.any() else None
    gt_world=np.array(target['centers'][0]);gt_camera=(gt_world-camera[:3,3])@camera[:3,:3]
    ordinary_read={'center_camera':ordinary.tolist(),'bbox_residual_px':(bbox(ordinary,rotation,dimensions,K)-observed).tolist(),
                   'success':bool(fit.success),'n':4}
    rows={'ordinary_bbox':ordinary_read,'dvgt_metric':raw_read,'dvgt_lidar_scaled':scaled_read,'reference_lidar':lidar_read}
    for key,row in rows.items():
        if row is None:continue
        c=np.array(row['center_camera']);cw=c@camera[:3,:3].T+camera[:3,3]
        row.update(center_world=cw.tolist(),center_error_m=float(np.linalg.norm(c-gt_camera)),
                   offset_world_m=(cw-gt_world).tolist(),depth_error_m=float(c[2]-gt_camera[2]),
                   projection_bounds=bbox(c,rotation,dimensions,K).tolist(),
                   max_projected_bound_change_px=float(np.max(abs(bbox(c,rotation,dimensions,K)-bbox(gt_camera,rotation,dimensions,K)))))
    reference_ok=lidar_read is not None and lidar_read['n']>=6 and lidar_read['center_error_m']<=.5 and lidar_read['face_retention']>=.8
    prospective=p.get('admission_policy')=='reference_and_raw_support_without_error_ranking'
    model_ok=all(r is not None and r['n']>=20 and r['face_retention']>=.8 and r['radial_scale']>0 for r in ([raw_read] if prospective else [raw_read,scaled_read]))
    residual=scaled_read is not None and scaled_read['center_error_m']>.3 and scaled_read['max_projected_bound_change_px']>2
    reasons=[]
    if not reference_ok:reasons.append('independent_reference_readout_not_reliable')
    if not model_ok:reasons.append('model_core_or_visible_face_support_insufficient')
    if not prospective and not residual:reasons.append('controlled_residual_below_frozen_threshold')
    result={'status':'complete','human_verdict':None,'target':p['target'],'central_network_pixels':int(central.sum()),
            'calibrated_ray_control':control,
            'coordinate_admission':'raw point-map orientation failed; projected metric numbers invalid as accuracy claims; calibrated ego-range control separately evaluated',
            'gt_center_camera':gt_camera.tolist(),'reference_only_gt_center_world':gt_world.tolist(),
            'readouts':rows,'projection_checks':projection_checks,'metric_scale_control':scale,
            'background_anchor_pixels':len(anchor_gt),'anchor_views':anchor_rows,
            'background_abs_depth_error_raw_median_m':float(np.median(abs(anchor_pred-anchor_gt))) if len(anchor_gt) else None,
            'background_abs_depth_error_scaled_median_m':float(np.median(abs(anchor_pred*scale-anchor_gt))) if len(anchor_gt) else None,
            'lidar_time_filter':{'source_points':len(lidar),'points_at_or_before_cutoff':int(allowed.sum()),
                                 'excluded_future_returns':int((~allowed).sum()),'source':'official av2.structures.sweep offsets from sweep timestamp',
                                 'target_points_in_3d_box':int(target_inside.sum()),'target_points_in_detector_core':int(in_core.sum())},
            'generation_admitted':bool(reference_ok and model_ok and (prospective or residual)),'stop_reasons':reasons,
            'admission_policy':p.get('admission_policy','legacy_scaled_residual_discovery'),
            'failure_ledger_delta':'none','world_model_generation_calls':0,
            'boundary':'one exposed log; real model depth but oracle size/yaw, known calibration and separate extra LiDAR scale; not an end-to-end pure-visual reconstructed scene'}
    np.savez(OUT/f'readout{suffix}_arrays.npz',depths=depths,central_mask=central,lidar_uv=uv[in_core],lidar_camera=pc[in_core],
             anchor_gt=anchor_gt,anchor_pred=anchor_pred)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    rgb=np.load(OUT/'network_rgb.npy')[0]
    im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
    draw.rectangle(observed.tolist(),outline='yellow',width=1)
    draw.rectangle([*(bcenter-half),*(bcenter+half)],outline='cyan',width=1)
    for pt in uv[in_core]:draw.ellipse((pt[0]-1,pt[1]-1,pt[0]+1,pt[1]+1),fill='red')
    im.save(OUT/f'support-overlay{suffix}.png')
    print(json.dumps({k:result[k] for k in ['central_network_pixels','readouts','metric_scale_control','background_anchor_pixels','lidar_time_filter','generation_admitted','stop_reasons']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()

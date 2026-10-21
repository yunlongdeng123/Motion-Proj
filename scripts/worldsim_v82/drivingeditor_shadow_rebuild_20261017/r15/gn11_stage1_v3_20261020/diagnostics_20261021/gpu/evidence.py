"""只读、保守的测量证据。标签是当前射线的诊断假设，不是自动删点规则。"""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates
from scipy.spatial import cKDTree

GATES = ('in_view','surface_supported','mask_clear','road_support','measured_depth_match',
         'boxes_clear','angle_ok','resolution_ok','quality_ok')
POLICY = dict(version='GEOMETRY_EVIDENCE_R1',sigma_extent=3.,lidar_floor_m=.20,
              lidar_relative=.02,surface_floor_m=.15,road_match_px=2.,
              road_anchor_radius_m=.20,color_copy_L1=.08,normal_RGB_L1=.05,
              normal_alpha=.95,RGB_harm_tolerance=1/255.,mask_dilation_px=1)

def sample(image, uv):
    """CHW -> NxC，原始浮点双线性采样。"""
    a=np.asarray(image)
    if a.ndim==2:a=a[None]
    return np.stack([map_coordinates(v,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=np.nan)
                     for v in a],1)

def rotation(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def conditional_depth(row, W, K, pixel, scale, neg_fov=True):
    """深度条件分布代理：3D协方差 + 实际2D conic + 原生J，非中心硬判。

    原生splat用中心z排序，不存在精确体积首交面。本条件区间是线性投影诊断代理，
    不是原生渲染深度/传感器置信区间。异常/不正定时拒绝分类。
    """
    R=rotation(row['quaternion']);cov=R@np.diag(np.square(row['sigma_m']))@R.T
    cam=np.asarray(W)[:3,:3]@np.asarray(row['model_xyz'])+np.asarray(W)[:3,3]
    cam=cam/scale; C=np.asarray(W)[:3,:3]@cov@np.asarray(W)[:3,:3].T
    z=cam[2];tx,ty=(np.tan(-.5),np.tan(-.5)) if neg_fov else (480/(2*K[0,0]),320/(2*K[1,1]))
    fx,fy=480/(2*tx),320/(2*ty)
    xz=min(1.3*tx,max(-1.3*tx,cam[0]/z));yz=min(1.3*ty,max(-1.3*ty,cam[1]/z))
    J=np.array([[fx/z,0,-fx*xz/z],[0,fy/z,-fy*yz/z]])
    a,b,d=row['cov2'];C2=np.array([[a,b],[b,d]],dtype=float)
    inv=np.linalg.inv(C2);delta=np.asarray(pixel)-row['uv'];cross=C[2]@J.T
    var=float(C[2,2]-cross@inv@cross)
    if var < -1e-7 or not np.isfinite(var):raise ValueError('invalid conditional covariance')
    mean=float(row['camera_z_m']+cross@inv@delta);sigma=float(np.sqrt(max(0,var)))
    return dict(mean_m=mean,sigma_m=sigma,interval_m=[mean-3*sigma,mean+3*sigma],
                screen_mahalanobis2=float(delta@inv@delta),
                method='native-projection linear conditional 3sigma proxy; NOT first-return surface')

def classify(row, support, depth):
    """冲突优先要求保守3sigma完整分离；重叠/混合/稀疏证据全部保留未知。"""
    result=dict(label='unknown',reason='insufficient_or_ambiguous_geometry',severity='unknown',
                conditional_depth=depth,policy_version=POLICY['version'])
    if row.get('weight',0)<=0:return dict(result,reason='no_actual_contribution')
    if not depth or depth['screen_mahalanobis2']>9:return dict(result,reason='uncertain_projection_support')
    lo,hi=depth['interval_m'];road=support.get('road');vehicle=support.get('vehicle')
    if support.get('non_target_occluder'):
        # 没有独立身份/运动/表面匹配，不能把非目标mask直接当合法context。
        return dict(result,reason='non_target_occluder_present_identity_or_surface_unverified')
    if road and road['supported']:
        z,tol=road['z_m'],road['uncertainty_m']
        if lo>=z-tol and hi<=z+tol:
            if vehicle and z>vehicle['far_m']+tol:
                return dict(result,label='legal_occlusion',severity='protect',reason='road_surface_behind_measured_target_vehicle')
            if road['current_visible']:
                return dict(result,label='legal_road',severity='protect',reason='within_measured_visible_road_surface_band')
        if road['current_visible'] and road['confirmed_sources']>=2 and hi<z-tol and not vehicle:
            return dict(result,label='geometry_conflict',severity='suspect',reason='static_in_verified_free_space_before_visible_road') if row['branch']!='dynamic' else dict(result,reason='dynamic_contribution_on_visible_road_needs_identity')
    if vehicle and support.get('lidar_m') and row['branch'] in ('ordinary','road'):
        tol=support['lidar_uncertainty_m'];lidar=support['lidar_m']
        if lo>=lidar-tol and hi<=lidar+tol and support.get('target_mask') and support.get('color_L1') is not None and support['color_L1']<=POLICY['color_copy_L1']:
            return dict(result,label='geometry_conflict',severity='suspect',reason='static_near_measured_vehicle_surface_and_vehicle_color',color_L1=support['color_L1'])
    if vehicle and row['branch']=='dynamic' and support.get('target_mask') and support.get('lidar_m'):
        z=support['lidar_m'];tol=support['lidar_uncertainty_m']
        if lo>=z-tol and hi<=z+tol:return dict(result,label='legal_occlusion',severity='protect',reason='dynamic_matches_measured_target_vehicle_surface')
    return result

def evaluate_arrays(before, after, gt, measured_z, valid, normal, unit_ids):
    """同一固定valid分母，空集=未知，不以after重新选择。"""
    e0=np.abs(before['rgb']-gt).mean(1);e1=np.abs(after['rgb']-gt).mean(1)
    def group(mask):
        ix=np.flatnonzero(mask)
        if not len(ix):return dict(count=0,L_before=None,L_after=None,delta_L_valid=None,
                                   interpretation='unknown_empty_verified_denominator')
        d0=np.abs(before['depth'][:,0]-measured_z);d1=np.abs(after['depth'][:,0]-measured_z)
        return dict(count=len(ix),L_before=float(e0[ix].mean()),L_after=float(e1[ix].mean()),
            delta_L_valid=float((e0[ix]-e1[ix]).mean()),RGB_change=float(np.abs(before['rgb'][ix]-after['rgb'][ix]).mean()),
            RGB_harmed_count=int((e1[ix]-e0[ix]>POLICY['RGB_harm_tolerance']).sum()),
            depth_MAE_before_m=float(d0[ix].mean()),depth_MAE_after_m=float(d1[ix].mean()),
            depth_change_m=float(np.abs(before['depth'][ix]-after['depth'][ix]).mean()),
            depth_harmed_count=int((d1[ix]-d0[ix]>.20+.02*measured_z[ix]).sum()),
            alpha_change=float(np.abs(before['alpha'][ix]-after['alpha'][ix]).mean()),
            alpha_drop=float((before['alpha'][ix]-after['alpha'][ix]).mean()),
            alpha_drop_over_1_255_count=int(((before['alpha'][ix,0]-after['alpha'][ix,0])>1/255.).sum()),
            road_contribution_change=float((after['road'][ix]-before['road'][ix]).mean()),
            unit_ids=np.asarray(unit_ids)[ix].tolist())
    return dict(valid=group(valid),normal_control=group(valid&normal),
                denominator_fixed_before_intervention=True,unit_weighting='equal world-anchor observations; not independent pixel/area samples',
                RGB_space='original unfused source sRGB float; no display quantization',hidden_RGB_ground_truth=False)

class Evidence:
    def __init__(self,campaign,meta,scale):
        self.campaign=Path(campaign);self.meta=meta;self.scale=scale
        config=json.loads((self.campaign/'arms/full/run30k/effective_config.json').read_text())
        bundle=Path(config['v3_bundle']);z=np.load(bundle.parent/meta['units'])
        self.u={k:z[k] for k in z.files};z.close();self.frames={}
        self.A=np.asarray(meta['world_to_model']);self.invA=np.linalg.inv(self.A)

    def frame(self,raw):
        if raw in self.frames:return self.frames[raw]
        from PIL import Image
        m=self.meta['frames'][f'0008{raw:03d}'];W=np.asarray(m['world_to_camera']).copy();W[:3]*=self.scale;W=W@self.invA
        xyz=self.u['model_xyz'];cam=xyz@W[:3,:3].T+W[:3,3];K=np.asarray(m['intrinsic'])
        uv=cam[:,:2]/cam[:,2:]*np.diag(K)[:2]+K[:2,2]-.5
        with np.load(self.campaign/f'final49/full_r2/0008{raw:03d}/render_float.npz') as z:
            lidar=z['lidar_depth_m'][0].copy();gt=z['gt'].copy();target=z['body_mask'][0].astype(bool)
        paths=m['occluder_masks']['paths'];mask=target.copy()
        for path in paths[:2]:mask|=np.asarray(Image.open(path).resize((480,320),Image.Resampling.NEAREST))>0
        with np.load(paths[2]) as z:native=z['masks'].astype(bool).any(0)
        mask|=np.asarray(Image.fromarray(native).resize((480,320),Image.Resampling.NEAREST))>0
        mask=binary_dilation(mask,iterations=POLICY['mask_dilation_px'])
        supported=self.u['surface_supported']&(cam[:,2]>.01)
        ids=np.flatnonzero(supported);tree=cKDTree(uv[ids])
        validlidar=np.argwhere(np.isfinite(lidar)&(lidar>0));ltree=cKDTree(validlidar[:,::-1])
        out=dict(W=W,K=K,uv=uv,z=cam[:,2]/self.scale,mask=mask,target=target,gt=gt,lidar=lidar,
                 ids=ids,tree=tree,ltree=ltree,lcoords=validlidar,meta=m)
        self.frames={raw:out};return out

    def measured(self,f,uv):
        distance,index=f['ltree'].query(uv);yx=f['lcoords'][index];z=f['lidar'][yx[:,0],yx[:,1]]
        return z, distance<=1.5, distance

    def ray_support(self,raw,pixel):
        f=self.frame(raw);x,y=pixel;distance,i=f['tree'].query(pixel);uid=int(f['ids'][i]);u=self.u
        # 以当前像素真实射线与局部实测道路切平面求交，不把附近中心z当该像素道路z。
        inv=np.linalg.inv(f['W']);d=inv[:3,:3]@np.linalg.inv(f['K'])@np.r_[np.asarray(pixel)+.5,1.]
        origin=inv[:3,3];normal=u['normal_model'][uid];anchor=u['model_xyz'][uid]
        den=float(normal@d);z=float(normal@(anchor-origin)/den/self.scale) if abs(den)>1e-8 else -1
        point=origin+d*z*self.scale;offset=float(np.linalg.norm(point-anchor)/self.scale)
        source_count=int((u['base_weight'][:,uid]>0).sum());current=all(u['gate_'+g][raw-89,uid] for g in GATES)
        road=dict(unit_id=uid,z_m=z,uncertainty_m=POLICY['surface_floor_m']+POLICY['lidar_relative']*max(0,z),
                  supported=bool(z>0 and distance<=POLICY['road_match_px'] and offset<=POLICY['road_anchor_radius_m']),
                  anchor_offset_m=offset,projection_distance_px=float(distance),confirmed_sources=source_count,
                  current_visible=bool(current and source_count>=2 and not f['mask'][y,x]),
                  source_raws=(np.flatnonzero(u['base_weight'][:,uid]>0)+89).tolist())
        mz,valid,dist=self.measured(f,np.array([pixel]));lidar=float(mz[0]) if valid[0] else None
        # 来源锚点的可见性不能替代所选像素自己的深度证据；邻近静态遮挡也需拒绝。
        lidar_tol=None if lidar is None else POLICY['lidar_floor_m']+POLICY['lidar_relative']*lidar
        road['selected_pixel_lidar_agrees']=bool(lidar is not None and abs(lidar-z)<=max(road['uncertainty_m'],lidar_tol))
        road['current_visible']=bool(road['current_visible'] and road['selected_pixel_lidar_agrees'])
        vehicle=None
        for box in f['meta']['selected_boxes_model']:
            # OBB只给范围，不把其前面当真实车表面；必须另有同射线LiDAR。
            axes=np.asarray(box['axes']);o=(origin-box['center'])@axes;v=d@axes;half=np.asarray(box['half'])
            if np.any((abs(v)<1e-10)&(abs(o)>half)):continue
            safe=np.where(abs(v)<1e-10,1,v);a=(-half-o)/safe;b=(half-o)/safe
            near=np.where(abs(v)<1e-10,-np.inf,np.minimum(a,b)).max();far=np.where(abs(v)<1e-10,np.inf,np.maximum(a,b)).min()
            if far>=max(near,0):
                candidate=dict(near_m=float(max(0,near)/self.scale),far_m=float(far/self.scale),surface='OBB interval, NOT measured surface')
                if vehicle is None or candidate['near_m']<vehicle['near_m']:vehicle=candidate
        if vehicle and lidar is not None and not(vehicle['near_m']-.3<=lidar<=vehicle['far_m']+.3):vehicle=None
        return dict(road=road,vehicle=vehicle,lidar_m=lidar,lidar_offset_px=float(dist[0]),
                    lidar_uncertainty_m=lidar_tol,
                    target_mask=bool(f['target'][y,x]),non_target_occluder=bool(f['mask'][y,x] and not f['target'][y,x]),
                    gt_observed_rgb=f['gt'][:,y,x].tolist())

    def enrich_trace(self,result):
        f=self.frame(result['raw']);support=self.ray_support(result['raw'],result['pixel']);counts={}
        for row in result['top']:
            evidence=dict(support,color_L1=float(np.abs(np.asarray(row['color'])-support['gt_observed_rgb']).mean()))
            try:depth=conditional_depth(row,f['W'],f['K'],result['pixel'],self.scale)
            except (ValueError,np.linalg.LinAlgError):depth=None
            row['geometry_evidence']=classify(row,evidence,depth);label=row['geometry_evidence']['label'];counts[label]=counts.get(label,0)+1
        result['geometry_evidence']=dict(policy=POLICY,support=support,counts=counts,
            scope='exported actual Top-weight union Top-color contributors only; unexported contributors unclassified',
            labels_are_diagnostic_not_deletion_authority=True,uncertainty='declared conservative tolerance, NOT calibrated sensor confidence')
        return result

    def evaluate_cf(self,raw,before,after,pixel):
        f=self.frame(raw);u=self.u;frame=raw-89;uv=u['source_uv'][frame]-.5
        inside=np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)
        safe=np.nan_to_num(uv,nan=-10,posinf=-10,neginf=-10);p=np.floor(safe).astype(int);pc=np.clip(p,[0,0],[478,318]);x,y=pc.T
        clear=~(f['mask'][y,x]|f['mask'][y+1,x]|f['mask'][y,x+1]|f['mask'][y+1,x+1])
        gates=np.logical_and.reduce([u['gate_'+g][frame] for g in GATES]);confirmed=(u['base_weight']>0).sum(0)>=2
        mz,depthvalid,dist=self.measured(f,safe);tol=POLICY['lidar_floor_m']+POLICY['lidar_relative']*f['z']
        depthagree=depthvalid&(np.abs(mz-f['z'])<=tol)
        valid=inside&gates&confirmed&clear&depthagree&(u['base_weight'][frame]>0)
        b={k:sample(v,safe) for k,v in before.items()};a={k:sample(v,safe) for k,v in after.items()}
        finite=np.logical_and.reduce([np.isfinite(v).all(1) for v in [*b.values(),*a.values()]])
        if np.any(valid&~finite):raise FloatingPointError('nonfinite on fixed valid denominator; cannot silently drop after-intervention pixels')
        gt=u['source_rgb'][frame];e0=np.abs(b['rgb']-gt).mean(1)
        outside=np.max(np.abs(uv-np.asarray(pixel)),axis=1)>10
        normal=(e0<=POLICY['normal_RGB_L1'])&(b['alpha'][:,0]>=POLICY['normal_alpha'])&(abs(b['depth'][:,0]-mz)<=tol)&outside
        result=evaluate_arrays(b,a,gt,mz,valid,normal,u['unit_ids'])
        unobserved=u['is_query']&inside&~valid&finite
        result.update(policy=POLICY,raw=raw,rejection_counts=dict(total=len(uv),inside=int(inside.sum()),
            all_nine_gates=int((inside&gates).sum()),two_sources=int((inside&gates&confirmed).sum()),
            clear_footprint=int((inside&gates&confirmed&clear).sum()),depth_confirmed=int(valid.sum())),
            lidar_matching='nearest actual depth within1.5px, conservative z band; sparse, not dense GT',
            unique_valid_pixels=len(set(map(tuple,np.rint(uv[valid]).astype(int)))),
            unobserved_query=dict(count=int(unobserved.sum()),RGB_loss=None,
                                 road_contribution_change=float((a['road'][unobserved]-b['road'][unobserved]).mean()) if unobserved.any() else None,
                                 depth_change_m=float(abs(a['depth'][unobserved]-b['depth'][unobserved]).mean()) if unobserved.any() else None,
                                 alpha_change=float(abs(a['alpha'][unobserved]-b['alpha'][unobserved]).mean()) if unobserved.any() else None,
                                 reason='no verified same-world current-view RGB denominator; do not invent hidden road GT'))
        return result

"""以普通类别先验替换目标GT形状，并检查可见表面能约束哪些任务状态。"""
from datetime import datetime,timezone
from pathlib import Path
import copy,json,os,time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch
from readout_natural import bbox,ray_faces,read_center
from following_geometry import scene,Route
from prepare_argoverse import CORNERS
from raster_ground import RasterGround
from rgb_idm_policy import RGBIDMPolicy
from run_approach_state_control import rollout

ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
OUT=ROOT/'WS-V75-SHAPE-PRIOR-AUDIT-01/20260920-r1'
SOURCES=[('WS-V75-APPROACH-CLOSEDLOOP-01/20260920-association-r2','WS-V75-APPROACH-STATE-CONTROL-01/20260920-r1'),
         ('WS-V75-APPROACH-CLOSEDLOOP-02/20260920-r1','WS-V75-APPROACH-STATE-CONTROL-02/20260920-r1')]
DIMS=np.array([3.9,1.6,1.56])
ARMS=['bbox_class_prior','dvgt_class_prior','dvgt_visible_extent','scaled_visible_extent','lidar_visible_extent']


def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def fit_class(observed,K,rotation):
    center=(observed[:2]+observed[2:])/2
    ray=np.linalg.inv(K)@np.r_[center,1.]
    distance=K[1,1]*DIMS[2]/(observed[3]-observed[1])
    fit=least_squares(lambda c:bbox(c,rotation,DIMS,K)-observed,ray*distance,
                      bounds=([-200,-200,2],[200,200,200]),loss='soft_l1',f_scale=1)
    assert fit.success and np.isfinite(fit.x).all()
    return fit.x,{'success':bool(fit.success),'bbox_residual_px':fit.fun.tolist()}


def fit_visible(observed,K,rotation,rays,depth,initial):
    """已知朝向下，近端面的中位位置加固定车长；四条像素边界求横向/高度及宽高。"""
    good=np.isfinite(depth)&(depth>.2);assert good.sum()>=6
    points=rays[good]*depth[good,None]
    plane=float(np.median(points@rotation[:,0]))
    fixed_along=plane+DIMS[0]/2
    def unpack(z):
        center=rotation@np.array([fixed_along,z[0],z[1]])
        dimensions=np.r_[DIMS[0],np.exp(z[2:])]
        return center,dimensions
    def residual(z):
        center,dimensions=unpack(z)
        return bbox(center,rotation,dimensions,K)-observed
    initial_local=initial@rotation
    result=least_squares(residual,np.r_[initial_local[1:],np.log(DIMS[1:])],
                         bounds=([-200,-200,np.log(.1),np.log(.1)],[200,200,np.log(10),np.log(10)]),
                         loss='soft_l1',f_scale=1)
    center,dimensions=unpack(result.x)
    singular=np.linalg.svd(result.jac,compute_uv=False)
    valid,entry,_,_,axes,signs=ray_faces(rays,center,rotation,dimensions)
    support=good&valid
    return center,dimensions,{'success':bool(result.success),'bbox_residual_px':result.fun.tolist(),
        'visible_plane_coordinate_m':plane,'plane_p90_residual_m':float(np.percentile(abs(points@rotation[:,0]-plane),90)),
        'jacobian_singular_values':singular.tolist(),'numerical_bound_active':bool(np.any(result.active_mask)),
        'support_points':int(support.sum()),'first_face_counts':{f'{axis}:{sign}':int(np.sum(support&(axes==axis)&(signs==sign))) for axis in range(3) for sign in [-1,1]},
        'median_abs_ray_depth_residual_m':float(np.median(abs(entry[support]-depth[support]))) if support.any() else None}


def task_geometry(center,rotation,dimensions,route,initial_ego):
    corners=(CORNERS*dimensions)@rotation.T+center
    coords=np.array([route.project(x) for x in corners]);ego_s,_=route.project(initial_ego)
    return {'unclipped_near_route_gap_m':float(coords[:,0].min()-ego_s-2.4),
            'lateral_min_m':float(coords[:,1].min()),'lateral_max_m':float(coords[:,1].max()),
            'intersects_route_corridor':bool(coords[:,1].min()<=1 and coords[:,1].max()>=-1)}


def case(source_path,state_path):
    s=json.loads((source_path/'protocol.json').read_text());base=Path(s['base']);r=source_path/'reconstruction'
    p=json.loads((r/'protocol.json').read_text());old=json.loads((r/'readout_ray_control_result.json').read_text())
    data=scene(base);tr=np.load(base/'trajectory.npz');target=next(x for x in data['tracks'] if x['id']==p['target'] and 0 in x['frames'])
    out=OUT/s['source_log'];out.mkdir()
    camera=np.array(p['views'][0]['camera_world']);K=np.array(p['views'][0]['K_network']);observed=np.array(p['bbox_network'])
    ego=np.array(p['views'][0]['ego_world']);yaw=Rotation.from_matrix(ego[:3,:3]).as_euler('xyz')[2]
    rotation_world=Rotation.from_euler('z',yaw).as_matrix();rotation=camera[:3,:3].T@rotation_world
    assert rotation[2,0]>0,'近端面先验只用于同向、朝向远离相机的跟车目标'
    arr=np.load(r/'readout_ray_control_arrays.npz');mask=arr['central_mask'];depth=arr['depths'][0][mask]
    yy,xx=np.mgrid[:512,:512];pixels=np.stack([xx,yy,np.ones_like(xx)],-1)
    rays=pixels[mask]@np.linalg.inv(K).T
    points=np.load(r/'reference_supplement_points.npz');lrays=np.c_[points['uv'],np.ones(len(points['uv']))]@np.linalg.inv(K).T
    ldepth=points['camera_points'][:,2]
    # 先复现原读出，防止新的比较建立在不同像素/单位或深度协议上。
    replay=read_center(rays,depth,np.array(old['readouts']['ordinary_bbox']['center_camera']),
                       camera[:3,:3].T@np.array(p['target_rotation_world_oracle']),np.array(p['target_dimensions_oracle']))
    replay_error=float(np.max(abs(np.array(replay['center_camera'])-old['readouts']['dvgt_metric']['center_camera'])))
    assert replay_error<1e-10
    ordinary,fit_info=fit_class(observed,K,rotation)
    raw=read_center(rays,depth,ordinary,rotation,DIMS);assert raw is not None
    fits={'bbox_class_prior':(ordinary,DIMS.copy(),fit_info),
          'dvgt_class_prior':(np.array(raw['center_camera']),DIMS.copy(),raw)}
    for name,rs,ds in [('dvgt_visible_extent',rays,depth),
                       ('scaled_visible_extent',rays,depth*old['metric_scale_control']),
                       ('lidar_visible_extent',lrays,ldepth)]:
        fits[name]=fit_visible(observed,K,rotation,rs,ds,ordinary)
    # 以上拟合未接目标GT位置、尺寸或朝向。以下参考只进入误差和信息覆盖评价。
    truth_center=np.array(target['centers'][0]);truth_rotation=Rotation.from_quat(target['quaternions'][0]).as_matrix()
    truth_dimensions=np.array(target['dimensions']);route=Route(tr['ego_world'][:,:3,3])
    truth_task=task_geometry(truth_center,truth_rotation,truth_dimensions,route,tr['ego_world'][0,:3,3])
    rows=[];conditions={}
    for arm in ARMS:
        center,dimensions,details=fits[arm];world=center@camera[:3,:3].T+camera[:3,3]
        geometry=task_geometry(world,rotation_world,dimensions,route,tr['ego_world'][0,:3,3])
        body={'arm':arm,'center_camera':center.tolist(),'center_world':world.tolist(),'dimensions_m':dimensions.tolist(),
              'yaw_world_rad':float(yaw),'rotation_world':rotation_world.tolist(),'fit':details,
              'center_error_m':float(np.linalg.norm(world-truth_center)),
              'dimension_error_m':(dimensions-truth_dimensions).tolist(),
              'projected_bounds':bbox(center,rotation,dimensions,K).tolist(),
              'yaw_error_deg':float(np.rad2deg(Rotation.from_matrix(rotation_world.T@truth_rotation).as_euler('xyz')[2])),
              'task_geometry':geometry,'near_gap_error_m':geometry['unclipped_near_route_gap_m']-truth_task['unclipped_near_route_gap_m'],
              'target_gt_shape_used_for_fitting':False,'reference_lidar_uses_gt_target_mask':arm=='lidar_visible_extent'}
        if arm.endswith('visible_extent'):
            body['readout_numerically_supported']=bool(details['success'] and not details['numerical_bound_active'] and details['support_points']>=(6 if arm=='lidar_visible_extent' else 20))
        else:body['readout_numerically_supported']=bool(fit_info['success'] and (arm=='bbox_class_prior' or raw['n']>=20 and raw['face_retention']>=.8))
        condition=copy.deepcopy(data);t=next(x for x in condition['tracks'] if (x['id'],x['segment'])==(target['id'],target['segment']))
        t['centers']=(np.array(target['centers'])-truth_center+world).tolist()
        # 仅保留公共GT相对运动，用于隔离初始形状/姿态；不是去除全部真值的端到端系统。
        gt_rot=Rotation.from_quat(target['quaternions']).as_matrix()
        new_rot=gt_rot@truth_rotation.T@rotation_world
        t['quaternions']=Rotation.from_matrix(new_rot).as_quat().tolist();t['dimensions']=dimensions.tolist()
        conditions[arm]=condition;rows.append(body)
    prior_center,prior_dimensions,_=fits['dvgt_visible_extent']
    hit,entry,*_=ray_faces(rays,prior_center,rotation,prior_dimensions)
    eps=.001
    longer=prior_dimensions.copy();longer[0]+=eps
    shifted=prior_center+rotation[:,0]*eps/2
    new_hit,new_entry,*_=ray_faces(rays,shifted,rotation,longer)
    common=hit&new_hit
    cert={'purpose':'one finite-difference verification of hidden-length ambiguity; not a generated perturbation or badcase search',
          'length_increment_m':eps,'center_shift_m':float(np.linalg.norm(shifted-prior_center)),
          'common_core_rays':int(common.sum()),'total_core_rays':len(rays),
          'max_first_depth_change_m':float(np.max(abs(entry[common]-new_entry[common]))) if common.any() else None,
          'projected_bounds_change_px':(bbox(shifted,rotation,longer,K)-bbox(prior_center,rotation,prior_dimensions,K)).tolist()}
    before=task_geometry(prior_center@camera[:3,:3].T+camera[:3,3],rotation_world,prior_dimensions,route,tr['ego_world'][0,:3,3])
    after=task_geometry(shifted@camera[:3,:3].T+camera[:3,3],rotation_world,longer,route,tr['ego_world'][0,:3,3])
    cert['near_route_gap_change_m']=after['unclipped_near_route_gap_m']-before['unclipped_near_route_gap_m']
    save(out/'readout.json',{'rows':rows,'gt_shape_reference_only':{'center':truth_center.tolist(),'dimensions':truth_dimensions.tolist(),'rotation':truth_rotation.tolist(),'task_geometry':truth_task},
                           'old_readout_replay_max_abs_m':replay_error,'hidden_length_certificate':cert,'human_verdict':None})
    # 所有固定读出均报告；只有数值有支持的读出进入CPU状态控制，无生成准入。
    extrinsic=np.linalg.inv(tr['ego_world'][0])@tr['camera_world'][0]
    policy=RGBIDMPolicy(tr['ego_world'][:,:3,3],tr['K'],extrinsic,[0,0,0],detector=False)
    terrain=RasterGround(base).mesh_for_route(tr['ego_world'][:,:2,3])
    baseline=json.loads((state_path/'gt_clean.json').read_text())
    old_control=json.loads((state_path/'dvgt_metric.json').read_text())
    source_rows=json.loads((source_path/'gt_clean/decisions.json').read_text())
    for row in rows:
        if not row['readout_numerically_supported']:
            row['direct_control']=None;continue
        value,_=rollout(base,tr,policy,terrain,conditions[row['arm']],data,target,source_rows,False)
        save(out/f'{row["arm"]}_control.json',value)
        a=np.array([x['applied_acceleration_mps2'] for x in value['decisions']])
        b=np.array([x['applied_acceleration_mps2'] for x in baseline['decisions']])
        row['direct_control']={k:value[k] for k in ['progress_m','final_speed_mps','final_target_reference_clearance_m','overlap_frames']}
        row['direct_control'].update(progress_change_vs_gt_m=value['progress_m']-baseline['progress_m'],
            mean_abs_action_change_vs_gt_mps2=float(np.mean(abs(a-b))),max_abs_action_change_vs_gt_mps2=float(np.max(abs(a-b))))
    result={'status':'complete','log_id':s['source_log'],'target':p['target'],'rows':rows,
            'old_oracle_dvgt':{'center_error_m':old['readouts']['dvgt_metric']['center_error_m'],
                               'progress_change_vs_gt_m':old_control['progress_m']-baseline['progress_m']},
            'truth_task':truth_task,'hidden_length_certificate':cert,'old_readout_replay_max_abs_m':replay_error,
            'new_direct_control_frames':117*sum(x['direct_control'] is not None for x in rows),
            'human_verdict':None,'failure_ledger_delta':'none','generation_admitted':False,
            'boundary':'controlled cuboid adapters from DVGT range, saved RGB box and ordinary priors; not native object detection/shape outputs; future relative actor motion remains shared extra information'}
    save(out/'result.json',result)
    print(json.dumps({'log':s['source_log'],'hidden_length_certificate':cert,'rows':[{k:x[k] for k in ['arm','center_error_m','dimensions_m','near_gap_error_m','direct_control']} for x in rows]}),flush=True)
    return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not OUT.exists();OUT.mkdir(parents=True)
    protocol={'task_id':'WS-V75-SHAPE-PRIOR-AUDIT-01','run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'sources':[{'generated_run':str(ROOT/a),'direct_state_run':str(ROOT/b)} for a,b in SOURCES],
              'question':'Did oracle target shape hide important task-state errors, or does a standard prior plus observed surface suffice?',
              'role':'post hoc fixed two-task input-role audit; no reconstruction-error ranking or independent confirmation',
              'prior_dimensions_lwh_m':DIMS.tolist(),
              'prior_source':'https://raw.githubusercontent.com/open-mmlab/OpenPCDet/master/tools/cfgs/kitti_models/pointpillar.yaml',
              'prior_source_used':'Car anchor_sizes only; no PointPillars model, weights, rotations, scores or results imported',
              'orientation':'upright and aligned with known ego heading; no target GT yaw/pitch/roll in fit',
              'inputs':'saved detector bbox, calibration, ego heading, saved calibrated DVGT range; unknown length remains fixed class prior',
              'visible_fit':'median visible plane along ego-forward axis plus half prior length sets longitudinal center; four bbox edges fit lateral/vertical center and positive width/height; no GT shape or translation in fit',
              'numerical_bounds':'positive widths/heights0.1..10m and transverse coordinates+-200m; bound-active estimates are rejected, not retuned',
              'extra_information_controls':'old frozen background-LiDAR scale and old three-scan target-LiDAR core with GT target masks; not equal-information advantages',
              'arms':ARMS,'old_readout_replay_tolerance_m':1e-10,
              'cpu_feedback':'shared original initial RGB command and relative GT actor motion, same117frames15decisions; full other actors/route/terrain retained',
              'certificate':'single1mm length change and half-length center shift tests core first-depth invariance; algebraic observability check only, no model intervention',
              'stop':'exact two sources and five ordinary readouts; no threshold/seed/source/horizon change, no new detector/model or generation; no automatic promotion from CPU controls',
              'human_verdict':None,'failure_ledger_refs':['V74-H2-F20','V74-H2-F22'],'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol);start=time.monotonic();result={'status':'running','human_verdict':None,'failure_ledger_delta':'none'}
    try:
        cases=[case(ROOT/a,ROOT/b) for a,b in SOURCES]
        assert not torch.cuda.is_initialized()
        result.update(status='complete',cases=cases,new_model_calls=0,cuda_initialized=False,
                      direct_control_frames=sum(c['new_direct_control_frames'] for c in cases),generation_admitted=False)
    except BaseException as exc:
        result.update(status='failed_stopped',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['wall_s']=time.monotonic()-start;save(OUT/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()

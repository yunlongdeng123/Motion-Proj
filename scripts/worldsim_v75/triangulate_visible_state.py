"""已知相机/共同actor运动下的多视图点一致性；先用真实视频与LiDAR检验。"""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from prepare_visible_sources import OUT

FRAMES=[0,15,30,45,60]

def geometry(base,target):
    traj=np.load(base/'trajectory.npz');K=traj['K']
    actor=next(t for t in json.loads((base/'scene.json').read_text())['tracks'] if t['id']==target)
    relative=[];motion=[]
    for f in FRAMES:
        i=actor['frames'].index(f);T=np.eye(4)
        T[:3,:3]=Rotation.from_quat(actor['quaternions'][i]).as_matrix();T[:3,3]=actor['centers'][i]
        relative.append(np.linalg.inv(T)@traj['camera_world'][f]);motion.append(T)
    return K,np.array(relative),np.array(motion)

def fit_points(points,valid,K,cameras):
    n=points.shape[1];rows=[];fitted=np.full((n,3),np.nan);accepted=np.zeros(n,bool)
    for j in range(n):
        row={'point_id':j,'all_five_tracked':bool(valid[:,j].all()),'accepted':False};rows.append(row)
        if not valid[:,j].all():continue
        uv=points[:,j];d=np.c_[(uv-K[2:])/K[:2],np.ones(5)]
        d=np.einsum('fij,fj->fi',cameras[:,:3,:3],d);d/=np.linalg.norm(d,axis=1,keepdims=True)
        origins=cameras[:,:3,3];projectors=np.eye(3)[None]-d[:,:,None]*d[:,None,:]
        matrix=projectors.sum(0);condition=float(np.linalg.cond(matrix))
        angle=float(np.rad2deg(np.arccos(np.clip((d@d.T).min(),-1,1))))
        row.update(ray_angle_deg=angle,normal_matrix_condition=condition)
        if condition>1e8:continue
        initial=np.linalg.solve(matrix,np.einsum('fij,fj->i',projectors,origins))
        def project(x):
            q=np.einsum('fji,fj->fi',cameras[:,:3,:3],x-origins)
            return q[:,:2]/q[:,2:]*K[:2]+K[2:],q[:,2]
        solved=least_squares(lambda x:(project(x)[0]-uv).ravel(),initial,loss='linear',max_nfev=100)
        reprojection,depth=project(solved.x);errors=np.linalg.norm(reprojection-uv,axis=1)
        ok=bool(solved.success and (depth>0).all() and angle>=2 and np.median(errors)<=2 and errors.max()<=4)
        fitted[j]=solved.x;accepted[j]=ok
        row.update(accepted=ok,point_actor=solved.x.tolist(),camera_depths_m=depth.tolist(),
                   reprojection_error_px=errors.tolist(),median_reprojection_px=float(np.median(errors)),max_reprojection_px=float(errors.max()))
    return {'points':rows,'accepted_count':int(accepted.sum()),'initial_count':n},fitted,accepted

def prepare():
    folder=OUT/'triangulation';assert not folder.exists();folder.mkdir()
    protocol={'frozen_utc':datetime.now(timezone.utc).isoformat(),'role':'additional observer frozen after world-model execution began but before reading its image/metric results',
              'input':'fixed real/RAFT point identities, known camera trajectory and shared GT actor poses; no DVGT depth used in triangulation',
              'geometry':'least-distance ray initialization + ordinary unweighted pixel reprojection least squares over all five times',
              'point_gate':'all five tracked; positive depth; max ray angle>=2deg; median reprojection<=2px and max<=4px; no time trimming',
              'real_gate':'at least8 accepted points and >=25% original; at least6 accepted points with initial LiDAR projection within3px; median neighboring LiDAR depth discrepancy<=0.5m',
              'lidar_role':'extra independent metric reference, matching adjacent projected returns is not exact feature-point ground truth',
              'generated_report':'same unmodified gates; accepted/common counts and reprojection; metric depth differences only if real gate passes and enough common points remain',
              'boundary':'assumes points follow shared GT rigid actor motion; failed fit indicates inconsistency or correspondence error, not uniquely physical drift; GT trajectories are extra information',
              'stop':'one observer/configuration, no jitter/threshold/window grid; real failure leaves generated 3D claim unqualified',
              'reference':'https://docs.opencv.org/4.x/d0/dbd/group__triangulation.html','human_verdict':None}
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    selection=json.loads((OUT/'observation_selection.json').read_text());read=json.loads((OUT/'readout_queue_result.json').read_text());rows=[]
    for case in read['cases']:
        if not case['admitted']:continue
        log=case['log_id'];target=case['target'];base=OUT/'cases'/log/'base';dest=folder/log;dest.mkdir()
        tracks=np.load(OUT/'real'/log/target/'tracks.npz');K,cameras,motion=geometry(base,target)
        result,xyz,ok=fit_points(tracks['points'],tracks['valid'],K,cameras)
        lidar=np.load(Path(case['readout_dir'])/'readout_ray_control_arrays.npz')['lidar_camera']
        luv=lidar[:,:2]/lidar[:,2:]*K[:2]+K[2:]
        distance=np.linalg.norm(tracks['points'][0,:,None,:]-luv[None],axis=-1);nearest=distance.argmin(1)
        association=ok & (distance.min(1)<=3)
        qc=(xyz-cameras[0,:3,3])@cameras[0,:3,:3]
        delta=qc[:,2]-lidar[nearest,2];matched=int(association.sum())
        median=float(np.median(abs(delta[association]))) if matched else None
        passed=result['accepted_count']>=8 and result['accepted_count']/len(ok)>=.25 and matched>=6 and median<=.5
        result.update(log_id=log,target=target,real_metric_admitted=bool(passed),lidar_associations=matched,
                      lidar_neighbor_median_depth_error_m=median,association_pixel_distances=distance.min(1).tolist(),
                      actor_translation_over_window_m=float(np.linalg.norm(motion[-1,:3,3]-motion[0,:3,3])),
                      camera_baseline_in_actor_m=float(np.linalg.norm(cameras[-1,:3,3]-cameras[0,:3,3])),human_verdict=None)
        (dest/'real_result.json').write_text(json.dumps(result,indent=2)+'\n')
        np.savez(dest/'real_fit.npz',points_actor=xyz,accepted=ok,association=association,depth_error_m=delta,K=K,cameras=cameras)
        rows.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='points' and k!='association_pixel_distances'}),flush=True)
    (folder/'real_summary.json').write_text(json.dumps({'status':'complete','cases':rows,'human_verdict':None},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare']);p.parse_args();prepare()

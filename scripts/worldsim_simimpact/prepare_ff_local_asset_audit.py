"""两个检测翻转的有限局部网格干预；重新投射完整扫描并保留删除对照。"""
import json,time,shutil
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from pyquaternion import Quaternion
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1')
F=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FF-PERCEPTION-01/20260915-r1')
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
O=F/'local_asset_inputs_r2'
if O.exists():raise RuntimeError(f'Preserve {O}')
O.mkdir();shutil.copy2(__file__,O/'source_snapshot.py')
support=json.loads((F/'candidate_support_r1/summary.json').read_text())
reg={'task_id':'WS-SIM-FF-LOCAL-ASSET-01','run_id':O.name,'scenes':['scene-0004'],'seed':20260915,
     'selection':'Two previously frozen score0.3 CenterPoint center-distance detection misses: Omega car and Pi3X pedestrian. Both six/twelve geometry outputs audited; twelve-view downstream behavior is not yet observed.',
     'scope':'Actual saved triangle vertices/faces are edited, then all current measured rays are cast again. Existing nine real past sweeps/intensity and BUILD metric scale remain common extra inputs. Not a complete feedback simulator or independent confirmation.',
     'repair':'For all original mesh vertices within0.35m of an original model first-hit point on any GT-box target return, add one robust median radial range offset (GT minus model), preserving triangle connectivity. Only originally returned target beams fit this scalar. Save full modified mesh.',
     'deletion':'Delete every face touching the same selected vertices, without moving other vertices. Count all resulting changed or missing rays, including rays outside the target.',
     'coverage':'Restore only rays already missing in the original asset from real LiDAR in every condition. Newly missing rays after editing stay missing. No unseen ranges are supplied to the raycaster.',
     'information':'Evaluator-held GT target boxes and current ranges select/fit the ordinary oracle patch; this is not a same-information advantage or a proposed learning method.',
     'conditions':['original_twelve','local_radial','same_faces_deleted'],
     'budget':'Six-view original detector results reused; two twelve-view originals plus two interventions x two methods x two view counts =10 detector calls. One radius, one scalar protocol, no sweep.',
     'stop_rule':'Require improved target range readout and recovered target detection beyond same-face deletion. No recovery does not rule out every local repair, but this ordinary local-bias hypothesis closes without further tuning.',
     'failure_ledger_refs':['V74-H2-F19'],'failure_ledger_delta':'pending','human_verdict':None,'prepared':False,'start_unix':time.time()}
(O/'registration.json').write_text(json.dumps(reg,indent=2))
rows=json.loads((F/'detection_r1/summary.json').read_text());realrow=next(r for r in rows if r['scene']=='scene-0004' and r['condition']=='real');sample=realrow['sample_token']
meta=json.loads((N/'metadata/scene-0004.json').read_text());sd=meta['sample_data'][realrow['lidar_token']]
def pose(t,k):
    r=meta[t][k];p=np.eye(4);p[:3,:3]=Quaternion(r['rotation']).rotation_matrix;p[:3,3]=r['translation'];return p
views=json.loads((I/'inputs/scene-0004.json').read_text())['views'];to_lidar=np.linalg.inv(pose('ego_pose',sd['ego_pose_token'])@pose('calibrated_sensor',sd['calibrated_sensor_token']))@np.array(views[0]['world_from_ego_camera'])
real=np.load(I/'lidar_policy/scans/scene-0004/real.npz');gt=real['points'];origin=real['origin'];directions=real['directions'];ranges=real['ranges'];rays=np.c_[np.broadcast_to(origin,gt.shape),directions].astype(np.float32)
real_all=np.load(F/'scene-0004/real.npz')['points'];current=real_all[real_all[:,4]==0];history=real_all[real_all[:,4]>0];assert len(current)==len(gt)
def cast(V,Fs):
    scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)),o3d.core.Tensor(Fs.astype(np.uint32))))
    return scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)['t_hit'].numpy()
frames=[];audit=[]
for candidate in support:
    method=candidate['method'];g=candidate['gt'];box_center=np.array(g['center_lidar']);rot=Quaternion(g['rotation']).rotation_matrix;half=np.array(g['wlh'])[[1,0,2]]/2
    lidar_gt=gt@to_lidar[:3,:3].T+to_lidar[:3,3];target=(abs((lidar_gt-box_center)@rot)<half).all(1)
    for variant in ['six','twelve']:
        source=I/f'lidar_policy/scans/scene-0004/{method}_{variant}_build_scale.npz';z=np.load(source);V=z['vertices'];Fs=z['faces'];original=z['first_range'];finite=np.isfinite(original);target_returned=target&finite
        delta=float(np.median(ranges[target_returned]-original[target_returned]));hits=origin+directions[target_returned]*original[target_returned,None]
        dist,_=cKDTree(hits).query(V,workers=4);mask=dist<=.35;assert mask.any();changed=V.copy();radial=changed[mask]-origin;radial/=np.linalg.norm(radial,axis=1,keepdims=True);changed[mask]+=radial*delta
        touched=mask[Fs].any(1);folder=O/(method+'_'+variant);folder.mkdir();np.savez_compressed(folder/'local_radial_asset.npz',vertices=changed,faces=Fs,changed_vertices=mask)
        np.savez_compressed(folder/'same_faces_deleted_asset.npz',vertices=V,faces=Fs[~touched])
        paths={}
        def save_scan(condition,f):
            live=finite&np.isfinite(f);keep=~finite|live
            # 与已有 fill_missing 基线保留同一射线顺序，避免 voxel 每格点截断的顺序混杂。
            xyz=gt.copy();xyz[live]=origin+directions[live]*f[live,None]
            xyz=xyz[keep]@to_lidar[:3,:3].T+to_lidar[:3,3];p=np.c_[xyz,current[keep,3],np.zeros(int(keep.sum()))]
            p=np.r_[p,history].astype(np.float32)
            path=folder/(condition+'.npz');np.savez_compressed(path,points=p,first_range=f);paths[method+'_'+variant+'_'+condition]=str(path)
        if variant=='twelve':save_scan('original_twelve',original)
        for condition,vertices,faces in [('local_radial',changed,Fs),('same_faces_deleted',V,Fs[~touched])]:
            f=cast(vertices,faces);newfinite=np.isfinite(f);both=finite&newfinite;changed_ray=(finite!=newfinite)|(both&(abs(f-original)>.0001));valid_target=target_returned&newfinite
            info={'method':method,'variant':variant,'condition':condition,'instance':candidate['instance'],'class':candidate['class'],
                'source_asset':str(source),'delta_m':delta,'selected_vertices':int(mask.sum()),'faces_touched':int(touched.sum()),
                'target_gt_rays':int(target.sum()),'original_target_returned':int(target_returned.sum()),'new_target_returned':int(valid_target.sum()),
                'new_missing_on_originally_returned':int((finite&~newfinite).sum()),'changed_rays':int(changed_ray.sum()),'changed_non_target_rays':int((changed_ray&~target).sum()),
                'target_MAE_before_m':float(abs(original[target_returned]-ranges[target_returned]).mean()),
                'target_MAE_after_on_retained_m':float(abs(f[valid_target]-ranges[valid_target]).mean()) if valid_target.any() else None,
                'new_target_median_signed_error_m':float(np.median(f[valid_target]-ranges[valid_target])) if valid_target.any() else None}
            audit.append(info);save_scan(condition,f);print('LOCAL_ASSET_READOUT',json.dumps(info),flush=True)
        frames.append({'scene':'scene-0004','sample_token':sample,'points':paths,'method':method,'variant':variant,'target_instance':candidate['instance']})
(O/'geometry_audit.json').write_text(json.dumps(audit,indent=2));reg.update(prepared=True,expected_detector_forwards=sum(len(f['points']) for f in frames),end_unix=time.time())
(O/'registration.json').write_text(json.dumps(reg,indent=2));(O/'input_manifest.json').write_text(json.dumps({'registration':reg,'frames':frames},indent=2));print('LOCAL_ASSET_PREPARED',reg['expected_detector_forwards'],flush=True)

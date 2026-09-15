"""按实际检测翻转整理对象证据；只读诊断，不将点云偏差自动归因为几何。"""
import json, shutil, tarfile
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from nuscenes.utils.data_classes import Box

D=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-DETECTOR-01/20260915-r1')
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
out=D/'object_audit_r2'
if out.exists(): raise RuntimeError(f'Preserve {out}')
out.mkdir();shutil.copy2(__file__,out/'source_snapshot.py')
class SceneMetadata:
    def __init__(self,path):self.tables=json.loads(path.read_text())
    def get(self,table,token):return self.tables[table][token]
nusc=SceneMetadata(N/'metadata/scene-0004.json')
classes=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
runs={'real':D/'real_scene0004_r1','raw':D/'native_scene0004_step030000_r1','median':D/'native_scene0004_step030000_r1',
      'logged_raw':D/'logged_pattern_detection_scene0004_r1','logged_median':D/'logged_pattern_detection_scene0004_r1'}
rows={c:[r for r in json.loads((p/'summary.json').read_text()) if r.get('condition','real')==c.split('_')[-1]] for c,p in runs.items()}
def box(g):return Box(g['center_lidar'],g['wlh'],Quaternion(g['rotation']))
def poly(b):return Polygon(b.bottom_corners()[:2].T)
def pred_boxes(c,i):
    name=f'prediction_{i:02d}.npz' if c=='real' else f'prediction_{c.split("_")[-1]}_{i:02d}.npz'
    z=np.load(runs[c]/name);bb=z['boxes']
    # 对齐官方 output_to_nusc_box：底部中心 -> 几何中心，dx/dy/dz -> w/l/h。
    return [(Box([b[0],b[1],b[2]+b[5]/2],[b[4],b[3],b[5]],Quaternion(axis=[0,0,1],radians=float(b[6]))),float(s),classes[int(l)]) for b,s,l in zip(bb,z['scores'],z['labels'])]
def pose(t,k):
    r=nusc.get(t,k);p=np.eye(4);p[:3,:3]=Quaternion(r['rotation']).rotation_matrix;p[:3,3]=r['translation'];return p
def inside(p,b,margin=0):
    q=(p[:,:3]-b.center)@b.orientation.rotation_matrix
    return (np.abs(q)<(b.wlh[[1,0,2]]/2+margin)).all(1)
allrows=[];instances={}
for i in range(8):
    points={'real':np.load(D/f'real_scene0004_r1/input_{i:02d}.npz')['points']}
    points.update({c:np.load(D/f'native_scans_scene0004_step030000_r1/{c}_{i:02d}.npz')['points'] for c in ['raw','median']})
    predictions={c:pred_boxes(c,i) for c in runs}
    for g in rows['real'][i]['eligible_gt']:
        if g['class'] not in ['car','truck','construction_vehicle','bus','trailer','motorcycle']:continue
        b=box(g);record={'index':i,**g,'conditions':{},'point_support':{}}
        for c in runs:
            matched={m['instance'] for m in rows[c][i]['scores']['0.5']['bev_iou0p5']['matches']}
            pp=predictions[c];near=sorted([(float(np.linalg.norm(p.center[:2]-b.center[:2])),p,s,cl) for p,s,cl in pp if s>=.1],key=lambda a:a[0])[:1]
            record['conditions'][c]={'matched':g['instance'] in matched}
            if near:
                dist,p,s,cl=near[0];record['conditions'][c]['nearest_prediction']={'distance_m':dist,'score':s,'class':cl,'bev_iou':poly(p).intersection(poly(b)).area/poly(p).union(poly(b)).area,'center':p.center.tolist(),'wlh':p.wlh.tolist()}
        for c,p in points.items():
            current=p[p[:,4]==0];region=current[inside(current,b,1.)];inside_points=current[inside(current,b)]
            info={'current_in_gt_box':len(inside_points),'current_within_1m':len(region),'all_sweeps_in_gt_box':int(inside(p,b).sum())}
            if len(inside_points): info['intensity_quantiles']=np.quantile(inside_points[:,3],[.1,.5,.9]).tolist()
            if c!='real' and len(region):
                real=points['real'];real=real[(real[:,4]==0)&inside(real,b,1.)]
                if len(real):info['pred_to_real_local_NN_quantiles_m']=np.quantile(cKDTree(real[:,:3]).query(region[:,:3])[0],[.5,.9]).tolist()
            record['point_support'][c]=info
        allrows.append(record)
        v=instances.setdefault(g['instance'],{'class':g['class'],'observations':0,**{c:0 for c in runs}});v['observations']+=1
        for c in runs:v[c]+=record['conditions'][c]['matched']
    # 保存首、中、末时刻全场景与真实投影；不人为绘制假表面。
    if i in [0,5,7]:
        sd=nusc.get('sample_data',rows['real'][i]['lidar_token']);sample=nusc.get('sample',rows['real'][i]['sample_token'])
        ext=pose('calibrated_sensor',sd['calibrated_sensor_token']);pc={c:p[(p[:,4]==0)] for c,p in points.items()}
        data={c:np.c_[p[:,:3]@ext[:3,:3].T+ext[:3,3],p[:,3:]] for c,p in pc.items()}
        data['lidar_to_ego']=ext
        np.savez_compressed(out/f'frame_{i:02d}_points.npz',**data)
        for cam in ['CAM_FRONT','CAM_FRONT_RIGHT']:
            csd=nusc.get('sample_data',sample['data'][cam]);shutil.copy2(N/'data'/csd['filename'],out/f'frame_{i:02d}_{cam}.jpg')
            transform=np.linalg.inv(pose('ego_pose',csd['ego_pose_token'])@pose('calibrated_sensor',csd['calibrated_sensor_token']))@pose('ego_pose',sd['ego_pose_token'])@ext
            intrinsic=np.array(nusc.get('calibrated_sensor',csd['calibrated_sensor_token'])['camera_intrinsic']);proj=[]
            for g in rows['real'][i]['eligible_gt']:
                if g['class'] not in ['car','construction_vehicle']:continue
                xyz=transform[:3,:3]@box(g).corners()+transform[:3,3:4]
                if (xyz[2]<=.1).any():continue
                uv=intrinsic@xyz;uv=uv[:2]/uv[2:];xyxy=[*uv.min(1),*uv.max(1)]
                proj.append({'instance':g['instance'],'class':g['class'],'xyxy':xyxy,'corners_uv':uv.T.tolist()})
            (out/f'frame_{i:02d}_{cam}_projection.json').write_text(json.dumps(proj,indent=2))
(out/'objects.json').write_text(json.dumps(allrows,indent=2))
(out/'instance_summary.json').write_text(json.dumps(instances,indent=2))
with tarfile.open(D/'object_audit_r2.tar.gz','w:gz') as t:
    for p in out.iterdir():t.add(p,arcname=p.name)
print(json.dumps(instances,indent=2))

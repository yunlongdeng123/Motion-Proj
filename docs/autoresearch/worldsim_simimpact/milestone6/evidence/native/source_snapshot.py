"""冻结 nuScenes 检测器先验证真实扫描；不把弱跨域检测基线当作重建证据。"""
import argparse,copy,json,sys,time,shutil
from pathlib import Path
import numpy as np
import torch
import importlib.metadata
from pyquaternion import Quaternion
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.utils import category_to_detection_name
from mmengine.config import Config
from mmdet3d.apis import init_model,inference_detector
from mmdet3d.evaluation.metrics.nuscenes_metric import output_to_nusc_box

B=Path('/root/autodl-tmp/external/worldsim_simimpact/mmdetection3d')
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
I=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-INTERACTION-01/20260915-r1')
D=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-DETECTOR-01/20260915-r1')
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--frames',type=int,default=8);ap.add_argument('--scans-dir',type=Path);a=ap.parse_args()
if a.out.exists():raise RuntimeError(f'Preserve existing run: {a.out}')
a.out.mkdir(parents=True)
shutil.copy2(__file__,a.out/'source_snapshot.py')
seed=20260915;np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
reg={'task_id':'WS-SIM-NATIVE-DETECTOR-01','run_id':a.out.name,'scene':'scene-0004','frames':a.frames,'seed':seed,
     'role':'Frozen native-dataset real-sensor detection reference, no reconstruction condition or training.',
     'method':'CenterPoint voxel0.1 circle NMS, MMDetection3D v1.4.0 implementation; not a new geometry SOTA.',
     'code_revision':'fe25f7a51d36e3702f961e198894580d83c4387b',
     'checkpoint':str(D/'assets/mmdet3d_centerpoint_voxel01.pth'),
     'checkpoint_source':'https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth',
     'inputs':'Current raw LiDAR plus nine causal past sweeps; known metric poses. Preaggregated xyz/intensity/age replaces file and sweep loaders; native model, voxelization, range filter and postprocessing retained.',
     'preprocessing':'Current sweep retains its original points; historical sweeps remove the central +/-1m square, then transform into current LiDAR frame. Intensity remains raw nuScenes units and time is seconds.',
     'evaluation':'Class-aware center distance <=2m and BEV IoU>=0.5, separately at fixed confidence 0.3/0.5; front0..32m, lateral +/-16m, >=5 current LiDAR returns. Counts are frame observations, not AP/NDS or unique actors.',
     'failure_ledger_refs':['V74-H2-F18'],'failure_ledger_delta':'pending','human_verdict':None,'completed':False,'start_unix':time.time()}
reg['runtime']={name:importlib.metadata.version(name) for name in ['torch','numpy','mmcv','mmengine','mmdet','nuscenes-devkit']}
reg['runtime']['extra_dependencies']=str(D/'pydeps')
conditions=['raw','median'] if a.scans_dir else ['real']
reg['conditions']=conditions
if a.scans_dir:
    scan_reg=json.loads((a.scans_dir/'registration.json').read_text());assert scan_reg['completed']
    reg.update(role='Frozen detection on native SplatAD sensor outputs; no geometry causal conclusion.',
               inputs=str(a.scans_dir),sensor_registration=scan_reg,
               preprocessing='Native exported xyz/intensity/age arrays, matched ten-sweep timestamps. Native detector range filter/voxelization/postprocessing retained.')
(a.out/'registration.json').write_text(json.dumps(reg,indent=2))
cfg=Config.fromfile(B/'configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py')
# 输入已包含十帧及其时间，不让演示 API 把同一帧重复十次或清空时间通道。
pipe=copy.deepcopy(cfg.test_dataloader.dataset.pipeline)
pipe=[p for p in pipe if p['type']!='LoadPointsFromMultiSweeps']
cfg.test_dataloader.dataset.pipeline=pipe
model=init_model(cfg,reg['checkpoint'],device='cuda:0')
checkpoint_state=torch.load(reg['checkpoint'],map_location='cpu')['state_dict']
model_state=model.state_dict()
load_check={'missing_keys':sorted(set(model_state)-set(checkpoint_state)),
            'unexpected_keys':sorted(set(checkpoint_state)-set(model_state)),
            'shape_differences':{k:[list(v.shape),list(model_state[k].shape)] for k,v in checkpoint_state.items() if k in model_state and v.shape!=model_state[k].shape}}
reg['checkpoint_compatibility']=load_check
(a.out/'registration.json').write_text(json.dumps(reg,indent=2))
assert not any(load_check.values()), load_check
del checkpoint_state,model_state
classes=list(model.dataset_meta['classes'])
nusc=NuScenes(version='v1.0-trainval',dataroot=str(N/'data'),verbose=False)
sample=nusc.get('sample',json.loads((I/'inputs/scene-0004.json').read_text())['views'][0]['sample_token'])
reg['dataset_split']='train' if 'scene-0004' in create_splits_scenes()['train'] else 'val'
(a.out/'registration.json').write_text(json.dumps(reg,indent=2))

def pose(table,token):
    r=nusc.get(table,token);p=np.eye(4);p[:3,:3]=Quaternion(r['rotation']).rotation_matrix;p[:3,3]=r['translation'];return p
def world_lidar(sd):return pose('ego_pose',sd['ego_pose_token'])@pose('calibrated_sensor',sd['calibrated_sensor_token'])
def multisweep(sd):
    inv=np.linalg.inv(world_lidar(sd));current=sd;parts=[];sources=[]
    for j in range(10):
        p=np.fromfile(N/'data'/current['filename'],np.float32).reshape(-1,5)[:,:4].copy()
        if j:p=p[~((abs(p[:,0])<1)&(abs(p[:,1])<1))]
        rel=inv@world_lidar(current);p[:,:3]=p[:,:3]@rel[:3,:3].T+rel[:3,3]
        age=(sd['timestamp']-current['timestamp'])/1e6
        parts.append(np.c_[p,np.full(len(p),age,np.float32)])
        sources.append({'token':current['token'],'age_s':age,'points':len(p)})
        if j<9:
            if not current['prev']:raise RuntimeError('Missing causal sweep; do not silently pad')
            current=nusc.get('sample_data',current['prev'])
    return np.concatenate(parts).astype(np.float32),sources
def polygon(box):return Polygon(box.bottom_corners()[:2].T)
def compare(pred,gt,threshold):
    pp=[p for p in pred if p.score>=threshold];matches={}
    for metric in ['center2m','bev_iou0p5']:
        values=np.zeros((len(pp),len(gt)));valid=np.zeros_like(values,dtype=bool)
        for i,p in enumerate(pp):
            for j,g in enumerate(gt):
                if classes[p.label]!=g['class']:continue
                if metric=='center2m':
                    v=np.linalg.norm(p.center[:2]-g['box'].center[:2]);values[i,j]=max(0,2-v);valid[i,j]=v<=2
                else:
                    p0,g0=polygon(p),polygon(g['box']);v=p0.intersection(g0).area/max(p0.union(g0).area,1e-9);values[i,j]=v;valid[i,j]=v>=.5
        pairs=[]
        if len(pp) and len(gt):
            # 优先匹配数，再在合法匹配内按几何质量分配。
            ri,ci=linear_sum_assignment(-(valid*1000+values))
            for i,j in zip(ri,ci):
                if valid[i,j]:pairs.append({'instance':gt[j]['instance'],'class':gt[j]['class'],'score':float(pp[i].score),'center_error_m':float(np.linalg.norm(pp[i].center[:2]-gt[j]['box'].center[:2]))})
        matches[metric]={'matched':len(pairs),'eligible':len(gt),'matches':pairs}
    return matches

rows=[]
for i in range(a.frames):
    sd=nusc.get('sample_data',sample['data']['LIDAR_TOP'])
    _,boxes,_=nusc.get_sample_data(sd['token']);gt=[];lidar_to_ego=pose('calibrated_sensor',sd['calibrated_sensor_token'])
    for box in boxes:
        ann=nusc.get('sample_annotation',box.token);category=category_to_detection_name(ann['category_name']);p=lidar_to_ego[:3,:3]@box.center+lidar_to_ego[:3,3]
        if category is None or ann['num_lidar_pts']<5 or not (0<=p[0]<=32 and abs(p[1])<=16):continue
        gt.append({'class':category,'instance':ann['instance_token'],'box':box,'lidar_points':ann['num_lidar_pts']})
    gt_serial=[{k:v for k,v in g.items() if k!='box'}|{'center_lidar':g['box'].center.tolist(),'wlh':g['box'].wlh.tolist(),'rotation':g['box'].orientation.elements.tolist()} for g in gt]
    for condition in conditions:
        if a.scans_dir:
            points=np.load(a.scans_dir/f'{condition}_{i:02d}.npz')['points'];sources=str(a.scans_dir/'registration.json')
        else:
            points,sources=multisweep(sd);np.savez_compressed(a.out/f'input_{i:02d}.npz',points=points)
        result,packed=inference_detector(model,points)
        native=result.pred_instances_3d.to('cpu');pred,_=output_to_nusc_box(native)
        np.savez_compressed(a.out/f'prediction_{condition}_{i:02d}.npz',boxes=native.bboxes_3d.tensor.numpy(),scores=native.scores_3d.numpy(),labels=native.labels_3d.numpy())
        row={'index':i,'condition':condition,'sample_token':sample['token'],'lidar_token':sd['token'],'timestamp_us':sd['timestamp'],'points':len(points),'sweeps':sources,'eligible_gt':gt_serial,'scores':{str(t):compare(pred,gt,t) for t in [.3,.5]}}
        rows.append(row);(a.out/'summary.json').write_text(json.dumps(rows,indent=2));print('CENTERPOINT_RESULT',condition,i,{t:{m:v['matched'] for m,v in s.items()} for t,s in row['scores'].items()},'eligible',len(gt),flush=True)
    sample=nusc.get('sample',sample['next'])
reg.update(completed=True,end_unix=time.time(),detector_forwards=a.frames*len(conditions),failure_ledger_delta='pending_analysis')
(a.out/'registration.json').write_text(json.dumps(reg,indent=2));print('CENTERPOINT_COMPLETED',a.out,flush=True)

"""复用完成拟合的SplatAD，导出六相机日志配对；不训练、不运行策略。"""
import copy,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from splatad_compat import apply
apply()
from nerfstudio.utils.eval_utils import eval_setup

N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-SPARSE-NATIVE-01/20260915-r1');O=R/'scene0004_sensor_pairs_r1'
fit=N/'fits/scene-0004/splatad/native-r2';assert json.loads((fit/'fit_manifest.json').read_text())['completed']
O.mkdir(exist_ok=False);shutil.copy2(__file__,O/'source_snapshot.py')
meta=json.loads((N/'metadata/scene-0004.json').read_text());seq=sorted(meta['sample'].values(),key=lambda s:s['timestamp'])
order=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
reg={'task_id':'WS-SIM-SPARSE-NATIVE-SENSOR-01','run_id':O.name,'scene':'scene-0004','started_unix':time.time(),'seed':20260915,'fit':str(fit),'step':30000,
 'scope':'Prepare real/native-rendered six-camera inputs for the one native nuScenes planner. Original indices0..9: two warmup and eight candidate evaluation starts2..9. This known scene is not an independent log.',
 'selection':'All six cameras at each of ten contiguous keyframes, frozen before rendering or SparseDrive inference; no failure-based filtering.',
 'information':'Existing full-log RGB+LiDAR and annotated actor-trajectory scene fit is a strong extra-information reference, not feed-forward or equal budget. Per-image train/heldout role remains explicit.',
 'sensor_contract':'Use exact official dataset Cameras object at each source image: its calibrated pose, timestamp and native rolling metadata. No interpolated pose or synchronized camera-time replacement.',
 'crop_contract':'Save native render plus identical top-left valid support from original RGB as lossless PNG. CAM_BACK native bottom crop is recorded; common-support preprocessing must be applied equally for later planning. Original image remains referenced for an ordinary crop control.',
 'geometry_limit':'This stage only supplies complete six-camera pairs. Any planning discrepancy still requires appearance/timing/crop controls and actual local-asset repair before a geometry-causal claim.',
 'expected_camera_renders':60,'completed':False,'policy_forwards':0,'new_training_steps':0,'human_verdict':None}
(O/'registration.json').write_text(json.dumps(reg,indent=2))
torch.manual_seed(20260915);torch.set_num_threads(4)
def update(c):
    c.load_dir=fit/'nerfstudio_models';c.load_step=30000;c.pipeline.datamanager.cache_images='cpu';c.pipeline.datamanager.cache_lidars='cpu';c.pipeline.datamanager.max_thread_workers=4;c.pipeline.calc_fid_steps=();return c
config,pipeline,_,step=eval_setup(fit/'config.yml',test_mode='val',update_config_callback=update);dm=pipeline.datamanager;model=pipeline.model;model.eval()
lookup={}
for split,ds in [('train',dm.train_dataset),('heldout',dm.eval_dataset)]:
    for i,p in enumerate(ds.image_filenames):lookup[str(p)]=(split,ds,i)
rows=[]
with torch.inference_mode():
    for index,s in enumerate(seq[:10]):
        folder=O/f'frame_{index:02d}';folder.mkdir()
        for name in order:
            sd=meta['sample_data'][s['data'][name]];source=N/'data'/sd['filename'];split,ds,i=lookup[str(source)]
            camera=copy.deepcopy(ds.cameras[i:i+1]).to(model.device);camera.metadata['cam_idx']=i
            begin=time.time();pred=model.get_camera_outputs(camera);rgb=np.clip(pred['rgb'].detach().cpu().numpy()*255,0,255).astype(np.uint8);real=np.asarray(Image.open(source).convert('RGB'));h,w=rgb.shape[:2];assert real.shape[0]>=h and real.shape[1]>=w
            Image.fromarray(rgb).save(folder/f'{name}_render.png');Image.fromarray(real[:h,:w]).save(folder/f'{name}_real_common.png')
            expected_time=sd['timestamp']/1e6-float(dm.train_dataparser_outputs.time_offset);camera_time=float(camera.times.reshape(-1)[0]);assert abs(expected_time-camera_time)<1e-4
            row={'original_index':index,'sample_token':s['token'],'camera':name,'source_image':str(source),'split':split,'native_shape':list(rgb.shape),'original_shape':list(real.shape),'timestamp_us':sd['timestamp'],'camera_time_s':camera_time,'render_image':str(folder/f'{name}_render.png'),'real_common_image':str(folder/f'{name}_real_common.png'),'intrinsics':camera.get_intrinsics_matrices().detach().cpu().numpy().tolist(),'camera_to_world':camera.camera_to_worlds.detach().cpu().numpy().tolist(),'seconds':time.time()-begin,'RGB_MAE_0_255':float(np.mean(np.abs(rgb.astype(float)-real[:h,:w].astype(float))))}
            rows.append(row);(O/'pairs.json').write_text(json.dumps(rows,indent=2));print('PAIR',index,name,split,'shape',rgb.shape,flush=True);del pred,camera
reg.update(completed=True,completed_unix=time.time(),actual_camera_renders=len(rows),train_images=sum(r['split']=='train' for r in rows),heldout_images=sum(r['split']=='heldout' for r in rows),native_crop_count=sum(r['native_shape']!=r['original_shape'] for r in rows));(O/'registration.json').write_text(json.dumps(reg,indent=2));print('SIX_CAMERA_PAIRS_COMPLETED',len(rows),flush=True)

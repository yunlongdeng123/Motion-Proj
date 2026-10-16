"""只读旧/新相机与图像采样契约；无GPU，不修改基线或训练输入。"""
from pathlib import Path
import json,numpy as np,cv2,runpy,math
from PIL import Image
campaign=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1');root=campaign/'shadow_rebuild_20261017_v3'
data=root/'data_r2';m=json.loads((data/'manifest.json').read_text());old=json.loads((campaign/'rigid_runs/scene0008_089_137/full30k/cameras.json').read_text());old={r['img_name']:r for r in old}
with np.load(campaign/'rigid_prepared/scene0008_089_137/instances.npz') as f:norm=f['normalization'];scale=float(f['world_scale'])
origin=np.asarray(m['original_world_origin_m']);source=Path(m['source_scene'])
g=runpy.run_path(str(campaign/'rigid_source/scene0008_089_137/utils/graphics_utils.py'))
samples=np.array([[0,0,10],[2,1,10],[-3,-2,20],[4,-1,50]],float)
records=[]
for row in m['frames']:
 camera=old[row['frame_id']];pose=np.asarray(row['world_from_camera']);k=np.asarray(row['intrinsic'])
 world=samples@pose[:3,:3].T+pose[:3,3]+origin
 normalized=world@norm[:3,:3].T+norm[:3,3]
 old_cam=(normalized-np.asarray(camera['position']))@np.asarray(camera['rotation'])
 h,w=320,480;ratio=w/camera['width']
 P=g['getProjectionMatrixCenterShift'](.01,1000,camera['cx']*ratio,camera['cy']*ratio,camera['fx']*ratio,camera['fy']*ratio,w,h).numpy()
 hp=np.concatenate((old_cam,np.ones((len(old_cam),1))),1)@P.T
 ndc=hp[:,:2]/hp[:,3:4]
 old_uv=((ndc+1)*np.array([w,h])-1)/2
 new_uv=(samples@k.T);new_uv=new_uv[:,:2]/new_uv[:,2:]
 center=(pose[:3,3]+origin)@norm[:3,:3].T+norm[:3,3]
 rotation=(norm[:3,:3]/scale)@pose[:3,:3]
 original=np.asarray(Image.open(source/'image/image_0'/(row['frame_id']+'.png')).convert('RGB'),dtype=np.float32)/255
 cv=cv2.resize(original,(w,h),interpolation=cv2.INTER_LINEAR)
 new=np.asarray(Image.open(data/row['rgb']).convert('RGB'),dtype=np.float32)/255
 error=(new-cv)*255
 records.append(dict(frame_id=row['frame_id'],normalized_camera_center_max_error=float(abs(center-np.asarray(camera['position'])).max()),
                     camera_rotation_max_error=float(abs(rotation-np.asarray(camera['rotation'])).max()),
                     old_minus_new_pixels=(old_uv-new_uv).tolist(),new_PIL_vs_legacy_cv2_RGB_MAE_8bit=float(abs(error).mean()),
                     new_PIL_vs_legacy_cv2_RGB_max_8bit=float(abs(error).max())))
delta=np.array([r['old_minus_new_pixels'] for r in records]);count=delta.reshape(-1,2)
result=dict(scope='CPU source/measurement pairing audit, not a rendered-quality result',frames=len(records),world_scale=scale,
            camera_center_max_error_normalized=max(r['normalized_camera_center_max_error'] for r in records),
            camera_rotation_max_error=max(r['camera_rotation_max_error'] for r in records),
            old_minus_new_pixel_delta_mean=count.mean(0).tolist(),pixel_delta_standard_deviation=count.std(0).tolist(),
            RGB_sampler_MAE_8bit_mean=float(np.mean([r['new_PIL_vs_legacy_cv2_RGB_MAE_8bit'] for r in records])),
            legacy_raster_pixel_convention='ndc2Pix=((ndc+1)*size-1)/2 with cx=raw_cx/4, cy=raw_cy/4',
            new_pixel_convention='pixel-center resized K: cx=raw_cx/4-0.375; raster projection explicitly targets integer-center coordinates',
            GT_sampler_legacy='cv2.INTER_LINEAR on raw floatRGB per copied loader',GT_sampler_new='PIL bilinear resized uint8 images',
            required_policy='For model-to-model metrics, evaluate against one explicitly chosen common observed RGB and correct the measured camera projection shift in a separate read-only rerender. Keep legacy-native views and metrics; no silent image translation.',
            normalized_pose_contract_matches=bool(max(r['normalized_camera_center_max_error'] for r in records)<1e-4 and max(r['camera_rotation_max_error'] for r in records)<1e-4),
            original_checkpoints_or_inputs_modified=False,records=records,human_verdict=None)
(root/'camera_pairing_audit_r1.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'}))

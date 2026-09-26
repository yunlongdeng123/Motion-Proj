"""连续真实帧的原始/factual/MOVE六相机视频；不插帧、不做时序平滑。"""
import argparse,datetime,json,pathlib,sys,time,subprocess
from fractions import Fraction
import numpy as np
import torch
from PIL import Image
from geometry import box_mask,edit_actor,project_bbox,target_pose,transform
from evaluate import dump,read_geometry,render

BASE=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-P0-24ACTOR-20260926/r1')
TEMP=BASE.parents[1]/'WS-V77-P0-TEMPORAL-20260926/r1'
KINDS=['original','factual','edited']
TARGETS={'official_000':'14','scene_0230':'22','scene_0255':'25'}

def prepare(root):
 old=json.loads((BASE/'registration.json').read_text());scenes=[]
 root.mkdir(parents=True,exist_ok=True);assert not (root/'registration.json').exists()
 for s in old['scenes']:
  data=pathlib.Path(s['root']);frames=sorted(int(p.stem.split('_')[0]) for p in (data/'images').glob('*_0.jpg'))
  assert frames==list(range(max(frames)+1))
  for f in frames:
   for c in range(6):
    assert (data/'images'/f'{f:03d}_{c}.jpg').exists()
    assert (data/'extrinsics'/f'{f:03d}_{c}.txt').exists()
   assert (data/'lidar'/f'{f:03d}.bin').exists()
   assert (data/'lidar_pose'/f'{f:03d}.txt').exists()
  actor=next(a for a in s['actors'] if a['actor_id']==TARGETS[s['name']])
  lateral=np.array(actor['pose'])[:3,1];lateral[2]=0;lateral=lateral/np.linalg.norm(lateral)*2
  inst=json.loads((data/'instances/instances_info.json').read_text())
  out=root/s['name'];out.mkdir();dump(out/'instances_snapshot.json',inst)
  scenes.append(dict(s,frames=frames,video_actor=actor,delta_world_m=lateral.tolist(),fps=10))
 reg={'task_id':'WS-V77-VIDEO-REVIEW-20260926','run_dir':str(root),'registered_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'scenes':scenes,
  'seed':7701,'omega_source':old['omega_source'],'checkpoint':old['checkpoint'],
  'readout':'calibrated_control: GT K/c2w + per-frame background LiDAR scale; same fixed P0 rule',
  'selection':'用户要求连续视频；完整现有processed序列；展示前轮已审阅的truck14/car22/car25，不择优替换目标。六相机全部输出。',
  'temporal_mode':'每时刻六相机独立冻结前向，未跨时刻融合/平滑/插帧；这是当前适配管线的观测时序回放，不是持久4D资产或未见视角验证',
  'timing':'按processed 10Hz等间距回放，原始精确camera timestamps不可得',
  'edit':'MOVE，目标每帧GT位姿加同一世界位移；位移为frame20目标水平侧向2m，yaw和尺寸不变；目标缺GT时不猜测、不编辑',
  'cached_predictions':[str(BASE),str(TEMP)],'failure_ledger_refs':['V77-F01'],'failure_ledger_delta':'none','human_verdict':None}
 dump(root/'registration.json',reg)
 print(json.dumps({'frames':{s['name']:len(s['frames']) for s in scenes},'total':sum(len(s['frames']) for s in scenes)},ensure_ascii=False),flush=True)

def scene_frame(spec,frame,instances):
 s=dict(spec,frame=frame,views=[],all_boxes=[])
 data=pathlib.Path(s['root'])
 for v in spec['views']:
  c=v['camera'];s['views'].append(dict(v,image=str(data/'images'/f'{frame:03d}_{c}.jpg'),c2w=np.loadtxt(data/'extrinsics'/f'{frame:03d}_{c}.txt').tolist()))
 s['origin_world']=np.mean([np.array(v['c2w'])[:3,3] for v in s['views']],0).tolist()
 for aid,r in instances.items():
  fa=r['frame_annotations']
  if frame not in fa['frame_idx']:continue
  j=fa['frame_idx'].index(frame)
  s['all_boxes'].append({'actor_id':aid,'track_id':r['id'],'category':r['class_name'],'pose':fa['obj_to_world'][j],'size_lwh':fa['box_size'][j]})
 return s

def mosaic(arrays):return np.concatenate([np.concatenate(arrays[:3],axis=1),np.concatenate(arrays[3:],axis=1)],axis=0)

def render_frames(root,limit=None):
 reg=json.loads((root/'registration.json').read_text());torch.set_num_threads(4);torch.manual_seed(reg['seed'])
 assert torch.cuda.is_available(),'连续序列需GPU；禁止用三个缓存时刻伪造视频'
 sys.path.insert(0,reg['omega_source'])
 from vggt_omega.models import VGGTOmega
 from vggt_omega.utils.load_fn import load_and_preprocess_images
 from vggt_omega.utils.pose_enc import encoding_to_camera
 model=None;done=0
 for spec in reg['scenes']:
  out=root/spec['name'];instances=json.loads((out/'instances_snapshot.json').read_text())
  for frame in spec['frames']:
   fd=out/'frames'/f'{frame:03d}';fd.mkdir(parents=True,exist_ok=True)
   if (fd/'metrics.json').exists() and all((fd/(k+'.png')).exists() for k in KINDS):continue
   start=time.monotonic();scene=scene_frame(spec,frame,instances)
   if frame==20:cached=BASE/spec['name']/'prediction.npz'
   elif frame in [0,40]:cached=TEMP/(spec['name']+f'_f{frame:03d}')/'prediction.npz'
   else:cached=fd/'prediction.npz'
   is_old=frame in [0,20,40]
   if cached.exists():
    with np.load(cached) as saved:pred={k:saved[k] for k in saved.files}
    new_forward=False;runtime=0.;peak=0.
    if 'images' not in pred:
     pred['images']=load_and_preprocess_images([v['image'] for v in scene['views']],image_resolution=512,mode='balanced').numpy()
   else:
    if model is None:
     with torch.device('meta'):model=VGGTOmega()
     model.load_state_dict(torch.load(reg['checkpoint'],map_location='cpu',weights_only=True,mmap=True),strict=True,assign=True)
     model.requires_grad_(False);model=model.eval().cuda();print('MODEL_READY',flush=True)
    images=load_and_preprocess_images([v['image'] for v in scene['views']],image_resolution=512,mode='balanced').cuda()
    torch.cuda.reset_peak_memory_stats();t=time.monotonic()
    with torch.inference_mode():
     raw=model(images);e,k=encoding_to_camera(raw['pose_enc'],images.shape[-2:])
    torch.cuda.synchronize();runtime=time.monotonic()-t;peak=torch.cuda.max_memory_allocated()/2**30
    pred={k:raw[k].float().cpu().numpy() for k in ['depth','depth_conf','pose_enc']}
    pred.update(extrinsics=e.float().cpu().numpy(),intrinsics=k.float().cpu().numpy())
    assert all(np.isfinite(v).all() for v in pred.values())
    np.savez_compressed(cached,**pred);pred['images']=images.float().cpu().numpy()
    del raw,images,e,k;new_forward=True
   assert all(np.isfinite(v).all() for v in pred.values())
   variants,cams,ks,lidar,original,alignment=read_geometry(scene,pred,fd)
   points,colors,_=variants['calibrated_control'];del variants,lidar,pred
   h,w=original.shape[1:3];assert (h,w)==(384,688)
   box=next((b for b in scene['all_boxes'] if b['actor_id']==spec['video_actor']['actor_id']),None)
   selected=np.zeros(len(points),dtype=bool);source_boxes=[None]*6;target_boxes=[None]*6;error=None
   if box:
    assert box['track_id']==spec['video_actor']['track_id']
    pose=np.array(box['pose']);pose[:3,3]-=np.array(scene['origin_world']);size=box['size_lwh']
    selected=box_mask(points,pose,size);target=target_pose(pose,spec['delta_world_m'])
    edited,edited_colors,ids=edit_actor(points,colors,selected,pose,target,'MOVE')
    assert np.array_equal(edited[~selected],points[~selected]) and np.array_equal(edited_colors,colors)
    if selected.any():
     error=float(np.linalg.norm((edited[selected]-points[selected])-np.array(spec['delta_world_m']),axis=1).max());assert error<1e-8
    source_boxes=[project_bbox(pose,size,c,k,[h,w]) for c,k in zip(cams,ks)]
    target_boxes=[project_bbox(target,size,c,k,[h,w]) for c,k in zip(cams,ks)]
   else:edited=points;edited_colors=colors
   factual=[render(points,colors,c,k,(h,w))[0] for c,k in zip(cams,ks)]
   moved=[render(edited,edited_colors,c,k,(h,w))[0] for c,k in zip(cams,ks)] if selected.any() else factual
   for kind,arrays in zip(KINDS,[original,factual,moved]):Image.fromarray(mosaic(arrays)).save(fd/(kind+'.png'),compress_level=2)
   metric={'frame':frame,'time_s':frame/10,'gt_present':box is not None,'target_point_count':int(selected.sum()),
    'source_boxes':source_boxes,'target_boxes':target_boxes,'delta_world_m':spec['delta_world_m'],'non_target_max_change_m':0.,
    'max_point_adherence_error_m':error,'depth_scale':alignment['calibrated_global_depth_scale'],
    'anchor_count':alignment['calibration_background_anchor_count'],'finite':True,'prediction_path':str(cached),
    'reused_previous_p0':is_old,'new_forward_this_attempt':new_forward,'forward_s':runtime,'peak_forward_gpu_gib':peak,
    'camera_c2w_world':[v['c2w'] for v in scene['views']],'source_gt_box':box,'elapsed_s':time.monotonic()-start,'human_verdict':None}
   dump(fd/'metrics.json',metric)
   print(json.dumps({'scene':spec['name'],'frame':frame,'total':len(spec['frames']),'points':metric['target_point_count'],'seconds':round(metric['elapsed_s'],2)}),flush=True)
   del points,colors,edited,edited_colors,original,factual,moved;done+=1
   if limit and done>=limit:return

def encode(root,scene_name=None):
 import av
 reg=json.loads((root/'registration.json').read_text());delivery=root/'review';delivery.mkdir(exist_ok=True)
 previous=json.loads((delivery/'review_data.json').read_text()) if (delivery/'review_data.json').exists() else {'scenes':[]}
 completed={s['name']:s for s in previous['scenes']}
 for spec in reg['scenes']:
  if scene_name and spec['name']!=scene_name:continue
  out=root/spec['name'];dest=delivery/spec['name'];dest.mkdir(exist_ok=True)
  rows=[json.loads((out/'frames'/f'{f:03d}'/'metrics.json').read_text()) for f in spec['frames']]
  videos={}
  for kind in KINDS:
   target=dest/(kind+'.mp4');temp=dest/(kind+'.encoding.mp4')
   with av.open(str(temp),'w',options={'movflags':'+faststart'}) as container:
    stream=container.add_stream('libx264',rate=10);stream.width=2064;stream.height=768;stream.pix_fmt='yuv420p';stream.thread_count=4
    stream.options={'crf':'18','preset':'fast','g':'10','bf':'0'}
    for i,f in enumerate(spec['frames']):
     rgb=np.asarray(Image.open(out/'frames'/f'{f:03d}'/(kind+'.png')).convert('RGB'))
     frame=av.VideoFrame.from_ndarray(rgb,format='rgb24');frame.pts=i;frame.time_base=Fraction(1,10)
     for packet in stream.encode(frame):container.mux(packet)
    for packet in stream.encode():container.mux(packet)
   temp.replace(target)
   # 逐一解码所有输出帧，检查尺寸、时戳和帧数，而非仅检查容器声明。
   with av.open(str(target)) as check:
    count=0;last=-1.;first=None
    for frame in check.decode(video=0):
     assert (frame.width,frame.height)==(2064,768);assert frame.time>last;last=frame.time;first=frame.time if first is None else first;count+=1
    assert count==len(spec['frames']) and abs(first)<1e-6 and abs(last-(count-1)/10)<1e-6
   poster=np.asarray(Image.open(out/'frames'/'020'/(kind+'.png')).convert('RGB'));Image.fromarray(poster).save(dest/(kind+'_poster.jpg'),quality=93)
   videos[kind]={'path':str(target.relative_to(delivery)),'frames':count,'fps':10,'duration_s':count/10,'decoded_last_pts_s':last,'bytes':target.stat().st_size}
   print(json.dumps({'encoded':str(target),'frames':count,'duration_s':count/10}),flush=True)
  specdata={'name':spec['name'],'actor_id':spec['video_actor']['actor_id'],'track_id':spec['video_actor']['track_id'],
   'category':spec['video_actor']['category'],'default_camera':spec['video_actor']['primary_camera'],'delta_world_m':spec['delta_world_m'],
   'frame_count':len(rows),'fps':10,'videos':videos,'frames':rows,
   'gt_present_frames':sum(r['gt_present'] for r in rows),'nonempty_edit_frames':sum(r['target_point_count']>0 for r in rows),
   'scale_min':min(r['depth_scale'] for r in rows),'scale_max':max(r['depth_scale'] for r in rows)}
  dump(dest/'sequence_metrics.json',specdata);completed[spec['name']]=specdata
 scenes=[completed[s['name']] for s in reg['scenes'] if s['name'] in completed]
 summary={'task_id':reg['task_id'],'scenes':scenes,'total_scene_frames':sum(s['frame_count'] for s in scenes),
  'model_forwards_reused':3*len(scenes),'model_forwards_new':sum(s['frame_count'] for s in scenes)-3*len(scenes),'training_steps':0,
  'failure_ledger_refs':['V77-F01'],'failure_ledger_delta':'none','human_verdict':None}
 dump(delivery/'review_data.json',summary)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','render','encode']);p.add_argument('--run-dir',required=True);p.add_argument('--limit',type=int);p.add_argument('--scene',choices=list(TARGETS));a=p.parse_args()
 root=pathlib.Path(a.run_dir)
 if a.phase=='prepare':prepare(root)
 elif a.phase=='render':render_frames(root,a.limit)
 else:encode(root,a.scene)

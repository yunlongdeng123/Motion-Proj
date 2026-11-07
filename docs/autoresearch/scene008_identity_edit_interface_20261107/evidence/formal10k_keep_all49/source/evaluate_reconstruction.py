"""KEEP-only 重建检查：先97/105/109，再显式49；固定原a5/upper域。"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback
import numpy as np
from PIL import Image

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full_dynamic_edit_20261107_r1')
ENGINE=ROOT.parent/'r15_baseline_owner5_20261018/training_source/scene0008_089_137'
API=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82')
OBS=ROOT.parent/'scene008_instances_20261105_r1/observations/frames'
PILOT=(97,105,109)
FULL_SHA='aa8e7d3b56a8c35ce6177ad1e3378df7afcbc83e6db5d813bdddbfc55a717419'
UPPER_SHA='d5c24aeee22bb804311e28f89de6f0652438465263c3e45a2f9d245d6b67379f'
GT_SHA='e1865e95d0f6e549bb104f7cec64f3a339bb23cb824ed471b269c2c6b8c0079a'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def measure(rgb,gt,alpha):
 n=len(rgb);error=np.abs(np.clip(rgb,0,1).astype(np.float64)-gt.astype(np.float64)).mean(1)
 return dict(pixels=n,rgb_error_sum=float(error.sum()),alpha_sum=float(alpha.astype(np.float64).sum()),
  coverage_counts={str(t):int((alpha>=t).sum()) for t in [.1,.5,.9,.99]})
def aggregate(rows,key):
 n=sum(r[key]['pixels'] for r in rows)
 return dict(pixels=n,keep_rgb_mae=sum(r[key]['rgb_error_sum'] for r in rows)/n,
  keep_mean_alpha=sum(r[key]['alpha_sum'] for r in rows)/n,
  keep_coverage={str(t):sum(r[key]['coverage_counts'][str(t)] for r in rows)/n for t in [.1,.5,.9,.99]},
  weighting='Fixed pixel-frame denominator, not frame-mean')
def identity_measure(channels,alpha,select,actors):
 values=channels[:,select].astype(np.float64);n=values.shape[1];index=1+list(actors).index(5)
 return dict(pixels=n,owner5_alphaT_sum=float(values[index].sum()),static_alphaT_sum=float(values[0].sum()),
  other_owner_alphaT_sum=float((values[1:].sum(0)-values[index]).sum()),
  environment_transmission_sum=float((1-alpha[select].astype(np.float64)).sum()),
  argmax_owner5_count=int((np.argmax(values,axis=0)==index).sum()))
def identity_aggregate(rows,key):
 n=sum(r[key]['pixels'] for r in rows)
 return dict(pixels=n,owner5_alphaT_mean=sum(r[key]['owner5_alphaT_sum'] for r in rows)/n,
  static_alphaT_mean=sum(r[key]['static_alphaT_sum'] for r in rows)/n,
  other_owner_alphaT_mean=sum(r[key]['other_owner_alphaT_sum'] for r in rows)/n,
  environment_transmission_mean=sum(r[key]['environment_transmission_sum'] for r in rows)/n,
  argmax_owner5_rate=sum(r[key]['argmax_owner5_count'] for r in rows)/n,
  argmax_domain='raw joint static channel plus five dynamic actor alphaT channels; no alpha threshold selection')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--iteration',type=int,required=True)
 ap.add_argument('--output',type=Path,required=True);ap.add_argument('--scope',choices=['pilot','all49'],default='pilot');ap.add_argument('--pilot-reference',type=Path)
 opts=ap.parse_args();opts.model=opts.model.resolve();opts.output=opts.output.resolve()
 assert opts.model.is_relative_to(ROOT) and opts.output.is_relative_to(ROOT) and not opts.output.exists()
 assert opts.scope!='all49' or opts.pilot_reference is not None,'Full49 follows matching KEEP pilot'
 common=load('reconstruction_native_common',ROOT/'source/native_evaluation_common.py')
 helpers=load('reconstruction_readonly_helpers',ROOT/'source/evaluate_scene.py')
 opts.engine=ENGINE;opts.observations=OBS;opts.expected_mask_pixels=121128;opts.crop=[0,160,175,320]
 paths=[Path(__file__),ROOT/'source/evaluate_scene.py',ROOT/'source/native_evaluation_common.py',*common.source_model_files(opts.model,opts.iteration)]
 paths+=sorted(ENGINE.rglob('*.py'))+sorted((API/'drivingeditor_dynamic_edit').glob('*.py'))+[OBS/f'{r:03d}.npz' for r in range(89,138)]
 if opts.pilot_reference:
  previous=common.read_json(opts.pilot_reference/'summary.json')
  assert previous['status']=='PASS_NATIVE_KEEP_RECONSTRUCTION_PILOT' and previous['iteration']==opts.iteration and Path(previous['model']).resolve()==opts.model
  paths+=[opts.pilot_reference/'summary.json']+[opts.pilot_reference/f'frames/{r:03d}/keep.png' for r in PILOT]
 opts.output.mkdir();before=common.inventory(paths);write(opts.output/'source_manifest_before.json',dict(files=before))
 error=None;summary=None;started=time.monotonic()
 try:
  frames,full_sha,crop=common.observation_headers(opts,common.read_json(opts.model/'cameras.json'),common.read_json(opts.model/'actual_scene_binding.json'))
  assert full_sha==FULL_SHA
  upper_hash=hashlib.sha256();gt_hash=hashlib.sha256();full_count=0;upper_count=0
  for frame in frames:
   raw=frame['raw'];yx=np.column_stack(np.nonzero(frame['mask'])).astype('<i2');select=np.zeros(len(yx),dtype=bool)
   if len(yx):select=yx[:,0]<=yx[:,0].min()+.25*(yx[:,0].max()-yx[:,0].min())
   frame['upper_select']=select
   with np.load(frame['path'],allow_pickle=False) as archive:gt=np.ascontiguousarray(archive['rgb'][frame['mask']],dtype='<f4')
   upper_hash.update(np.asarray(raw,dtype='<i2').tobytes());upper_hash.update(np.ascontiguousarray(yx[select],dtype='<i2').tobytes())
   gt_hash.update(np.asarray(raw,dtype='<i2').tobytes());gt_hash.update(gt.tobytes())
   full_count+=len(yx);upper_count+=int(select.sum())
  assert full_count==121128 and upper_count==25906 and upper_hash.hexdigest()==UPPER_SHA and gt_hash.hexdigest()==GT_SHA
  chosen=frames if opts.scope=='all49' else [f for f in frames if f['raw'] in PILOT]
  sys.path[:0]=[str(ENGINE),str(API)]
  import torch,gaussian_renderer
  from scene.cameras import Camera
  from drivingeditor_dynamic_edit import install_identity_serialization,actor_ids,render_scene
  install_identity_serialization();torch.set_num_threads(1)
  config=common.read_json(opts.model/'effective_config.json');opts.dynamic_aliases=config['dynamic_actor_ids']
  args,static,dynamic,env=common.restore_models(opts,config);actors=tuple(dynamic._dynamic_actor_ids_declared);owners=actor_ids(dynamic)
  models=dict(static=static,dynamic=dynamic,environment=env);baseline=helpers.fingerprints(models,torch)
  current_counts={str(a):int((owners==a).sum()) for a in actors}
  source_counts={str(a):int((dynamic._dynamic_actor_source_ids==a).sum()) for a in actors}
  static_tags=torch.zeros((len(static.get_xyz),len(actors)+1),device='cuda');static_tags[:,0]=1.
  dynamic_tags=torch.cat([torch.zeros((len(owners),1),device='cuda'),torch.stack([owners==a for a in actors],1).float()],dim=1)
  extras=dict(env_map=env,other=[static_tags],dynamic_other=[dynamic_tags],is_training=False)
  background=torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda')
  (opts.output/'a5_float').mkdir();(opts.output/'anchors').mkdir();rows=[];replay=[];partition_max=0.
  for frame in chosen:
   raw,mask,K,W=[frame[k] for k in ['raw','mask','K','W']]
   camera=Camera(raw,W[:3,:3].T.copy(),W[:3,3].copy(),cx=K[0,2],cy=K[1,2],fx=K[0,0],fy=K[1,1],
    image=torch.zeros((3,320,480)),image_name=f'0008{raw:03d}',resolution=(480,320),timestamp=frame['timestamp'])
   assert np.allclose(camera.world_view_transform.detach().cpu().numpy().T,W,atol=1e-5,rtol=0)
   with torch.no_grad():package=render_scene(camera,static,dynamic,args,background,dynamic_actor_ids=actors,delete_actor_ids=[],renderer=gaussian_renderer,**extras)
   keep=common.tensor_hwc(package['render']);alpha=common.tensor_alpha(package['alpha']);feature=package['feature'].detach().cpu().numpy()
   assert np.isfinite(keep).all() and np.isfinite(alpha).all() and np.isfinite(feature).all()
   assert feature.shape==(len(actors)+1,320,480)
   partition_error=float(np.abs(feature.sum(0)-alpha).max());partition_max=max(partition_max,partition_error)
   assert partition_error<=1e-5,'Raw disjoint identity feature mass does not equal original joint alpha'
   with np.load(frame['path'],allow_pickle=False) as archive:gt=np.ascontiguousarray(archive['rgb'],dtype=np.float32)
   yx=np.column_stack(np.nonzero(mask)).astype(np.int16);upper=frame['upper_select'];rgb=keep[mask];target=gt[mask];opacity=alpha[mask]
   np.savez_compressed(opts.output/f'a5_float/{raw:03d}.npz',raw=np.int16(raw),yx=yx,upper_selected=upper,
    keep_rgb=rgb,keep_alpha=opacity,gt_rgb=target,joint_alphaT=feature[:,mask].T,actor_channel_ids=np.asarray([0]+list(actors),dtype=np.int64),mask_sha256=np.asarray(frame['mask_sha256']),model_time=np.float64(frame['timestamp']))
   directory=opts.output/f'frames/{raw:03d}';directory.mkdir(parents=True);image=common.rgb8(keep)
   Image.fromarray(image).save(directory/'keep.png');Image.fromarray(common.rgb8(gt)).save(directory/'gt.png')
   if raw in PILOT:common.contact_sheet(opts.output/f'anchors/{raw:03d}_crop.png',[common.rgb8(gt),image],['GT','NATIVE KEEP'],crop)
   if opts.pilot_reference and raw in PILOT:
    previous=np.asarray(Image.open(opts.pilot_reference/f'frames/{raw:03d}/keep.png').convert('RGB'))
    maximum=int(np.abs(image.astype(np.int16)-previous.astype(np.int16)).max());assert maximum<=1;replay.append(dict(raw=raw,max_uint8_difference=maximum))
   contributions={str(a):dict(max=float(feature[i+1].max()),sum=float(feature[i+1].astype(np.float64).sum()),positive_pixels=int((feature[i+1]>1e-6).sum()),
    a5_full_mean=float(feature[i+1][mask].mean()) if len(yx) else None,
    a5_upper_mean=float(feature[i+1][mask][upper].mean()) if upper.any() else None) for i,a in enumerate(actors)}
   rows.append(dict(raw=raw,full=measure(rgb,target,opacity),upper=measure(rgb[upper],target[upper],opacity[upper]),
    joint_owner_contributions=contributions,identity_full=identity_measure(feature[:,mask],opacity,np.ones(len(yx),dtype=bool),actors),
    identity_upper=identity_measure(feature[:,mask],opacity,upper,actors),identity_partition_max_abs=partition_error,
    fixed_mask_sha256=frame['mask_sha256']))
   print(json.dumps(dict(raw=raw,a5_pixels=len(yx),upper_pixels=int(upper.sum()),keep_rgb_mae=rows[-1]['full']['rgb_error_sum']/len(yx) if len(yx) else None)),flush=True)
   del package,camera
  assert helpers.fingerprints(models,torch)==baseline
  selected=aggregate(rows,'full');upper_selected=aggregate(rows,'upper')
  summary=dict(status='PASS_NATIVE_KEEP_RECONSTRUCTION_PILOT' if opts.scope=='pilot' else 'PASS_NATIVE_KEEP_RECONSTRUCTION_ALL49',
   scope=opts.scope,model=str(opts.model),iteration=opts.iteration,frame_count=len(chosen),raw_ids=[f['raw'] for f in chosen],frames=rows,
   dynamic_actor_ids=list(actors),delete_actor_ids=[],current_owner_rows=current_counts,initial_source_owner_rows=source_counts,
   native_static_rows=len(static.get_xyz),native_dynamic_rows=len(dynamic.get_xyz),selected=selected,upper_selected=upper_selected,
   full49_fixed_reference=dict(full_pixels=121128,upper_pixels=25906,full_mask_sha256=FULL_SHA,upper_index_sha256=UPPER_SHA,gt_sha256=GT_SHA),
   upper_rule='y <= min_y+.25*(max_y-min_y), from original per-frame a5 mask; empty frame contributes0',
   identity_selected=identity_aggregate(rows,'identity_full'),upper_identity_selected=identity_aggregate(rows,'identity_upper'),
   identity_partition_max_abs_error=partition_max,identity_feature_is_normalized=False,identity_channel_order=[0]+list(actors),
   parameters_unchanged=True,optimizer_created=False,optimization_updates=0,source_integrity='PENDING',replay=replay,
   identity_is_saved_source_ownership=True,alpha_or_owner_is_not_body_quality_pass=True,
   body_quality_verdict=None,quality_accepted=False,human_verdict=None)
  if opts.scope=='all49':summary.update(all49=selected,upper_all49=upper_selected,identity_all49=summary['identity_selected'],upper_identity_all49=summary['upper_identity_selected'])
 except BaseException:error=traceback.format_exc()
 finally:
  after=common.inventory(paths);changed=[p for p in before if before[p]!=after.get(p)]
  write(opts.output/'source_integrity.json',dict(status='PASS_UNCHANGED' if not changed else 'FAIL',changed=changed,before=before,after=after))
 if error or changed:
  write(opts.output/'failure.json',dict(traceback=error,changed=changed));raise RuntimeError(error or str(changed))
 summary['source_integrity']='PASS_UNCHANGED_BYTES_MTIME_SHA256';summary['seconds']=time.monotonic()-started
 write(opts.output/'summary.json',summary)
if __name__=='__main__':main()

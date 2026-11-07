"""原生全动态场景：只读切换删除目标，三帧 pilot 后才显式扩展49帧。"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback
import numpy as np
from PIL import Image

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full_dynamic_edit_20261107_r1')
ENGINE=ROOT.parent/'r15_baseline_owner5_20261018/training_source/scene0008_089_137'
API=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82')
OBS=ROOT.parent/'scene008_instances_20261105_r1/observations/frames'
PILOT=(105,130,137)
ACTORS=(5,6,18,26,37)
QUERIES={'keep':(),'delete_a37':(37,),'delete_a6':(6,),'delete_a6_a37':(6,37),'keep_again':()}

def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def fingerprints(models,torch):
 out={}
 for group,model in models.items():
  values={name:value for name,value in vars(model).items() if torch.is_tensor(value)}
  if callable(getattr(model,'named_parameters',None)):values.update(dict(model.named_parameters()))
  if callable(getattr(model,'named_buffers',None)):values.update(dict(model.named_buffers()))
  for name,value in values.items():out[group+'.'+name]=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
 if 'environment' in models:assert 'environment.base' in out,'Registered environment.base was not inventoried'
 return out
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--model',type=Path,required=True);ap.add_argument('--iteration',type=int,required=True)
 ap.add_argument('--output',type=Path,required=True);ap.add_argument('--scope',choices=['pilot','all49'],default='pilot')
 ap.add_argument('--pilot-reference',type=Path)
 opts=ap.parse_args();opts.model=opts.model.resolve();opts.output=opts.output.resolve()
 assert opts.model.is_relative_to(ROOT) and opts.output.is_relative_to(ROOT) and not opts.output.exists()
 assert opts.scope!='all49' or opts.pilot_reference is not None,'Full49 must follow a matching completed pilot'
 common_path=Path(__file__).with_name('native_evaluation_common.py')
 assert hashlib.sha256(common_path.read_bytes()).hexdigest()=='be18a16926f8ebebe345dd7e1ae5cf6ce80378efb2fa6c43ed7ab3271ecaf1dc'
 spec=importlib.util.spec_from_file_location('identity_native_evaluation_common',common_path)
 common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
 opts.engine=ENGINE;opts.observations=OBS;opts.expected_mask_pixels=121128;opts.crop=[0,160,175,320];opts.dynamic_aliases=list(ACTORS)
 paths=[Path(__file__),common_path,*common.source_model_files(opts.model,opts.iteration)]
 paths+=sorted(ENGINE.rglob('*.py'))+sorted((API/'drivingeditor_dynamic_edit').glob('*.py'))
 paths+=[OBS/f'{raw:03d}.npz' for raw in range(89,138)]
 if opts.pilot_reference:
  pilot=common.read_json(opts.pilot_reference/'summary.json')
  assert pilot['status']=='PASS_NATIVE_IDENTITY_EDIT_PILOT_ENGINEERING' and pilot['iteration']==opts.iteration
  assert Path(pilot['model']).resolve()==opts.model
  paths+=[opts.pilot_reference/'summary.json']+[opts.pilot_reference/f'frames/{raw:03d}/{q}.png' for raw in PILOT for q in QUERIES]
 opts.output.mkdir();before=common.inventory(paths);write(opts.output/'source_manifest_before.json',dict(files=before))
 started=time.monotonic();error=None;result=None
 try:
  frames,denominator_sha,crop=common.observation_headers(opts,common.read_json(opts.model/'cameras.json'),common.read_json(opts.model/'actual_scene_binding.json'))
  frames=frames if opts.scope=='all49' else [f for f in frames if f['raw'] in PILOT]
  sys.path[:0]=[str(ENGINE),str(API)]
  import torch
  import gaussian_renderer
  from scene.cameras import Camera
  from drivingeditor_dynamic_edit import install_identity_serialization,actor_ids,render_scene
  install_identity_serialization();torch.set_num_threads(1)
  assert torch.cuda.is_available()
  config=common.read_json(opts.model/'effective_config.json')
  args,static,dynamic,env=common.restore_models(opts,config)
  owners=actor_ids(dynamic);assert tuple(dynamic._dynamic_actor_ids_declared)==ACTORS
  models=dict(static=static,dynamic=dynamic,environment=env)
  baseline=fingerprints(models,torch)
  source_counts={str(a):int((owners==a).sum()) for a in ACTORS}
  bg=torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda')
  static_tags=torch.zeros((len(static.get_xyz),len(ACTORS)),device='cuda')
  dynamic_tags=torch.stack([owners==a for a in ACTORS],dim=1).float()
  extras=dict(env_map=env,other=[static_tags],dynamic_other=[dynamic_tags],is_training=False)
  (opts.output/'a5_float').mkdir();(opts.output/'anchors').mkdir()
  rows=[];native_max=0.;again_max=0.;replay_max=0;target37_witness=False
  for frame in frames:
   raw,mask,K,W=[frame[k] for k in ['raw','mask','K','W']]
   camera=Camera(raw,W[:3,:3].T.copy(),W[:3,3].copy(),cx=K[0,2],cy=K[1,2],fx=K[0,0],fy=K[1,1],
    image=torch.zeros((3,320,480)),image_name=f'0008{raw:03d}',resolution=(480,320),timestamp=frame['timestamp'])
   assert np.allclose(camera.world_view_transform.detach().cpu().numpy().T,W,atol=1e-5,rtol=0)
   assert abs(float(camera.timestamp)-frame['timestamp'])<=2e-7
   with np.load(frame['path'],allow_pickle=False) as archive:
    gt=np.ascontiguousarray(archive['rgb'],dtype=np.float32);assert np.array_equal(archive['actor_labels']==1,mask)
   directory=opts.output/f'frames/{raw:03d}';directory.mkdir(parents=True)
   Image.fromarray(common.rgb8(gt)).save(directory/'gt.png')
   with torch.no_grad():native=gaussian_renderer.render(camera,static,dynamic,args,bg,**extras)
   rgb={};alpha={};images={};query_rows={};keep_tensor=None
   for name,deleted in QUERIES.items():
    public=ACTORS if name!='delete_a6_a37' else tuple(reversed(ACTORS))
    with torch.no_grad():package=render_scene(camera,static,dynamic,args,bg,
      dynamic_actor_ids=public,delete_actor_ids=deleted,renderer=gaussian_renderer,**extras)
    assert fingerprints(models,torch)==baseline,'Query changed model parameters or identity'
    if name=='keep':
     delta=float((package['render']-native['render']).abs().max());native_max=max(native_max,delta)
     assert delta<=1e-6,'Empty delete differs from original native KEEP'
     keep_tensor=package['render'].detach().clone()
    if name=='keep_again':
     delta=float((package['render']-keep_tensor).abs().max());again_max=max(again_max,delta)
     assert delta<=1e-6,'Switching target changed KEEP'
    feature=package['feature'];assert feature.shape==(len(ACTORS),320,480)
    contribution={str(a):dict(max=float(feature[i].abs().max()),sum=float(feature[i].double().sum())) for i,a in enumerate(ACTORS)}
    for actor in deleted:assert contribution[str(actor)]['max']==0.,'Target owner still contributes in native CUDA joint render'
    query_rows[name]=dict(delete_actor_ids=list(deleted),parameters_unchanged=True,owner_contribution=contribution,
     hidden_rows=sum(source_counts[str(a)] for a in deleted),native_joint_depth_rerender=True)
    rgb[name]=common.tensor_hwc(package['render']);alpha[name]=common.tensor_alpha(package['alpha'])
    assert np.isfinite(rgb[name]).all() and np.isfinite(alpha[name]).all()
    images[name]=common.rgb8(rgb[name]);Image.fromarray(images[name]).save(directory/f'{name}.png')
    if opts.pilot_reference and raw in PILOT:
     expected=np.asarray(Image.open(opts.pilot_reference/f'frames/{raw:03d}/{name}.png').convert('RGB'))
     maximum=int(np.abs(images[name].astype(np.int16)-expected.astype(np.int16)).max());assert maximum<=1
     replay_max=max(replay_max,maximum)
    del package
   if raw in (130,137) and query_rows['keep']['owner_contribution']['37']['sum']>1e-6:
    target37_witness=True
   values={name:dict(pixels=int(mask.sum()),rgb_error_sum=float(np.abs(np.clip(value[mask],0,1).astype(np.float64)-gt[mask].astype(np.float64)).mean(1).sum()),
      alpha_sum=float(alpha[name][mask].astype(np.float64).sum())) for name,value in rgb.items()}
   arrays=dict(raw=np.int16(raw),yx=np.column_stack(np.nonzero(mask)).astype(np.int16),gt_rgb=gt[mask],
      mask_sha256=np.asarray(frame['mask_sha256']),model_time=np.float64(frame['timestamp']))
   for name in QUERIES:arrays[name+'_rgb']=rgb[name][mask];arrays[name+'_alpha']=alpha[name][mask]
   np.savez_compressed(opts.output/f'a5_float/{raw:03d}.npz',**arrays)
   rows.append(dict(raw=raw,fixed_mask_sha256=frame['mask_sha256'],queries=query_rows,a5=values))
   if raw in PILOT:
    common.contact_sheet(opts.output/f'anchors/{raw:03d}_full.png',[common.rgb8(gt)]+[images[q] for q in QUERIES],['GT']+list(QUERIES))
    common.contact_sheet(opts.output/f'anchors/{raw:03d}_a5_crop.png',[common.rgb8(gt),images['keep'],images['delete_a37']],['GT','KEEP','DELETE37'],crop)
   print(json.dumps(dict(raw=raw,status='PASS_NATIVE_QUERY_FILTER_PARAMETERS_AND_KEEP',target37_keep_sum=query_rows['keep']['owner_contribution']['37']['sum'])),flush=True)
   del native,camera,keep_tensor
  assert target37_witness,'Pilot has no visible owner37 contribution witness; do not claim target disappearance'
  n=sum(row['a5']['keep']['pixels'] for row in rows)
  selected={name:dict(pixels=n,keep_rgb_mae=sum(row['a5'][name]['rgb_error_sum'] for row in rows)/n,
    mean_alpha=sum(row['a5'][name]['alpha_sum'] for row in rows)/n) for name in QUERIES}
  result=dict(status='PASS_NATIVE_IDENTITY_EDIT_PILOT_ENGINEERING' if opts.scope=='pilot' else 'PASS_NATIVE_IDENTITY_EDIT_ALL49_ENGINEERING',
   scope=opts.scope,frame_count=len(rows),raw_ids=[f['raw'] for f in frames],frames=rows,model=str(opts.model),iteration=opts.iteration,
   dynamic_actor_ids=list(ACTORS),queries={k:list(v) for k,v in QUERIES.items()},current_owner_rows=source_counts,
   optimizer_created=False,optimization_updates=0,parameters_unchanged=True,checkpoint_identity_self_contained=True,
   keep_native_max_abs_difference=native_max,keep_again_max_abs_difference=again_max,
   cuda_joint_target_contribution_zero=True,visible_owner37_witness=target37_witness,
   selected_a5=selected,full49_fixed_reference=dict(pixels=121128,sha256=denominator_sha),
   fixed_denominator=dict(pixels=n,selection='original observed actor_labels==1, not rendered alpha'),
   native_engine=str(ENGINE),source_integrity='PENDING',pilot_to_all49_max_uint8_difference=replay_max if opts.pilot_reference else None,
   acceptance=dict(KEEP_BODY='PENDING_IMAGE_REVIEW',TARGET_ONLY_EXACT_FILTER='PASS',TARGET_ONLY_VISUAL='PENDING_IMAGE_REVIEW',SWITCH_NO_TRAIN='PASS'),
   caveat='Owner filtering and parameter invariance do not prove visual disappearance or roof quality. Static residual ghosts, if present, remain unmodified.',
   quality_accepted=False,human_verdict=None)
  if opts.scope=='all49':result['all49']=selected
 except BaseException:error=traceback.format_exc()
 finally:
  after=common.inventory(paths);changed=[p for p in before if before[p]!=after.get(p)]
  write(opts.output/'source_integrity.json',dict(status='PASS_UNCHANGED' if not changed else 'FAIL_SOURCE_CHANGED',changed=changed,checked_files=len(before),before=before,after=after))
 if error or changed:
  write(opts.output/'failure.json',dict(traceback=error,source_changed=changed));raise RuntimeError(error or str(changed))
 result['source_integrity']='PASS_UNCHANGED_BYTES_MTIME_SHA256';result['seconds']=time.monotonic()-started
 write(opts.output/'summary.json',result)
 print(json.dumps(dict(status=result['status'],frame_count=result['frame_count'],keep_native_max_abs=result['keep_native_max_abs_difference'])),flush=True)
if __name__=='__main__':main()

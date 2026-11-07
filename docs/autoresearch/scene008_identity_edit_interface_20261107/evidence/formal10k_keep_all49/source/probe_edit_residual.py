"""同一 checkpoint 零更新归因：D37/D6+37/全部动态删除/原静态输出。"""
from pathlib import Path
import argparse,importlib.util,json,sys,time,traceback
import numpy as np
from PIL import Image

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full_dynamic_edit_20261107_r1')
ENGINE=ROOT.parent/'r15_baseline_owner5_20261018/training_source/scene0008_089_137'
API=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82')
ACTORS=(5,6,18,26,37);PILOT=(105,130,137)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--iteration',type=int,required=True);ap.add_argument('--output',type=Path,required=True)
 opts=ap.parse_args();assert opts.model.resolve().is_relative_to(ROOT) and opts.output.resolve().is_relative_to(ROOT) and not opts.output.exists()
 common=load('residual_native_common',ROOT/'source/native_evaluation_common.py')
 evaluation=load('residual_query_helpers',ROOT/'source/evaluate_scene.py')
 opts.engine=ENGINE;opts.observations=ROOT.parent/'scene008_instances_20261105_r1/observations/frames'
 opts.expected_mask_pixels=121128;opts.crop=[0,160,175,320];opts.dynamic_aliases=list(ACTORS)
 paths=[Path(__file__),ROOT/'source/evaluate_scene.py',ROOT/'source/native_evaluation_common.py',*common.source_model_files(opts.model,opts.iteration)]
 paths+=sorted(ENGINE.rglob('*.py'))+sorted((API/'drivingeditor_dynamic_edit').glob('*.py'))
 paths+=[opts.observations/f'{r:03d}.npz' for r in range(89,138)]
 opts.output.mkdir();before=common.inventory(paths);write(opts.output/'source_manifest_before.json',dict(files=before))
 started=time.monotonic();error=None;result=None
 try:
  frames,_,crop=common.observation_headers(opts,common.read_json(opts.model/'cameras.json'),common.read_json(opts.model/'actual_scene_binding.json'))
  frames=[f for f in frames if f['raw'] in PILOT]
  sys.path[:0]=[str(ENGINE),str(API)]
  import torch,gaussian_renderer
  from scene.cameras import Camera
  from drivingeditor_dynamic_edit import install_identity_serialization,actor_ids,render_scene
  install_identity_serialization();torch.set_num_threads(1)
  args,static,dynamic,env=common.restore_models(opts,common.read_json(opts.model/'effective_config.json'))
  models=dict(static=static,dynamic=dynamic,environment=env);fingerprint=evaluation.fingerprints(models,torch)
  owners=actor_ids(dynamic);bg=torch.zeros(3,device='cuda')
  extras=dict(env_map=env,other=[torch.zeros((len(static.get_xyz),5),device='cuda')],
   dynamic_other=[torch.stack([owners==a for a in ACTORS],1).float()],is_training=False)
  queries={'keep':(),'delete_a37':(37,),'delete_a6_a37':(6,37),'delete_all_dynamic':ACTORS}
  rows=[]
  for frame in frames:
   raw,K,W=[frame[k] for k in ['raw','K','W']]
   camera=Camera(raw,W[:3,:3].T.copy(),W[:3,3].copy(),cx=K[0,2],cy=K[1,2],fx=K[0,0],fy=K[1,1],
    image=torch.zeros((3,320,480)),image_name=f'0008{raw:03d}',resolution=(480,320),timestamp=frame['timestamp'])
   arrays={};records={};images={};directory=opts.output/f'frames/{raw:03d}';directory.mkdir(parents=True)
   with np.load(frame['path'],allow_pickle=False) as d:gt=np.ascontiguousarray(d['rgb'],dtype=np.float32)
   Image.fromarray(common.rgb8(gt)).save(directory/'gt.png')
   for name,deleted in queries.items():
    with torch.no_grad():package=render_scene(camera,static,dynamic,args,bg,dynamic_actor_ids=ACTORS,delete_actor_ids=deleted,renderer=gaussian_renderer,**extras)
    assert evaluation.fingerprints(models,torch)==fingerprint
    rgb=common.tensor_hwc(package['render']);feature=package['feature'].detach().cpu().numpy();alpha=common.tensor_alpha(package['alpha'])
    arrays[name+'_rgb']=rgb;arrays[name+'_owner_alpha']=feature;arrays[name+'_alpha']=alpha
    images[name]=common.rgb8(rgb);Image.fromarray(images[name]).save(directory/f'{name}.png')
    records[name]=dict(delete_actor_ids=list(deleted),owner_contributions={str(a):dict(max=float(np.abs(feature[i]).max()),sum=float(feature[i].astype(np.float64).sum())) for i,a in enumerate(ACTORS)})
    for actor in deleted:assert records[name]['owner_contributions'][str(actor)]['max']==0.
    if name=='delete_a37':
     for index,actor in enumerate(ACTORS):Image.fromarray((np.clip(feature[index],0,1)*255+.5).astype(np.uint8)).save(directory/f'delete_a37_source_owner_{actor}.png')
     # False-color diagnostics are separate files, never used to replace RGB.
     palette=np.asarray([[0,.2,1],[0,1,0],[1,0,1],[0,1,1],[1,0,0]],dtype=np.float32)
     color=np.einsum('ahw,ac->hwc',feature,palette)
     Image.fromarray(common.rgb8(color)).save(directory/'delete_a37_source_owner_colors.png')
    if name=='delete_all_dynamic':
     static_rgb=common.tensor_hwc(package['render_s']);arrays['native_static_rgb']=static_rgb
     images['native_static']=common.rgb8(static_rgb);Image.fromarray(images['native_static']).save(directory/'native_static.png')
     delta=float(np.abs(rgb-static_rgb).max());assert delta<=1e-6
     records[name]['native_static_max_abs_difference']=delta
    del package
   np.savez_compressed(opts.output/f'raw{raw:03d}_native_attribution.npz',**arrays)
   common.contact_sheet(opts.output/f'raw{raw:03d}_comparison.png',[common.rgb8(gt)]+[images[k] for k in ['keep','delete_a37','delete_a6_a37','delete_all_dynamic','native_static']],
    ['GT','KEEP','DELETE37','DELETE6+37','DELETE ALL DYNAMIC','NATIVE STATIC'])
   rows.append(dict(raw=raw,queries=records,
    delete37_vs_delete6_37_mae=float(np.abs(arrays['delete_a37_rgb']-arrays['delete_a6_a37_rgb']).mean()),
    delete6_37_vs_all_dynamic_mae=float(np.abs(arrays['delete_a6_a37_rgb']-arrays['delete_all_dynamic_rgb']).mean())))
   print(json.dumps(dict(raw=raw,all_dynamic_equals_static=records['delete_all_dynamic']['native_static_max_abs_difference'],parameters_unchanged=True)),flush=True)
  result=dict(status='PASS_ZERO_UPDATE_NATIVE_RESIDUAL_SOURCE_ATTRIBUTION',model=str(opts.model),iteration=opts.iteration,
   raw_ids=list(PILOT),frame_count=3,optimization_updates=0,parameters_unchanged=True,source_integrity='PENDING',frames=rows,
   owner_color_legend={'5':'blue','6':'green','18':'magenta','26':'cyan','37':'red'},
   interpretation='source-owner feature maps identify native Gaussian contributions, not independent physical identity truth. ALL DYNAMIC == native render_s locates any remaining appearance in static/environment. No point reassignment or image repair.',
   quality_accepted=False,human_verdict=None)
 except BaseException:error=traceback.format_exc()
 finally:
  after=common.inventory(paths);changed=[p for p in before if before[p]!=after.get(p)]
  write(opts.output/'source_integrity.json',dict(status='PASS_UNCHANGED' if not changed else 'FAIL',changed=changed,before=before,after=after))
 if error or changed:
  write(opts.output/'failure.json',dict(traceback=error,changed=changed));raise RuntimeError(error or str(changed))
 result['source_integrity']='PASS_UNCHANGED_BYTES_MTIME_SHA256';result['seconds']=time.monotonic()-started
 write(opts.output/'summary.json',result)
if __name__=='__main__':main()

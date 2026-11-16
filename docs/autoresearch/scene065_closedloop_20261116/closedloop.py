"""Persistent road / native static-opacity handoff. No image-space final paste."""
from pathlib import Path
import argparse, hashlib, json, os, random, sys, time, types
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import numpy as np
from PIL import Image

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene065_closedloop_20261116_r1')
OLD=C.parent/'scene065_material_loop_20261114_r1'
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
KEYS=[0,60,86,110,150,179,198]
read=lambda p:json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.writing');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''): h.update(b)
 return h.hexdigest()
def differences(a,b,path=''):
 import torch
 if torch.is_tensor(a):return [] if torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu()) else [path]
 if isinstance(a,np.ndarray):return [] if isinstance(b,np.ndarray) and np.array_equal(a,b) else [path]
 if isinstance(a,dict):
  if not isinstance(b,dict) or a.keys()!=b.keys():return [path]
  return [p for k in a for p in differences(a[k],b[k],path+'/'+str(k))]
 if isinstance(a,(tuple,list)):
  if type(a)!=type(b) or len(a)!=len(b):return [path]
  return [p for i,(x,y) in enumerate(zip(a,b)) for p in differences(x,y,path+'/'+str(i))]
 return [] if a==b else [path]
def binding_for(manifest,road):
 return dict(manifest=sha(manifest),candidates=sha(C/'candidates.npz'),road=sha(road),runner=sha(__file__),domains=[sha(C/'domains'/f'{key:03d}.npz') for key in range(199)])
def tensor_image(p,size): return np.asarray(Image.open(p).convert('RGB').resize(size,Image.Resampling.BILINEAR),np.float32)/255
def save_rgb(p,v):
 v=v.detach().cpu().numpy() if hasattr(v,'detach') else v
 if v.shape[0]==3:v=v.transpose(1,2,0)
 Image.fromarray(np.uint8(np.clip(v,0,1)*255+.5)).save(p)
def active_manifest():
 if (C/'active_scene.json').exists():return Path(read(C/'active_scene.json')['manifest'])
 p=C/'manifests/road_visible_guarded.json'
 if p.exists():return p
 p=C/'manifests/road_visible_safe.json'
 return p if p.exists() else C/'manifests/road_before_handoff.json'

def load_scene(manifest,metric=True):
 sys.path.insert(0,str(R/'scripts/worldsim_v82'))
 import torch
 from drivingeditor_dynamic_edit.v44_factory import from_manifest
 from drivingeditor_dynamic_edit.v44_scene import NativeOps
 scene,m=from_manifest(manifest)
 if metric:
  # Uniform similarity of ALL raster rows, native center-frustum set retained.
  scale=1/float(np.load(m.road)['meter_to_model']);ops=scene._bindings();native_raster=ops.raster
  def raster(cam,pipe,degree,rows,bg):
   view=cam.world_view_transform.clone();view[3,:3]*=scale
   projection=cam.world_view_transform.inverse()@cam.full_proj_transform
   camera=types.SimpleNamespace(image_height=cam.image_height,image_width=cam.image_width,FoVx=cam.FoVx,FoVy=cam.FoVy,
    world_view_transform=view,full_proj_transform=view@projection,camera_center=cam.camera_center*scale)
   v=dict(rows,means3D=rows['means3D']*scale)
   if rows['scales'] is not None:v['scales']=rows['scales']*scale
   if rows['cov3D_precomp'] is not None:v['cov3D_precomp']=rows['cov3D_precomp']*scale**2
   return native_raster(camera,pipe,degree,v,bg)
  scene._ops=NativeOps(ops.prepare_rasterization,ops.eval_sh,ops.merge_rows,raster)
  old_rows=scene._rows;ns=len(scene.static.get_xyz);nd=len(scene.dynamic.get_xyz) if scene._include_dynamic else 0
  def rows(key,focus,diag,branch):
   v,owners,cam,cov=old_rows(key,focus,diag,branch);mask=v['mask'].clone()
   mask[:ns+nd]&=v['features'][:ns+nd,0]>.2
   return dict(v,mask=mask),owners,cam,cov
  scene._rows=rows
  scene._physical_raster_scale=scale
 # A fixed road-row fallback retains legacy RGB/near qualification at measured
 # foreground conflicts. It is a persistent 3D point policy, not a pixel crop.
 with np.load(m.road) as asset:
  if 'minimum_model_depth' in asset:
   from dataclasses import replace
   minimum=torch.as_tensor(asset['minimum_model_depth'][asset['active']],device='cuda',dtype=torch.float32)
   providers=[]
   for p in scene._providers:
    if p.kind!='road':providers.append(p);continue
    fn=p.rows
    def guarded_rows(key,cam,pipe,fn=fn,minimum=minimum):
     v=fn(key,cam,pipe)
     return dict(v,mask=v['mask']&(v['features'][:,0]>minimum))
    providers.append(replace(p,rows=guarded_rows))
   scene._providers=tuple(providers);scene._watched=scene._watched+(minimum,);scene._versions=scene._snapshot()
 return scene,m

def render_parts(scene,key,gate_ids=None,gate=None):
 import torch
 values,owners,cam,cov=scene._rows(key,(10,),(10,8,21,20,24),True)
 values={k:v.detach() if hasattr(v,'detach') else v for k,v in values.items()}
 if gate_ids is not None:
  values=dict(values,opacities=values['opacities'].index_copy(0,gate_ids,values['opacities'][gate_ids]*gate[:,None]))
 values=dict(values,mask=values['mask']&~scene._contains(owners,(10,)))
 pack=scene._bindings().raster(cam,scene.pipe,0,values,scene._background)
 env=scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
 pack['rgb']=pack['rgb']+(1-pack['alpha'])*env
 return pack,values,cam

def contributions(scene,key,gate_ids=None,gate=None):
 import torch
 pack,v,cam=render_parts(scene,key,gate_ids,gate)
 n=len(scene.static.get_xyz);nd=len(scene.dynamic.get_xyz) if scene._include_dynamic else 0
 index=torch.arange(len(v['means3D']),device='cuda')
 groups=[index<n,(index>=n)&(index<n+nd),index>=n+nd]
 f=v['features'].new_zeros(v['features'].shape);f[:,:2]=v['features'][:,:2]
 for j,flag in enumerate(groups):f[:,2+3*j:5+3*j]=v['colors_precomp']*flag[:,None]
 c=scene._bindings().raster(cam,scene.pipe,0,dict(v,features=f),scene._background)
 parts=[c['extra'][i*3:i*3+3] for i in range(3)]
 env=(1-pack['alpha'])*scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
 error=float((sum(parts)+env-pack['rgb']).abs().max());assert error<1e-5,error
 return pack,dict(static=parts[0],dynamic=parts[1],road=parts[2],environment=env),error

def preflight():
 import torch
 cfg=read(OLD/'config.json');scene,m=load_scene(OLD/'manifests/corrected_delete.json',metric=False)
 dest=C/'preflight';dest.mkdir(parents=True,exist_ok=True);rows=[]
 for key in KEYS:
  tick=time.time()
  with torch.no_grad():out,parts,error=contributions(scene,key)
  d=dest/f'{key:03d}';d.mkdir(exist_ok=True)
  for name,v in dict(old_delete=out['rgb'],**parts).items():save_rgb(d/(name+'.png'),v)
  np.savez_compressed(d/'old_arrays.npz',rgb=out['rgb'].detach().cpu().numpy(),road_alpha=out['extra'][3:4].detach().cpu().numpy(),static_rgb=parts['static'].detach().cpu().numpy())
  rows.append(dict(raw=key,seconds=time.time()-tick,conservation=error))
 scene.assert_persistent_unchanged()
 write(dest/'completed.json',dict(rows=rows,source_manifest=str(OLD/'manifests/corrected_delete.json'),GPU='keep_on'))
 print(json.dumps(rows),flush=True)

def get_domains(scene,m,asset):
 """BUILD masks use world road projection, accepted identity and foreground guards."""
 import torch
 cfg=read(OLD/'config.json');views=Path(cfg['road_views']);d=C/'domains';d.mkdir(exist_ok=True)
 with np.load(C/'field/road_provenance.npz') as z:changed=z['new_completion']|z['legacy_completion']
 rows=[]
 for frame in m.frames:
  key=frame.key;cam=scene.camera_provider(key);size=(cam.image_width,cam.image_height)
  labels=np.asarray(Image.open(m.data['frames'][key]['instance_mask']).resize(size,Image.Resampling.NEAREST));target=labels==10;K=(labels>0)&~target
  with np.load(views/f'{key:03d}.npz') as z:
   ix=z['index'];inside=z['inside'];guard=z['retained_reveal_guard']|z['retained_box_guard'];observed=z['observed_road']
   fit=inside&changed[ix]&~guard&~K
   # World chart / projected road intersection, never dilated pixel->3D retirement.
   fit&=target|z['target']|z['real_domain']|z['inferred_domain']
  protect=K|(~fit&~target)
  np.savez_compressed(d/f'{key:03d}.npz',fit=fit,target=target,K=K,protect=protect,inside=inside,observed=observed)
  rows.append(dict(raw=key,fit=int(fit.sum()),target=int(target.sum()),K=int(K.sum())))
 write(d/'summary.json',dict(frames=rows,fit_uses_generated_world_road=True,not_independent_GT=True))
 return rows

def select_candidates(scene,m):
 import torch
 n=len(scene.static.get_xyz);device=scene.static.get_xyz.device
 score=torch.zeros(n,device=device);guard=torch.zeros_like(score);count=torch.zeros_like(score,dtype=torch.int32)
 # Target association first requires membership in the measured trajectory's swept OBB.
 transform=np.loadtxt(m.data['world_to_model']);inv=np.linalg.inv(transform)
 xyz=scene.static.get_xyz.detach().cpu().numpy();world=xyz@inv[:3,:3].T+inv[:3,3]
 swept=np.zeros(n,bool)
 track=next(a['track_id'] for a in m.data['actors'] if a['actor_id']==10)
 for frame in m.data['frames']:
  obs=read(frame['observation'])
  for box in obs['boxes']:
   if box['track']!=track:continue
   pose=np.asarray(box['world_from_actor']).reshape(4,4);p=np.linalg.inv(pose);local=world@p[:3,:3].T+p[:3,3]
   # .25m bound matches trajectory/measurement tolerance, independent of 2D H/W.
   swept|=(abs(local)<=np.asarray(box['dimensions'])/2+.25).all(1)
 candidate=torch.as_tensor(swept,device=device)&(scene._owners_s<=0)
 for key in range(199):
  with np.load(C/'domains'/f'{key:03d}.npz') as z:fit=z['fit'];target=z['target'];K=z['K']
  if not target.any() and not K.any():continue
  with torch.no_grad():v,owners,cam,_=scene._rows(key,(),(),False)
  f=v['features'].detach().clone();f[:,8:10]=0;f[:n,8:10]=1;f.requires_grad_(True)
  v=dict(v,mask=v['mask']&~scene._contains(owners,(10,)),features=f)
  out=scene._bindings().raster(cam,scene.pipe,0,v,scene._background)
  t=torch.as_tensor(target&fit,device=device);k=torch.as_tensor(K,device=device)
  obj=out['extra'][6][t].sum()+out['extra'][7][k].sum()
  grad=torch.autograd.grad(obj,f)[0][:n,8:10]
  score=torch.maximum(score,grad[:,0]);guard=torch.maximum(guard,grad[:,1]);count+=(grad[:,0]>1e-4).int()
  del out,v,f,grad,obj
  if key%25==0:print('SELECT',key,flush=True)
 eligible=candidate&(score>1e-4)&(count>=2)&~(guard>score*.02)
 ids=torch.where(eligible)[0]
 np.savez_compressed(C/'candidates.npz',ids=ids.cpu().numpy(),score=score[ids].cpu().numpy(),guard=guard[ids].cpu().numpy(),target_views=count[ids].cpu().numpy())
 write(C/'candidate_receipt.json',dict(candidates=len(ids),swept_candidates=int(candidate.sum()),criterion='unknown static; measured swept OBB + >=2 target image contributions + low retained contribution',two_dimensional_dilation_used_for_selection=False,source_owners_unchanged=True))
 return ids

def optimization(mode,steps=600,resume=None):
 import torch
 torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);random.seed(0)
 manifest=active_manifest();scene,m=load_scene(manifest)
 for model in [scene.static,scene.dynamic,scene.environment]:
  for p in model.parameters() if hasattr(model,'parameters') else []:p.requires_grad_(False)
 asset=np.load(m.road)
 if not (C/'domains/summary.json').exists():get_domains(scene,m,asset)
 ids=select_candidates(scene,m) if not (C/'candidates.npz').exists() else torch.as_tensor(np.load(C/'candidates.npz')['ids'],device='cuda')
 if not len(ids):raise RuntimeError('No verified target-associated residual candidates')
 destination=C/mode;destination.mkdir(exist_ok=True);g=torch.nn.Parameter(torch.full((len(ids),),6.,device='cuda'))
 opt=torch.optim.Adam([g],lr=.04,eps=1e-15);sampler=random.Random(0);remaining=[];start=0
 binding=binding_for(manifest,m.road)
 def state(step):return dict(step=step,gate=g.detach().clone(),optimizer=opt.state_dict(),sampler=sampler.getstate(),remaining=remaining.copy(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),binding=binding,contributors=ids.cpu())
 if resume:
  z=torch.load(resume,map_location='cpu',weights_only=False);assert z['binding']==binding
  g.data.copy_(z['gate']);opt.load_state_dict(z['optimizer']);sampler.setstate(z['sampler']);remaining=z['remaining'];start=z['step']
  torch.set_rng_state(z['torch_rng']);torch.cuda.set_rng_state_all(z['cuda_rng']);np.random.set_state(z['numpy_rng']);random.setstate(z['python_rng'])
  diff=differences(z,state(start));assert not diff,diff
  write(destination/'restore.json',dict(full_state_restored=True,step=start,differences=diff))
 else:torch.save(state(0),destination/'checkpoint000000.pt')
 records=[];begin=time.time();roadrows=next(p for p in scene._providers if p.kind=='road')
 for step in range(start+1,steps+1):
  if not remaining:remaining=list(range(199));sampler.shuffle(remaining)
  key=remaining.pop();cam=scene.camera_provider(key)
  with np.load(C/'domains'/f'{key:03d}.npz') as z:fit=torch.as_tensor(z['fit'],device='cuda');K=torch.as_tensor(z['K'],device='cuda');preserve=torch.as_tensor(z['protect'],device='cuda')
  with torch.no_grad():
   base,_,_=render_parts(scene,key)
   rows=roadrows.rows(key,cam,scene.pipe);road=scene._bindings().raster(cam,scene.pipe,0,rows,scene._background)
   reference=road['rgb']/road['alpha'].clamp_min(.01);valid=fit&(road['alpha'][0]>.9)
  opt.zero_grad(set_to_none=True);out,v,cam=render_parts(scene,key,ids,g.sigmoid())
  photo=(out['rgb']-reference).abs()[:,valid].mean() if valid.any() else g.sum()*0
  keep=(out['rgb']-base['rgb']).abs()[:,preserve].mean() if preserve.any() else g.sum()*0
  with np.load(OLD/'rendered/corrected_delete/frames'/f'{key:06d}/arrays.npz') as frozen:
   original=torch.as_tensor(frozen['rgb'].copy(),device='cuda')
  vehicle=(out['rgb']-original).abs()[:,K].mean() if K.any() else g.sum()*0
  # Explicitly surface-driven: reduce competing static occlusion only on road-supported target footprint.
  # No pixel mask retires rows, and no unsupported road gets zero-opacity supervision.
  loss=photo+4*keep+20*vehicle+.0001*(1-g.sigmoid()).mean()
  if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
  loss.backward()
  if not torch.isfinite(g.grad).all():raise RuntimeError('nonfinite gradient')
  opt.step();record=dict(step=step,raw=key,loss=float(loss),road=float(photo),preserve=float(keep),vehicle=float(vehicle),gate_mean=float(g.sigmoid().mean()),seconds=time.time()-begin);records.append(record)
  if step%25==0 or step==start+1:
   print(json.dumps(record),flush=True);write(destination/'progress.json',record)
  if step%100==0 or step==steps:torch.save(state(step),destination/f'checkpoint{step:06d}.pt')
  if mode=='smoke' and time.time()-begin>=600:break
 step=records[-1]['step'];torch.save(state(step),destination/f'checkpoint{step:06d}.pt')
 write(destination/'training.json',records)
 scene.assert_persistent_unchanged();write(destination/'completed.json',dict(completed=True,steps=step,updates=step-start,seconds=time.time()-begin,trainable='local persistent static opacity residual only',road_frozen=True,raw_models_unchanged=True,checkpoint=str(destination/f'checkpoint{step:06d}.pt')))

def render(mode):
 import torch
 manifest=active_manifest();scene,m=load_scene(manifest)
 formal=read(C/'formal/completed.json') if (C/'formal/completed.json').exists() else None
 pilot=read(C/'smoke_guarded/completed.json') if not formal and (C/'smoke_guarded/completed.json').exists() else None
 state= torch.load((formal or pilot)['checkpoint'],map_location='cpu',weights_only=False) if formal or pilot else None
 if state:
  assert state['binding']==binding_for(manifest,m.road)
  assert np.array_equal(state['contributors'].numpy(),np.load(C/'candidates.npz')['ids'])
 ids=state['contributors'].to('cuda') if state else None
 gate=state['gate'].to('cuda').sigmoid() if state else None
 frames=KEYS if mode!='full' else list(range(199));root=C/('pilot_formal' if formal else 'pilot_smoke_guarded' if pilot else 'pilot_guarded_final_before' if (C/'active_scene.json').exists() else 'pilot_guarded_before' if manifest.name=='road_visible_guarded.json' else 'pilot_safe_before' if manifest.name=='road_visible_safe.json' else 'pilot_before') if mode!='full' else C/'rendered'
 rows=[];roadprovider=next(p for p in scene._providers if p.kind=='road')
 with torch.no_grad():
  for key in frames:
   tick=time.time();d=root/f'{key:03d}';d.mkdir(parents=True,exist_ok=True)
   pack,parts,error=contributions(scene,key,ids,gate)
   cam=scene.camera_provider(key);rv=roadprovider.rows(key,cam,scene.pipe);road=scene._bindings().raster(cam,scene.pipe,0,rv,scene._background)
   for name,v in dict(new_delete=pack['rgb'],road_only=road['rgb'],static_contribution=parts['static'],road_contribution=parts['road'],dynamic_contribution=parts['dynamic']).items():save_rgb(d/(name+'.png'),v)
   np.savez_compressed(d/'arrays.npz',rgb=pack['rgb'].cpu().numpy(),road_alpha=pack['extra'][3:4].cpu().numpy(),road_only_alpha=road['alpha'].cpu().numpy(),static_rgb=parts['static'].cpu().numpy(),road_rgb=parts['road'].cpu().numpy(),alpha=pack['alpha'].cpu().numpy())
   rows.append(dict(raw=key,seconds=time.time()-tick,conservation=error,finite=bool(torch.isfinite(pack['rgb']).all())))
   if key%20==0:print('RENDER',key,flush=True);write(root/'progress.json',dict(raw=key,frames=len(rows)))
 scene.assert_persistent_unchanged();write(root/'completed.json',dict(completed=True,frames=len(frames),keys=frames,rows=rows,road=str(m.road),checkpoint=(formal or pilot),GPU='keep_on',image_space_final_compositing=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','smoke','formal','pilot','full']);p.add_argument('--steps',type=int,default=600);p.add_argument('--resume');a=p.parse_args()
 if a.command=='preflight':preflight()
 elif a.command in ['smoke','formal']:optimization(a.command,a.steps,a.resume)
 else:render(a.command)

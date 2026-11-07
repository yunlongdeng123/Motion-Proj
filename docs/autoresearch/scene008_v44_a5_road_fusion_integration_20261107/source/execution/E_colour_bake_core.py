"""Bounded paired colour-only bake with exact checkpoint restoration."""
from pathlib import Path
import argparse,copy,hashlib,json,os,random,sys,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import torch
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_v44_a5_road_fusion_integration_20261107_r1';S=B/'scene008_v44_full5_integration_20261107_r1'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');QUAL=R/'E4_appearance_qualification/qualification_manifest.json';CONTRACT=R/'contract/existing_surface_fusion_bake_pilot_v1.json'
RAWS=[94,105,128,137];SEED=20261107
sys.path[:0]=[str(REPO),str(B/'scene008_v44_rgb_asset_20261107_r1/source')]
from v44_runtime import V44Runtime
from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
from scripts.worldsim_v82.drivingeditor_dynamic_edit.baseline_road_appearance import baseline_appearance_binding,write_appearance_material,BaselineRoadAppearanceV44Integration
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def cpu(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,np.ndarray):return x.copy()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu(v) for v in x)
 return copy.deepcopy(x)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().clone(),'torch_cuda':[x.clone() for x in torch.cuda.get_rng_state_all()]}
def set_rng(x):random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch_cpu']);torch.cuda.set_rng_state_all(x['torch_cuda'])
def exact(a,b,path='state'):
 if torch.is_tensor(a):assert torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu()),path
 elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b),path
 elif isinstance(a,dict):
  assert set(a)==set(b),path
  for k in a:exact(a[k],b[k],path+'.'+str(k))
 elif isinstance(a,(list,tuple)):
  assert type(a)==type(b) and len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+f'[{i}]')
 else:assert a==b,(path,a,b)
class Sampler:
 def __init__(self):self.r=random.Random(SEED);self.order=[];self.pos=0;self.epoch=-1
 def next(self):
  if self.pos==len(self.order):self.order=RAWS.copy();self.r.shuffle(self.order);self.pos=0;self.epoch+=1
  raw=self.order[self.pos];self.pos+=1;return raw
 def state(self):return {'rng':self.r.getstate(),'order':self.order.copy(),'pos':self.pos,'epoch':self.epoch}
 def load(self,s):self.r.setstate(s['rng']);self.order=s['order'].copy();self.pos=s['pos'];self.epoch=s['epoch']
class Core:
 def __init__(self,out):
  self.out=out;out.mkdir(exist_ok=False);self.start=time.time();self.calls=0;self.backs=0;self.updates=0;self.times=[];self.history=[]
  assert sha(CONTRACT)=='62be3c0ddde3ec5421954a1e3c28d4813c9d713def0221204510d33d4c2ed6f1'
  assert sha(REPO/'scripts/worldsim_v82/drivingeditor_dynamic_edit/baseline_road_appearance.py')=='415bd33afac61e7d28e6d195051db0e33ab3f793f36d5e31208a2e89f936fa9f'
  assert sha(QUAL)=='48283b8f375d504553e827bdc3ba2bd846414f6e807c8c0a22520923a7686dae'
  torch.set_num_threads(1);assert torch.cuda.is_available();random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
  self.base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json')
  self.semantic=SemanticActor5V44Integration(self.base,semantic_sidecar_path=S/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=R/'contract/full5_transform_matrix.txt')
  self.binding=baseline_appearance_binding(self.semantic)
  q=json.loads(QUAL.read_text())
  with np.load(QUAL.parent/q['selection_file'],allow_pickle=False) as z:self.sid=torch.as_tensor(z['static_safe_row_ids'],device='cuda');self.rid=torch.as_tensor(z['road_safe_row_ids'],device='cuda')
  assert len(self.sid)==15262 and len(self.rid)==19114
  self.delta=torch.nn.Parameter(torch.zeros((34376,3),dtype=torch.float32,device='cuda'));self.opt=torch.optim.Adam([self.delta],lr=.01,betas=(.9,.999),eps=1e-8)
  self.sampler=Sampler();self.step=0;self.targets={};self.target_meta={};self.indices={}
  targetroot=B/'scene008_road_fusion_bake_20261107_r1/targets_edgeblend_v1'
  for raw in RAWS:
   origin='projected_BUILD40' if raw in [94,128] else 'projected_HELD9';p=targetroot/origin/f'raw{raw:03d}.npz'
   with np.load(p,allow_pickle=False) as z:rgb=z['edgeblend_rgb'].copy();mask=z['fit_mask'].astype(bool)
   assert rgb.shape==(320,480,3) and np.isfinite(rgb).all() and mask.shape==(320,480) and mask.any()
   self.targets[raw]=(torch.as_tensor(rgb.transpose(2,0,1),device='cuda'),torch.as_tensor(mask,device='cuda'))
   self.target_meta[str(raw)]={'path':str(p),'sha256':sha(p),'fit_pixels':int(mask.sum()),'original_role':origin,'current_role':'BUILD'}
  self.config={'fit_contract_sha256':sha(CONTRACT),'qualification_sha256':sha(QUAL),'targets':self.target_meta,'parameter_bounds':[-1.,1.],'sampler_seed':SEED,'model_binding':self.binding}
  write(out/'executed_config.json',self.config)
 def render_train(self,raw,state):
  scene=self.semantic.scene(raw,delete=state=='delete');kw={k:v.detach() if torch.is_tensor(v) else v for k,v in scene['rows'].items()}
  key=(raw,state)
  if key not in self.indices:
   a=torch.where(kw['features'][:,5]==1)[0];b=torch.where(kw['features'][:,4]==1)[0];assert len(a)==1042776 and len(b)==143248
   self.indices[key]=torch.cat([a[self.sid],b[self.rid]])
  ids=self.indices[key];base=kw['colors_precomp'];assert base is not None and kw['shs'] is None
  kw['colors_precomp']=base.index_copy(0,ids,(base[ids]+self.delta).clamp_min(0))
  assert all(not v.requires_grad for k,v in kw.items() if torch.is_tensor(v) and k!='colors_precomp')
  out=self.base.raster(scene['cam'],self.base.pipe,0,kw,scene['bg']);self.calls+=1
  write(self.out/'kernel_progress.json',{'paired_step':self.step,'actual_native_forwards':self.calls,'actual_native_backwards':self.backs,'actual_optimizer_updates':self.updates})
  return out['rgb']+(1-out['alpha'])*scene['env'].detach()
 def paired(self):
  start=time.time();raw=self.sampler.next();target,mask=self.targets[raw];self.opt.zero_grad(set_to_none=True);losses=[]
  for state in ['keep','delete']:
   rgb=self.render_train(raw,state);loss=.5*(rgb[:,mask]-target[:,mask]).square().mean();assert torch.isfinite(loss)
   losses.append(float(loss.detach())*2);loss.backward();self.backs+=1
   write(self.out/'kernel_progress.json',{'paired_step':self.step,'actual_native_forwards':self.calls,'actual_native_backwards':self.backs,'actual_optimizer_updates':self.updates})
   del loss,rgb
  assert self.delta.grad is not None and torch.isfinite(self.delta.grad).all();self.opt.step()
  with torch.no_grad():self.delta.clamp_(-1,1)
  self.opt.zero_grad(set_to_none=True);self.step+=1;self.updates+=1;self.times.append(time.time()-start)
  self.history.append({'step':self.step,'raw':raw,'keep_MSE':losses[0],'delete_MSE':losses[1]})
  write(self.out/'progress.json',{'step':self.step,'last':self.history[-1],'native_forwards':self.calls,'native_backwards':self.backs,'optimizer_updates':self.updates,'mean_pair_seconds':sum(self.times)/len(self.times)})
  print(json.dumps(self.history[-1]),flush=True)
 def state(self):
  assert self.delta.grad is None and float(self.delta.abs().max())<=1.
  return cpu({'schema':'E_fixed_colour_bake_full_state_v1','step':self.step,'delta':self.delta,'optimizer':self.opt.state_dict(),'sampler':self.sampler.state(),'rng':rng(),'history':self.history,'config':self.config})
 def save(self,name):
  path=self.out/name;assert not path.exists();tmp=path.with_suffix('.tmp');torch.save(self.state(),tmp);tmp.rename(path);return path
 def restore(self,path):
  saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['config']==self.config
  with torch.no_grad():self.delta.copy_(saved['delta'].cuda())
  self.opt.load_state_dict(saved['optimizer']);self.sampler.load(saved['sampler']);self.step=saved['step'];self.history=copy.deepcopy(saved['history']);set_rng(saved['rng'])
  exact(saved,self.state());write(self.out/'restore_audit.json',{'status':'PASS_FULL_STATE_START_EXACT','checkpoint':str(path),'checkpoint_sha256':sha(path),'step':self.step,'delta_Adam_sampler_all_RNG_exact':True})
 def material(self,name):
  assert float(self.delta.abs().max())<=1.;a=self.delta.detach().cpu().numpy().copy()
  path=write_appearance_material(QUAL,{'native_static':a[:15262],'oldroad':a[15262:]},self.out/name)
  metadata=json.loads(path.read_text());metadata.update(fit_contract_sha256=sha(CONTRACT),parameter_bounds=[-1.,1.],paired_optimizer_steps=self.step)
  write(path,metadata);return path
 def evaluate(self,name):
  before=rng();material=self.material(name+'_material');api=BaselineRoadAppearanceV44Integration(self.semantic,QUAL,material)
  d=self.out/name;d.mkdir();(d/'frames').mkdir();rows=[]
  from PIL import Image
  for raw in RAWS:
   f=d/f'frames/{raw:03d}';f.mkdir()
   for state in ['keep','delete']:
    scene=api.scene(raw,delete=state=='delete');out=api.render(raw,scene);self.calls+=1
    rgb=out['rgb'].detach().cpu().numpy();alpha=out['alpha'].detach().cpu().numpy()[0];depthsum=out['depth_sum'].detach().cpu().numpy()[0]
    values={'rgb':rgb,'alpha':alpha,'depth_sum':depthsum,'depth_m':depthsum/np.maximum(alpha,1e-8)/self.base.scale,'group_alpha_T':out['group_alpha_T'].detach().cpu().numpy(),'added_owner_rgb_T':out['added_owner_rgb_T'].detach().cpu().numpy()}
    with np.load(R/f'baseline4/frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:
     for k in ['alpha','depth_sum','depth_m','group_alpha_T']:assert np.array_equal(values[k],z[k]),(raw,state,k)
    np.savez_compressed(f/(state+'_float.npz'),**values);Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(f/(state+'.png'))
    rows.append({'raw':raw,'state':state,'all_global_geometry_weights_exact':True});del out,scene
  api.assert_unchanged(full=True);set_rng(before)
  write(d/'receipt.json',{'status':'NATIVE_PUBLIC_415_MATERIAL_EVALUATION_GEOMETRY_EXACT_PENDING_GATES','iteration':self.step,'actual_native_forwards':8,'optimizer_steps_in_this_evaluation':0,'material_manifest':str(material),'runtime_receipt':api.receipt(),'rows':rows})
  return d
 def frozen(self):
  assert baseline_appearance_binding(self.semantic)==self.binding;self.semantic.assert_unchanged(full=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','resume2','formal'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path);a=p.parse_args()
 c=Core(a.output)
 if a.mode=='smoke':
  train_start=time.time()
  for _ in range(4):c.paired()
  assert sorted(x['raw'] for x in c.history)==sorted(RAWS);eta=400*sum(c.times)/len(c.times)
  if eta>=3600:
   while time.time()-train_start<600:c.paired()
  stop=c.step;saved=c.save(f'state{stop}.pth');c.evaluate('evaluation');exact(torch.load(saved,map_location='cpu',weights_only=False),c.state())
  for _ in range(2):c.paired()
  end=c.save(f'disposable_state{c.step}.pth');c.frozen()
  result={'status':'PASS_SMOKE_SAVE_PUBLIC_EVAL_AND_DISPOSABLE_NEXT2','smoke_updates':stop,'disposable_updates':2,'actual_optimizer_updates':c.updates,'actual_native_forwards':c.calls,'actual_native_backwards':c.backs,
   'checkpoint':str(saved),'disposable_checkpoint':str(end),'formal_ETA_seconds_from_actual_smoke':eta,'long_smoke_required':eta>=3600,'actual_training_smoke_seconds':sum(c.times[:-2]),'formal_not_started':True}
 elif a.mode=='resume2':
  assert a.checkpoint;c.restore(a.checkpoint)
  for _ in range(2):c.paired()
  path=c.save(f'restored_state{c.step}.pth');c.frozen();result={'status':'PASS_EXACT_RESTORE_AND_NEXT2_ENTRY','actual_optimizer_updates':2,'actual_native_forwards':c.calls,'actual_native_backwards':c.backs,'checkpoint':str(path),'final_step':c.step}
 else:
  for _ in range(400):
   c.paired()
   if c.step in [100,200,400]:c.save(f'state{c.step}.pth')
  c.evaluate('evaluation');c.frozen();result={'status':'COMPLETE_FIXED_400_PAIRED_COLOUR_RESIDUAL_BAKE_PENDING_GATES','actual_optimizer_updates':400,'actual_native_forwards':c.calls,'actual_native_backwards':c.backs,'final_step':c.step,'final_checkpoint':str(c.out/'state400.pth'),'evaluation':str(c.out/'evaluation')}
 result.update({'original_Gaussian_SH_geometry_opacity_identity_updates':0,'only_new_colour_residual_updates':c.updates,'parameter_bounds':[-1,1],'delta_max_abs':float(c.delta.abs().max()),'fit_contract_sha256':sha(CONTRACT),'all_original_model_binding_and_files_unchanged':True,'seconds':time.time()-c.start,'max_reserved_cuda_bytes':int(torch.cuda.max_memory_reserved()),'all49_auto_release':False,'shutdown':False,'reset_card':False})
 write(c.out/'returned.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()

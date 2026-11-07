"""Frozen21K plus complete C: qualified shared-parent RGB only."""
from pathlib import Path
import argparse,copy,hashlib,json,os,random,sys,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import torch
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_v44_a5_c49_rgb_20261107_r1';OLD=B/'scene008_v44_a5_road_fusion_integration_20261107_r1';SEM=B/'scene008_v44_full5_integration_20261107_r1';REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');QUAL=R/'parent_visibility49/qualification_manifest.json';CONTRACT=R/'contract/C49_parent_RGB_training_v1.json';SEED=20261107;RAWS=list(range(89,138));PILOT=[94,105,128,137]
sys.path[:0]=[str(R/'source'),str(REPO),str(B/'scene008_v44_rgb_asset_20261107_r1/source')]
from v44_runtime import V44Runtime
from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration,C_MANIFEST
from scripts.worldsim_v82.drivingeditor_dynamic_edit.baseline_road_appearance import baseline_appearance_binding
from road_fusion_material_runtime import RoadFusionMaterialRuntime
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def cpu(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,np.ndarray):return x.copy()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu(v) for v in x)
 return copy.deepcopy(x)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().clone(),'torch_cuda':[x.clone() for x in torch.cuda.get_rng_state_all()]}
def set_rng(s):random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch_cpu']);torch.cuda.set_rng_state_all(s['torch_cuda'])
def exact(a,b,path='state'):
 if torch.is_tensor(a):assert torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu()),path
 elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b),path
 elif isinstance(a,dict):
  assert set(a)==set(b),path
  for k in a:exact(a[k],b[k],path+'.'+str(k))
 elif isinstance(a,(list,tuple)):
  assert type(a)==type(b) and len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):exact(x,y,path+str(i))
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
  self.out=Path(out);self.out.mkdir(exist_ok=False);self.start=time.time();self.calls=0;self.backs=0;self.updates=0;self.step=0;self.history=[];self.times=[];self.sampler=Sampler()
  assert sha(R/'source/road_fusion_material_runtime.py')=='4d8932b39b8b0fd5034c1147068acf7a6b5474c62e556d2ebc79a04e319f1009'
  assert sha(QUAL)=='6df77ae629503373f81b4e36ba67e81332792ad66acfa9bb0518806eed9f9515'
  contract=json.loads(CONTRACT.read_text());assert contract['eligible_parents']==2119 and contract['formal_initial_stop']==2000 and contract['smoke_training_seconds_min']==600
  torch.set_num_threads(1);assert torch.cuda.is_available();random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
  self.base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json')
  self.semantic=SemanticActor5V44Integration(self.base,semantic_sidecar_path=SEM/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=OLD/'contract/full5_transform_matrix.txt')
  self.road=RoadFusionV44Integration(self.semantic,C_MANIFEST);self.runtime=RoadFusionMaterialRuntime(self.road);self.binding=baseline_appearance_binding(self.semantic)
  q=json.loads(QUAL.read_text());selection=QUAL.parent/q['selection_file'];assert sha(selection)==q['selection_sha256']
  with np.load(selection,allow_pickle=False) as z:
   ids=z['eligible_parent_ids'].copy();mass=z['parent_mass'].copy();initial=z['initial_parent_RGB'].copy()
  assert len(ids)==2119 and np.array_equal(ids,np.flatnonzero(mass>0)) and np.array_equal(initial,self.runtime.initial_parent_rgb.cpu().numpy())
  self.ids=torch.as_tensor(ids,device='cuda',dtype=torch.long);self.rgb=torch.nn.Parameter(self.runtime.initial_parent_rgb[self.ids].detach().clone());self.opt=torch.optim.Adam([self.rgb],lr=.01,betas=(.9,.999),eps=1e-8)
  self.targets={};self.target_meta={};guardpaths=[CONTRACT,QUAL,selection,Path(__file__),R/'source/road_fusion_material_runtime.py']
  for item in q['targets']:
   raw=item['raw'];p=Path(item['path']);assert sha(p)==item['sha256'];guardpaths.append(p)
   with np.load(p,allow_pickle=False) as z:target=z['edgeblend_rgb'].copy();mask=z['fit_mask'].astype(bool)
   assert target.shape==(320,480,3) and np.isfinite(target).all() and mask.shape==(320,480) and mask.any()
   self.targets[raw]=(torch.as_tensor(target.transpose(2,0,1),device='cuda'),torch.as_tensor(mask,device='cuda'));self.target_meta[str(raw)]=item
  source_manifest=B/'scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1/target_manifest.json';assert sha(source_manifest)=='d7cca4167e2bbb68347f63babae356320a1b0ddd60e580bf746e966f2ac8acbe';guardpaths.append(source_manifest)
  self.guard={str(p):{'sha256':sha(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in guardpaths}
  self.config={'training_contract_sha256':sha(CONTRACT),'qualification_manifest_sha256':sha(QUAL),'targets':self.target_meta,'source_target_manifest_sha256':sha(source_manifest),'model_binding':self.binding,
   'source_C_asset_sha256':'3afc6e25b7b152f851b4eb0de4a73e6d219c9526d75453d99097a31ddd5774c5','initializer':'existing C400 warm RGB; fresh private Adam and new step0',
   'initial_parent_RGB_sha256':hashlib.sha256(initial.tobytes()).hexdigest(),'eligible_parent_ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),'eligible_parents':2119,'RGB_bounds':[0.,1.],'sampler_seed':SEED,'files':self.guard}
  write(self.out/'executed_config.json',self.config);write(self.out/'source_target_guard_before.json',self.guard)
 def progress(self):write(self.out/'kernel_progress.json',{'cumulative_stage_step':self.step,'actual_native_forwards':self.calls,'actual_native_backwards':self.backs,'actual_process_optimizer_updates':self.updates})
 def table(self):return self.runtime.parent_table(self.ids,self.rgb)
 def render(self,raw,state):
  scene=self.runtime.scene(raw,delete=state=='delete',parent_rgb=self.table())
  assert all(not v.requires_grad for k,v in scene['rows'].items() if torch.is_tensor(v) and k not in ['colors_precomp','features']) and not scene['env'].requires_grad and not scene['bg'].requires_grad
  out=self.runtime.render(raw,scene);self.calls+=1;self.progress();return out
 def paired(self):
  start=time.time();raw=self.sampler.next();target,mask=self.targets[raw];self.opt.zero_grad(set_to_none=True);losses=[]
  for state in ['keep','delete']:
   out=self.render(raw,state);loss=.5*(out['rgb'][:,mask]-target[:,mask]).square().mean();assert torch.isfinite(loss);losses.append(float(loss.detach())*2)
   loss.backward();self.backs+=1;self.progress();del loss,out
  assert self.rgb.grad is not None and torch.isfinite(self.rgb.grad).all();nonzero=int((self.rgb.grad!=0).any(1).sum());self.opt.step()
  with torch.no_grad():self.rgb.clamp_(0,1)
  self.opt.zero_grad(set_to_none=True);self.step+=1;self.updates+=1;self.times.append(time.time()-start)
  row={'step':self.step,'raw':raw,'epoch':self.sampler.epoch,'keep_MSE':losses[0],'delete_MSE':losses[1],'finite_gradient':True,'nonzero_gradient_parents':nonzero,'RGB_min':float(self.rgb.min()),'RGB_max':float(self.rgb.max())};self.history.append(row)
  write(self.out/'progress.json',dict(row,actual_native_forwards=self.calls,actual_native_backwards=self.backs,actual_optimizer_updates=self.updates,training_seconds=sum(self.times)));print(json.dumps(row),flush=True)
  if self.sampler.pos==49:
   rows=self.history[-49:];assert sorted(x['raw'] for x in rows)==RAWS
   epochfile=self.out/'epoch49_losses.jsonl'
   with epochfile.open('a') as f:f.write(json.dumps({'epoch':self.sampler.epoch,'last_step':self.step,'keep_MSE':sum(x['keep_MSE'] for x in rows)/49,'delete_MSE':sum(x['delete_MSE'] for x in rows)/49})+'\n')
 def state(self):
  assert self.rgb.grad is None and torch.isfinite(self.rgb).all() and float(self.rgb.min())>=0 and float(self.rgb.max())<=1
  return cpu({'schema':'C49_PARENT_RGB_FULL_STATE_V1','step':self.step,'parent_RGB':self.rgb,'optimizer':self.opt.state_dict(),'sampler':self.sampler.state(),'rng':rng(),'history':self.history,'config':self.config})
 def save(self,name):
  p=self.out/name;assert not p.exists();temp=p.with_suffix('.tmp');torch.save(self.state(),temp);temp.rename(p);return p
 def restore(self,p):
  s=torch.load(p,map_location='cpu',weights_only=False);assert s['config']==self.config
  with torch.no_grad():self.rgb.copy_(s['parent_RGB'].cuda())
  self.opt.load_state_dict(s['optimizer']);self.sampler.load(s['sampler']);self.step=s['step'];self.history=copy.deepcopy(s['history']);set_rng(s['rng']);exact(s,self.state())
  write(self.out/'restore_audit.json',{'status':'PASS_FULL_PARAMETER_ADAM_PRIVATE_SAMPLER_ALL_RNG_EXACT','checkpoint':str(p),'checkpoint_sha256':sha(p),'step':self.step})
 def evaluate(self,name,raws=PILOT,initial=False):
  before=rng();d=self.out/name;d.mkdir();(d/'frames').mkdir();table=self.table().detach().cpu().numpy();material=d/'parent_RGB.npz';np.savez_compressed(material,parent_RGB=table,eligible_parent_ids=self.ids.cpu().numpy())
  records=[]
  for raw in raws:
   folder=d/f'frames/{raw:03d}';folder.mkdir()
   for state in ['keep','delete']:
    with torch.no_grad():out=self.render(raw,state)
    rgb=out['rgb'].detach().cpu().numpy();alpha=out['alpha'].detach().cpu().numpy()[0];depthsum=out['depth_sum'].detach().cpu().numpy()[0]
    values={'rgb':rgb,'alpha':alpha,'depth_sum':depthsum,'depth_m':depthsum/np.maximum(alpha,1e-8)/self.semantic.scale,'group_alpha_T':out['group_alpha_T'].detach().cpu().numpy(),'added_owner_rgb_T':out['added_owner_rgb_T'].detach().cpu().numpy(),'newroad_rgb_T':out['newroad_rgb_T'].detach().cpu().numpy()}
    errors={};ref=R/f'C_warm0_pilot/frames/{raw:03d}/{state}_float.npz'
    if ref.exists():
     with np.load(ref,allow_pickle=False) as z:
      keys=list(values) if initial else ['alpha','depth_sum','depth_m','group_alpha_T','added_owner_rgb_T']
      for key in keys:errors[key]=float(np.abs(values[key]-z[key]).max());assert np.array_equal(values[key],z[key]),(name,raw,state,key,errors[key])
    np.savez_compressed(folder/(state+'_float.npz'),**values)
    from PIL import Image
    Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(folder/(state+'.png'));records.append({'raw':raw,'state':state,'Cwarm0_frozen_max_abs':errors});del out
  set_rng(before);self.runtime.assert_unchanged(full=True)
  write(d/'receipt.json',{'status':'PASS_C49_PUBLIC_MATERIAL_NODE_GEOMETRY_FROZEN_PENDING_PHOTOMETRIC','step':self.step,'raw_ids':list(raws),'actual_native_forwards':2*len(raws),'optimizer_updates_in_evaluation':0,'parent_RGB_file':str(material),'parent_RGB_sha256':sha(material),'initial_Cwarm0_exact':initial,'records':records,'runtime_receipt':self.runtime.receipt(),'quality_accepted':False})
  return d
 def frozen(self):
  self.runtime.assert_unchanged(full=True);assert baseline_appearance_binding(self.semantic)==self.binding
  after={p:{'sha256':sha(p),'bytes':Path(p).stat().st_size,'mtime_ns':Path(p).stat().st_mtime_ns} for p in self.guard};assert after==self.guard;write(self.out/'source_target_guard_after.json',after)
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','resume2','formal'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path);a=p.parse_args();c=Core(a.output)
 if a.mode=='smoke':
  c.evaluate('initial_Cwarm0_public',initial=True)
  while sum(c.times)<600:
   c.paired()
   if c.step==49:
    early=c.save('early_state49.pth');c.evaluate('early_evaluation49');exact(torch.load(early,map_location='cpu',weights_only=False),c.state())
  stop=c.step;saved=c.save(f'state{stop}.pth');c.evaluate('evaluation');exact(torch.load(saved,map_location='cpu',weights_only=False),c.state())
  training_seconds=sum(c.times)
  for _ in range(2):c.paired()
  disposable=c.save(f'disposable_state{c.step}.pth');c.frozen()
  result={'status':'PASS_REAL600S_C49_SMOKE_SAVE_EVAL_DISPOSABLE_NEXT2','smoke_updates':stop,'disposable_updates':2,'checkpoint':str(saved),'disposable_checkpoint':str(disposable),'actual_training_seconds_before_disposable':training_seconds,'formal_not_started':True}
 elif a.mode=='resume2':
  assert a.checkpoint;c.restore(a.checkpoint)
  for _ in range(2):c.paired()
  saved=c.save(f'restored_state{c.step}.pth');c.frozen();result={'status':'PASS_C49_FRESH_FULL_RESTORE_NEXT2','checkpoint':str(saved)}
 else:
  for _ in range(2000):
   c.paired()
   if c.step in [500,2000]:
    c.save(f'state{c.step}.pth');c.evaluate(f'evaluation_{c.step}');c.frozen()
    request=R/'stop_at_next_node.json'
    if request.exists():break
  c.frozen();result={'status':'COMPLETE_C49_FRESH_FORMAL_TO_NODE_PENDING_ROOT_RELEASE','final_step':c.step,'maximum_registered_formal_updates':10000,'next_nodes_require_parent_release':[5000,10000]}
 result.update(actual_optimizer_updates=c.updates,actual_native_forwards=c.calls,actual_native_backwards=c.backs,original_scene_Gaussian_SH_opacity_geometry_identity_updates=0,
  only_new_C_parent_RGB_updates=c.updates,eligible_parents=2119,parent_RGB_bounds=[0,1],RGB_min=float(c.rgb.min()),RGB_max=float(c.rgb.max()),training_contract_sha256=sha(CONTRACT),source_and_target_guards_unchanged=True,
  seconds=time.time()-c.start,max_reserved_cuda_bytes=int(torch.cuda.max_memory_reserved()),quality_accepted=False,all49_auto_release=False,shutdown=False,reset_card=False)
 write(c.out/'returned.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()

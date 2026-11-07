"""Measure native joint point visibility against fixed BUILD identity masks; no fitting."""
from pathlib import Path
import hashlib, importlib.util, json, os, sys, time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_full5_integration_20261107_r1')
MODEL=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full_dynamic_edit_20261107_r1/inference10k')
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
MASKS=ROOT/'inputs/semantic_actor5_masks4_v1'
HELPER=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023/engine/utils/v4_road_render.py')
OUT=ROOT/'semantic_actor5_calibration4'
RAWS=[94,105,130,137]
MASK_SHA={94:'59c6b10df2621051f8a3e469e3d6e1ff1c710bdbf26e9695e6a2cc0b28871c32',105:'4a9616f4165be68f47fe7e8789815e56675daec8bfff0674bd2ae5f08d0a9d30',130:'832a0efcbc291b1bfc02a0c26478ca5448d8df8dd864e8b4aa5951fc8a13df53',137:'832a0efcbc291b1bfc02a0c26478ca5448d8df8dd864e8b4aa5951fc8a13df53'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def identity(path):
 p=Path(path);return {'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)}

def main():
 import numpy as np
 from PIL import Image
 started=time.time();assert not OUT.exists();OUT.mkdir();(OUT/'point_mass').mkdir()
 assert sha(HELPER)=='888dbe0c49eff4fa0fcf0e3999b49d2b6a44c746f806d45df869c22e8520e943'
 mask_values={};mask_files={}
 for raw in RAWS:
  p=MASKS/f'{raw:03d}.png';assert sha(p)==MASK_SHA[raw]
  mask=np.asarray(Image.open(p).convert('L'))>0;assert mask.shape==(320,480)
  assert int(mask.sum())=={94:2052,105:10773,130:0,137:0}[raw]
  mask_values[raw]=mask;mask_files[str(raw)]=identity(p)
 source_files=[MODEL/x for x in ['chkpnt10000.pth','dynamic_chkpnt10000.pth','env_light_chkpnt10000.pth','effective_config.json','cameras.json','actual_scene_binding.json']]
 sources={str(p):identity(p) for p in source_files}
 contract={'schema':'scene008.semantic_actor5.visible_mass.protocol.v1','status':'SEALED_BEFORE_GPU_CALIBRATION',
  'raw_ids':RAWS,'iteration':10000,'model':str(MODEL),'mask_role':'Original BUILD actor5 identity labels, NN-derived before measuring; not a held-out quality domain. QUERY must not read these masks.',
  'mask_manifest':identity(MASKS/'manifest.json'),'masks':mask_files,'source_files':sources,
  'input_roles':{'Gaussian_parameters':'frozen read-only model','mask':'BUILD identity calibration only','original_owner_and_lineage':'provenance retained; not overwritten','evaluation_ground_truth':'not read for selection'},
  'loss':'extra0[M5].sum()+extra1[~M5].sum(); no domain mean or per-view reweighting',
  'aggregation':'Sum actual inside/outside visible alpha_T masses over the four views',
  'threshold':{'purity_strictly_greater_than':.5,'total_mass_strictly_greater_than':0.},
  'self_checks':{'inside_mass_relative_error_max':1e-5,'outside_mass_relative_error_max':1e-5,'dummyones_alpha_max_abs':1e-6,'point_mass_finite_and_nonnegative':True,'model_grad_none':True},
  'planned_joint_forwards':4,'planned_dummy_feature_backwards':4,'optimization_updates':0,'optimizer_created':False,
  'no_automatic_composition_or_all49':True,'script':identity(Path(__file__)),'raster_helper':identity(HELPER),'shutdown':False,'reset_card':False}
 write(OUT/'calibration_protocol.json',contract)
 sys.path.insert(0,str(REPO))
 import torch
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.scene_io import load_native_models,make_camera,camera_headers,model_tensors
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.ownership import actor_ids
 torch.set_num_threads(1);assert torch.cuda.is_available()
 spec=importlib.util.spec_from_file_location('calibration_original_raster',HELPER);raster=importlib.util.module_from_spec(spec);sys.modules[spec.name]=raster;spec.loader.exec_module(raster)
 pipe,models,renderer=load_native_models(MODEL,10000)
 assert all(getattr(m,'optimizer',None) is None for m in models.values())
 assert models['static'].active_sh_degree==models['dynamic'].active_sh_degree==3
 tensors=model_tensors(models);assert all(not t.requires_grad and t.grad is None for t in tensors.values())
 def fingerprints():
  out={}
  for name,t in sorted(model_tensors(models).items()):
   a=t.detach().cpu().contiguous().numpy();h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());out[name]=h.hexdigest()
  return out
 before=fingerprints();write(OUT/'model_tensor_fingerprints_before.json',before)
 ns=len(models['static'].get_xyz);nd=len(models['dynamic'].get_xyz);n=ns+nd
 owners=actor_ids(models['dynamic']).detach().cpu().numpy().astype(np.int64);assert owners.shape==(nd,)
 assert sorted(np.unique(owners).tolist())==[5,6,18,26,37]
 lineage_s=models['static']._sgcm_lineage_ids.detach().cpu().numpy();lineage_d=models['dynamic']._sgcm_lineage_ids.detach().cpu().numpy()
 assert lineage_s.shape==(ns,) and lineage_d.shape==(nd,)
 np.savez_compressed(OUT/'original_owner_and_lineage.npz',static_lineage_ids=lineage_s,dynamic_lineage_ids=lineage_d,
  dynamic_original_owner_ids=owners,dynamic_source_owner_table=models['dynamic']._dynamic_actor_source_ids.detach().cpu().numpy())
 source_ids=np.concatenate([np.zeros(ns,dtype=np.int64),owners]);source_names=[('static',0),('5',5),('6',6),('18',18),('26',26),('37',37)]
 accumulated=np.zeros((n,2),dtype=np.float64);frames=[];cached=[]
 background=torch.ones(3,device='cuda') if bool(pipe.white_background) else torch.zeros(3,device='cuda')
 headers=camera_headers(MODEL)
 from utils.sh_utils import eval_sh
 for raw in RAWS:
  camera=make_camera(headers[raw]);mask=mask_values[raw];cuda_mask=torch.as_tensor(mask,device='cuda')
  with torch.no_grad():
   sr=renderer.prepare_rasterization(camera,models['static'],pipe,other=[])
   dr=renderer.prepare_rasterization(camera,models['dynamic'],pipe,other=[],dynamic=True)
   kw=raster.merge_rows(sr,dr);assert kw['features'].shape==(n,2)
   kw['means2D']=torch.zeros_like(kw['means3D'])
   point_colours=[]
   for rows,m in [(sr,models['static']),(dr,models['dynamic'])]:
    colour=rows['colors_precomp']
    if colour is None:
     direction=rows['means3D']-camera.camera_center;direction=direction/direction.norm(dim=-1,keepdim=True)
     colour=(eval_sh(m.active_sh_degree,rows['shs'].transpose(1,2),direction)+.5).clamp_min(0)
    point_colours.append(colour.detach())
   colours=torch.cat(point_colours).cpu().numpy()
  dummy=torch.ones((n,2),dtype=kw['features'].dtype,device='cuda',requires_grad=True)
  kw['features']=torch.cat([kw['features'],dummy],dim=1)
  rendered=raster.raster(camera,pipe,models['static'].active_sh_degree,kw,background)
  write(OUT/'kernel_execution_progress.json',{'current_raw':raw,'completed_joint_forwards':len(frames)+1,'completed_dummy_backwards':len(frames),'optimization_updates':0})
  assert rendered['extra'].shape==(2,320,480)
  alpha=rendered['alpha'].detach()[0]
  dummy_error=[float((rendered['extra'][c].detach()-alpha).abs().max()) for c in [0,1]]
  assert max(dummy_error)<=1e-6,dummy_error
  loss=rendered['extra'][0][cuda_mask].sum()+rendered['extra'][1][~cuda_mask].sum()
  loss.backward();write(OUT/'kernel_execution_progress.json',{'current_raw':raw,'completed_joint_forwards':len(frames)+1,'completed_dummy_backwards':len(frames)+1,'optimization_updates':0})
  mass=dummy.grad.detach().cpu().numpy().copy()
  assert np.isfinite(mass).all() and np.min(mass)>=0
  measured=mass.astype(np.float64).sum(0)
  alpha_np=alpha.cpu().numpy().astype(np.float64);expected=np.array([alpha_np[mask].sum(),alpha_np[~mask].sum()])
  absolute=np.abs(measured-expected);relative=np.divide(absolute,np.abs(expected),out=np.zeros(2),where=expected!=0)
  assert np.all(measured[expected==0]==0) and np.max(relative)<=1e-5,(measured,expected,relative)
  assert all(t.grad is None for t in model_tensors(models).values())
  accumulated+=mass.astype(np.float64)
  np.savez_compressed(OUT/f'point_mass/{raw:03d}.npz',inside_mass=mass[:,0],outside_mass=mass[:,1],branch_boundary=np.int64(ns))
  with torch.no_grad():
   env=models['environment'](camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
   rgb=(rendered['rgb']+(1-rendered['alpha'])*env).detach().cpu().numpy()
  Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(OUT/f'{raw:03d}_native_keep.png')
  frame={'raw':raw,'mask_pixels':int(mask.sum()),'inside_mass_sum':float(measured[0]),'outside_mass_sum':float(measured[1]),
   'joint_alpha_inside_sum':float(expected[0]),'joint_alpha_outside_sum':float(expected[1]),'mass_sum_abs_error':absolute.tolist(),
   'mass_sum_relative_error':relative.tolist(),'dummyones_alpha_max_abs':dummy_error,'joint_forwards':1,'dummy_feature_backwards':1,'model_grad_none':True}
  frames.append(frame);cached.append((raw,mass,colours));write(OUT/'progress.json',{'completed_raw_ids':[x['raw'] for x in frames],'joint_forwards':len(frames),'dummy_feature_backwards':len(frames),'optimization_updates':0})
  print(json.dumps(frame),flush=True)
  del rendered,kw,sr,dr,dummy,loss,camera,alpha,env,point_colours
  torch.cuda.empty_cache()
 inside=accumulated[:,0];total=accumulated.sum(1);purity=np.divide(inside,total,out=np.zeros(n),where=total>0)
 selected=(purity>.5)&(total>0)
 np.savez_compressed(OUT/'semantic_actor5_rows.npz',static_row_ids=np.flatnonzero(selected[:ns]).astype(np.int64),dynamic_row_ids=np.flatnonzero(selected[ns:]).astype(np.int64),
  static_inside_mass=inside[:ns],static_total_mass=total[:ns],dynamic_inside_mass=inside[ns:],dynamic_total_mass=total[ns:])
 matrix=[]
 for name,owner in source_names:
  source=source_ids==owner;chosen=source&selected
  matrix.append({'source':name,'rows':int(source.sum()),'observed_rows':int((source&(total>0)).sum()),'unobserved_rows':int((source&(total==0)).sum()),
   'selected_rows':int(chosen.sum()),'all_inside_mass':float(inside[source].sum()),'all_total_mass':float(total[source].sum()),
   'selected_inside_mass':float(inside[chosen].sum()),'selected_outside_mass':float((total[chosen]-inside[chosen]).sum()),
   'selected_total_mass':float(total[chosen].sum())})
 for raw,mass,colours in cached:
  frame=next(x for x in frames if x['raw']==raw);per_total=mass.astype(np.float64).sum(1);per_inside=mass[:,0].astype(np.float64)
  selected_inside=float(per_inside[selected].sum());selected_total=float(per_total[selected].sum())
  frame['selected_semantic5']={'inside_mass':selected_inside,'outside_mass':selected_total-selected_inside,'total_mass':selected_total,
   'captured_fraction_of_original_a5_mass':selected_inside/float(per_inside.sum()) if per_inside.sum()>0 else None,
   'outside_fraction_of_selected_mass':(selected_total-selected_inside)/selected_total if selected_total>0 else None,
   'selected_inside_RGB_T_sum':(per_inside[selected,None]*colours[selected]).sum(0,dtype=np.float64).tolist(),
   'unselected_inside_RGB_T_sum':(per_inside[~selected,None]*colours[~selected]).sum(0,dtype=np.float64).tolist(),
   'RGB_T_note':'Computed from visible point masses and original per-view SH colours; no fitting or pixel composite.'}
 assert before==fingerprints();assert all(t.grad is None for t in model_tensors(models).values())
 for path,old in sources.items():assert identity(path)==old,path
 for raw,old in mask_files.items():assert identity(old['path'])==old
 manifest={'schema':'scene008.semantic_actor5.visible_mass.v1','iteration':10000,'raw_ids':RAWS,
  'threshold':{'purity_strictly_greater_than':.5,'total_mass_strictly_greater_than':0.},
  'selection_file':'semantic_actor5_rows.npz','selection_sha256':sha(OUT/'semantic_actor5_rows.npz'),
  'branch_rows':{'static':ns,'dynamic':nd},'gaussian_files_sha256':{p.name:sources[str(p)]['sha256'] for p in source_files[:2]},
  'gaussian_parameter_updates':0,'protocol_file':'calibration_protocol.json','protocol_sha256':sha(OUT/'calibration_protocol.json'),
  'ancestry_file':'original_owner_and_lineage.npz','ancestry_sha256':sha(OUT/'original_owner_and_lineage.npz'),
  'query_requires_BUILD_masks':False,'source_model':str(MODEL)}
 write(OUT/'semantic_actor5_manifest.json',manifest)
 receipt={'status':'PASS_VISIBLE_MASS_IDENTITY_CALIBRATION_ENGINEERING_PENDING_ROOT_REVIEW','raw_ids':RAWS,'joint_forwards':4,'dummy_feature_backwards':4,
  'optimization_updates':0,'optimizer_created':False,'model_parameters_environment_and_original_identity_unchanged':True,
  'source_files_unchanged':True,'mask_files_unchanged':True,'branch_rows':{'static':ns,'dynamic':nd},'selected_rows':{'static':int(selected[:ns].sum()),'dynamic':int(selected[ns:].sum())},
  'selection_capture_fraction':float(inside[selected].sum()/inside.sum()),'selection_spill_fraction':float((total[selected]-inside[selected]).sum()/total[selected].sum()),
  'source_matrix':matrix,'frames':frames,'sidecar_manifest':'semantic_actor5_manifest.json','source_protocol_sha256':sha(OUT/'calibration_protocol.json'),
  'max_reserved_cuda_bytes':int(torch.cuda.max_memory_reserved()),'seconds':time.time()-started,
  'held_out_quality_validated':False,'automatic_composition_or_all49':False,'quality_accepted':False,'shutdown':False,'reset_card':False}
 write(OUT/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ['frames','source_matrix']}),flush=True)

if __name__=='__main__':main()

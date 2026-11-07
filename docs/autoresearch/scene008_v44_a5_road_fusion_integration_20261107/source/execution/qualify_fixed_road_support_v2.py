"""Read per-child native visible mass; never optimize any scene parameter."""
from pathlib import Path
import hashlib,json,os,sys,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=B/'scene008_v44_a5_road_fusion_integration_20261107_r1';OUT=ROOT/'support_qualification4_v2'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');SEM=B/'scene008_v44_full5_integration_20261107_r1'
OBS=B/'scene008_instances_20261105_r1/observations/frames'
REF=B/'scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz'
RAWS=[94,105,128,137];N=40392
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 import numpy as np
 started=time.time();assert not OUT.exists();OUT.mkdir();(OUT/'point_mass').mkdir()
 amendment=ROOT/'contract/fixed_build_support_amendment_v2.json';assert amendment.exists()
 domains_module=ROOT/'source/fixed_domains.py';assert sha(domains_module)=='a1724b86c90f1ca0b58540fc46182a94e82472679dd84f70ea9fecfdecfbda9d'
 paths=[amendment,ROOT/'contract/independent_evaluation_contract_v1.json',REF]+[OBS/f'{raw:03d}.npz' for raw in RAWS]
 paths += [ROOT/f'baseline4/frames/{raw:03d}/{state}_float.npz' for raw in RAWS for state in ['keep','delete']]
 inputs={str(p):identity(p) for p in paths}
 protocol={'schema':'fixed-road-support-v2-protocol','status':'SEALED_BEFORE_QUALIFICATION_GPU','raw_ids':RAWS,'states':['keep','delete'],
  'road_domain':'KEEP obs.road; DELETE obs.road OR (obs.road_edit AND frozen reference ground)',
  'guard_domain':'fixed_nonroad_geometry_guard OR a5 OR a6 OR latest-baseline-C2alpha>0 OR latest-baseline-a5alpha>0; KEEP additionally a37 OR latest-baseline-a37alpha>0',
  'input_role':'Posthoc BUILD qualification on these four views; not held-out. No RGB or current-C alpha is used to select the guard.',
  'inputs':inputs,'source':identity(Path(__file__)),'row_count':N,'threshold':{'inside_strictly_greater_than':0.,'outside_equal':0.},
  'dummy_features':'ones at full features16:18 only for newroad tail; extra14:16 must equal newroad alphaT extra7',
  'mass_sum_relative_error_limit':1e-5,'dummy_alpha_max_abs_limit':1e-6,
  'cached_C_parity_limits':{'rgb':1e-6,'alpha':1e-6,'depth_m':1e-5,'group_alpha_T':1e-6},
  'planned_direct_joint_forwards':8,'planned_dummy_feature_backwards':8,'optimizer_created':False,'model_updates':0,'colour_updates':0,
  'occluded_full_support_is_not_a_protection_proof':True,'all49_auto_release':False,'shutdown':False,'reset_card':False}
 write(OUT/'qualification_protocol.json',protocol)
 sys.path[:0]=[str(ROOT/'source'),str(B/'scene008_v44_rgb_asset_20261107_r1/source'),str(REPO)]
 from fixed_domains import fixed_domains
 import torch
 from v44_runtime import V44Runtime
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration,C_MANIFEST
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.fixed_road_support import runtime_support_binding
 torch.set_num_threads(1);assert torch.cuda.is_available()
 base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json')
 semantic=SemanticActor5V44Integration(base,semantic_sidecar_path=SEM/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=ROOT/'contract/full5_transform_matrix.txt')
 rt=RoadFusionV44Integration(semantic,C_MANIFEST);binding=runtime_support_binding(rt);write(OUT/'model_binding_before.json',binding)
 with np.load(REF,allow_pickle=False) as z:ground=np.asarray(z['depth_m'])[:,3]>0
 assert ground.shape==(49,320,480)
 total_inside=np.zeros(N,dtype=np.float64);total_outside=np.zeros(N,dtype=np.float64);rows=[];forward_count=0;backward_count=0
 for raw in RAWS:
  with np.load(OBS/f'{raw:03d}.npz',allow_pickle=False) as z:
   obs={k:z[k].copy() for k in ['actor_labels','visible','road','road_edit','protect']}
  domains=fixed_domains(obs)
  for state in ['keep','delete']:
   with np.load(ROOT/f'baseline4/frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:baseline_groups=z['group_alpha_T'].copy()
   inside=domains['road'].copy()
   if state=='delete':inside|=domains['road_edit']&ground[raw-89]
   guard=domains['fixed_nonroad_geometry_guard']|domains['a5']|domains['a6']|(baseline_groups[0]>0)|(baseline_groups[4]>0)
   if state=='keep':guard|=domains['a37']|(baseline_groups[1]>0)
   scene=rt.scene(raw,delete=state=='delete');n=int(scene['meta']['latest_semantic_prefix_rows'])
   assert len(scene['rows']['mask'])==n+N
   original=scene['rows'];kw={k:v.detach() if torch.is_tensor(v) else v for k,v in original.items()}
   assert kw['features'].shape==(n+N,24) and not bool((kw['features'][:,16:18]!=0).any())
   probe=torch.ones((N,2),dtype=kw['features'].dtype,device=kw['features'].device,requires_grad=True)
   dummy=torch.cat([torch.zeros((n,2),dtype=probe.dtype,device=probe.device),probe],0)
   kw['features']=torch.cat([kw['features'][:,:16],dummy,kw['features'][:,18:]],1)
   assert all(not v.requires_grad for k,v in kw.items() if torch.is_tensor(v) and k!='features')
   out=base.raster(scene['cam'],base.pipe,0,kw,scene['bg']);forward_count+=1
   write(OUT/'kernel_progress.json',{'raw':raw,'state':state,'joint_forwards':forward_count,'dummy_backwards':backward_count,'model_updates':0})
   support=out['extra'][7].detach();dummy_errors=[float((out['extra'][14+i].detach()-support).abs().max()) for i in range(2)]
   assert max(dummy_errors)<=1e-6,dummy_errors
   im=torch.as_tensor(inside,device=probe.device);gm=torch.as_tensor(guard,device=probe.device)
   loss=out['extra'][14][im].sum()+out['extra'][15][gm].sum()
   mass=torch.autograd.grad(loss,probe,retain_graph=False,create_graph=False)[0].detach().cpu().numpy();backward_count+=1
   write(OUT/'kernel_progress.json',{'raw':raw,'state':state,'joint_forwards':forward_count,'dummy_backwards':backward_count,'model_updates':0})
   assert mass.shape==(N,2) and np.isfinite(mass).all() and mass.min()>=0
   sup=support.cpu().numpy().astype(np.float64);expected=np.array([sup[inside].sum(),sup[guard].sum()]);measured=mass.astype(np.float64).sum(0)
   absolute=np.abs(expected-measured);relative=np.divide(absolute,np.abs(expected),out=np.zeros(2),where=expected!=0)
   assert np.all(measured[expected==0]==0) and relative.max()<=1e-5,(absolute,relative)
   rgb=(out['rgb']+(1-out['alpha'])*scene['env']).detach().cpu().numpy();alpha=out['alpha'].detach().cpu().numpy()[0]
   depth=out['depth_sum'].detach().cpu().numpy()[0]/np.maximum(alpha,1e-8)/base.scale;groups=out['extra'][:8].detach().cpu().numpy()
   parity={}
   with np.load(ROOT/f'C4_retry1/frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:
    for k,value in [('rgb',rgb),('alpha',alpha),('depth_m',depth),('group_alpha_T',groups)]:
     parity[k]=float(np.abs(value-z[k]).max());assert parity[k]<=protocol['cached_C_parity_limits'][k],(k,parity[k])
   total_inside+=mass[:,0].astype(np.float64);total_outside+=mass[:,1].astype(np.float64)
   np.savez_compressed(OUT/f'point_mass/{raw:03d}_{state}.npz',inside_mass=mass[:,0],outside_mass=mass[:,1],inside_mask=inside,guard_mask=guard)
   row={'raw':raw,'state':state,'inside_pixels':int(inside.sum()),'guard_pixels':int(guard.sum()),'inside_mass_sum':float(measured[0]),'outside_mass_sum':float(measured[1]),
    'readout_sum_expected':expected.tolist(),'mass_sum_absolute_error':absolute.tolist(),'mass_sum_relative_error':relative.tolist(),'dummy_to_newroad_alpha_max_abs':dummy_errors,
    'cached_C_parity_max_abs':parity,'only_probe_gradient_requested':True,'original_rows_all_detached':True}
   rows.append(row);write(OUT/'progress.json',{'completed':rows,'joint_forwards':forward_count,'dummy_backwards':backward_count,'model_updates':0});print(json.dumps(row),flush=True)
   rt.assert_unchanged();del scene,original,kw,out,probe,dummy,loss,support;torch.cuda.empty_cache()
  base._data.clear()
 safe=(total_inside>0)&(total_outside==0);ids=np.flatnonzero(safe).astype(np.int64)
 np.savez_compressed(OUT/'safe_road_children.npz',safe_child_row_ids=ids,inside_mass=total_inside,outside_mass=total_outside)
 assert binding==runtime_support_binding(rt)
 for path,before in inputs.items():assert identity(path)==before,path
 manifest={'schema':'fixed-road-support-v1','asset_sha256':rt.manifest['new_road']['sha256'],'row_count':N,'selection_file':'safe_road_children.npz',
  'selection_sha256':sha(OUT/'safe_road_children.npz'),'calibration_raw_ids':RAWS,'states':['keep','delete'],
  'threshold':{'inside_strictly_greater_than':0.,'outside_equal':0.},'model_binding':binding,'model_updates':0,'colour_updates':0,'safe_rows':int(safe.sum()),
  'calibration_support_mode':'full40392_unfiltered','guard_protocol':'fixed_build_support_amendment_v2','guard_protocol_sha256':sha(amendment),
  'source_qualification_protocol_sha256':sha(OUT/'qualification_protocol.json'),'baseline_float_input_sha256':{str(p):inputs[str(p)]['sha256'] for p in paths if 'baseline4/frames' in str(p)},
  'query_does_not_read_guard_masks_or_query_RGB':True,'occluded_full_support_is_not_a_protection_proof':True}
 write(OUT/'fixed_support_manifest.json',manifest)
 receipt={'status':'PASS_FIXED_BUILD_SUPPORT_QUALIFICATION_ENGINEERING_PENDING_D_RENDER' if len(ids) else 'PASS_QUALIFICATION_EMPTY_SAFE_SET_SKIP_D',
  'raw_ids':RAWS,'states':['keep','delete'],'joint_forwards':forward_count,'dummy_feature_backwards':backward_count,'optimizer_steps':0,'model_updates':0,'colour_updates':0,
  'safe_rows':len(ids),'rejected_rows':N-len(ids),'inside_positive_rows':int((total_inside>0).sum()),'outside_positive_rows':int((total_outside>0).sum()),
  'all_source_files_and_binding_unchanged':True,'rows':rows,'source_model_parameters_detached_for_every_raster':True,'only_probe_autograd_grad':True,
  'manifest_sha256':sha(OUT/'fixed_support_manifest.json'),'seconds':time.time()-started,'actual_D_render_required_for_protection':True,
  'max_reserved_cuda_bytes':int(torch.cuda.max_memory_reserved()),'all49_auto_release':False,'shutdown':False,'reset_card':False}
 write(OUT/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()

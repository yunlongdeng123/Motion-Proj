"""Measure fixed-view protected RGB coefficient constraints; no colour fitting."""
from pathlib import Path
import hashlib,json,os,sys,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_v44_a5_road_fusion_integration_20261107_r1';S=B/'scene008_v44_full5_integration_20261107_r1'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');OUT=R/'F4_bounded_appearance_qualification';OBS=B/'scene008_instances_20261105_r1/observations/frames';REF=B/'scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz'
RAWS=[94,105,128,137];NS=1042776;NR=143248;N=NS+NR;EPS=5e-7
NAMES=['a5_full','a5_upper','a6_visible','protected_nonroad','a37_visible_keep']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 import numpy as np
 start=time.time();assert not OUT.exists();OUT.mkdir();(OUT/'point_mass').mkdir()
 contract=R/'contract/bounded_appearance_constraint_feasibility_v1.json';assert sha(contract)=='81194d78c1aa130485a8907f40278924361348b755bd64d8e3e16f346d108417'
 paths=[contract,R/'contract/independent_evaluation_contract_v1.json',REF]+[OBS/f'{raw:03d}.npz' for raw in RAWS]
 paths += [R/f'baseline4/frames/{raw:03d}/{state}_float.npz' for raw in RAWS for state in ['keep','delete']]
 inputs={str(p):ident(p) for p in paths}
 protocol={'status':'SEALED_BEFORE_BOUNDED_APPEARANCE_FEASIBILITY_GPU','schema':'F4_analytic_protection_mass_protocol_v1','raw_ids':RAWS,'states':['keep','delete'],
  'inputs':inputs,'source':ident(Path(__file__)),'scene':'latest semantic baseline, no AA/C/D/E geometry or material',
  'row_order':'native_static1042776 followed by oldroad143248, each stable branch-local order',
  'feature_layout':'f18:24 six ones only on these two branches; extra16:22 = inside and original five protected domains',
  'domains':NAMES,'inside':'Original road; DELETE road OR (road_edit & fixed ground)',
  'candidate_predicate':'inside_mass>0; no zero-spill requirement and no RGB selection',
  'bound':'weights=max_m domain_mass[m,r]/(3*Npixels[m]*5e-7); future sum_r weights[r]*L1(delta[r])<=1',
  'epsilon':EPS,'original_tolerance':1e-6,'loss':'SUM, one independent dummy channel/domain; never divide before coefficient measurement',
  'empty_domains':'Not included in M, recorded explicitly; no invented denominator',
  'planned_joint_forwards':8,'planned_dummy_feature_backwards':8,'optimizer_steps':0,'model_updates':0,'colour_updates':0,'automatic_fit':False,'automatic_all49':False,'shutdown':False,'reset_card':False}
 write(OUT/'qualification_protocol.json',protocol)
 sys.path[:0]=[str(R/'source'),str(B/'scene008_v44_rgb_asset_20261107_r1/source'),str(REPO)]
 from fixed_domains import fixed_domains
 import torch
 from v44_runtime import V44Runtime
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.baseline_road_appearance import baseline_appearance_binding
 torch.set_num_threads(1);assert torch.cuda.is_available()
 base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json');rt=SemanticActor5V44Integration(base,semantic_sidecar_path=S/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=R/'contract/full5_transform_matrix.txt')
 binding=baseline_appearance_binding(rt);write(OUT/'model_binding_before.json',binding)
 with np.load(REF,allow_pickle=False) as z:ground=np.asarray(z['depth_m'])[:,3]>0
 inside_mass=np.zeros(N,np.float64);matrix=[];domain_records=[];empty=[];records=[];base_cache=[];calls=0;backs=0
 for raw in RAWS:
  with np.load(OBS/f'{raw:03d}.npz',allow_pickle=False) as z:obs={k:z[k].copy() for k in ['actor_labels','visible','road','road_edit','protect']}
  dom=fixed_domains(obs);upper=np.zeros((320,480),bool)
  yy=np.nonzero(dom['a5'])[0]
  if len(yy):upper=dom['a5']&(np.arange(320)[:,None]<=yy.min()+.25*(yy.max()-yy.min()))
  for state in ['keep','delete']:
   inside=dom['road'].copy()
   if state=='delete':inside|=dom['road_edit']&ground[raw-89]
   masks=[inside,dom['a5'],upper,dom['a6'],dom['protected_nonroad'],dom['a37'] if state=='keep' else np.zeros((320,480),bool)]
   scene=rt.scene(raw,delete=state=='delete');kw={k:v.detach() if torch.is_tensor(v) else v for k,v in scene['rows'].items()}
   a=torch.where(kw['features'][:,5]==1)[0];b=torch.where(kw['features'][:,4]==1)[0];assert len(a)==NS and len(b)==NR
   idx=torch.cat([a,b]);assert len(torch.unique(idx))==N and not bool((kw['features'][:,18:24]!=0).any())
   colours=kw['colors_precomp'][idx].detach().cpu().numpy().copy();active=kw['mask'][idx].detach().cpu().numpy().astype(bool)
   probe=torch.ones((N,6),device=kw['features'].device,dtype=kw['features'].dtype,requires_grad=True)
   full=torch.zeros((len(kw['features']),6),device=probe.device,dtype=probe.dtype).index_copy(0,idx,probe);kw['features']=torch.cat([kw['features'][:,:18],full],1)
   assert all(not v.requires_grad for k,v in kw.items() if torch.is_tensor(v) and k!='features')
   out=base.raster(scene['cam'],base.pipe,0,kw,scene['bg']);calls+=1;write(OUT/'kernel_progress.json',{'raw':raw,'state':state,'joint_forwards':calls,'dummy_backwards':backs,'model_updates':0})
   readout=out['extra'][16:22];pair_error=max(float((readout[i].detach()-readout[0].detach()).abs().max()) for i in range(1,6));assert pair_error==0
   target_error=float((readout[0].detach()-(out['extra'][2]+out['extra'][3]).detach()).abs().max());assert target_error<=2e-5
   loss=sum(readout[i][torch.as_tensor(mask,device=probe.device)].sum() for i,mask in enumerate(masks))
   mass=torch.autograd.grad(loss,probe,retain_graph=False,create_graph=False)[0].detach().cpu().numpy();backs+=1
   write(OUT/'kernel_progress.json',{'raw':raw,'state':state,'joint_forwards':calls,'dummy_backwards':backs,'model_updates':0})
   assert mass.shape==(N,6) and np.isfinite(mass).all() and mass.min()>=0
   ro=readout.detach().cpu().numpy().astype(np.float64);expected=np.array([ro[i][m].sum() for i,m in enumerate(masks)]);measured=mass.astype(np.float64).sum(0)
   error=np.abs(expected-measured);relative=np.divide(error,np.abs(expected),out=np.zeros(6),where=expected!=0)
   assert np.all(measured[expected==0]==0) and relative.max()<=1e-5
   alpha=out['alpha'].detach().cpu().numpy()[0];rgb=(out['rgb']+(1-out['alpha'])*scene['env']).detach().cpu().numpy();depth=out['depth_sum'].detach().cpu().numpy()[0]/np.maximum(alpha,1e-8)/base.scale
   parity={}
   with np.load(R/f'baseline4/frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:
    for key,value in [('rgb',rgb),('alpha',alpha),('depth_m',depth),('group_alpha_T',out['extra'][:7].detach().cpu().numpy())]:
     parity[key]=float(np.abs(value-z[key]).max());assert np.array_equal(value,z[key]),(raw,state,key,parity[key])
   inside_mass+=mass[:,0].astype(np.float64)
   for i,name in enumerate(NAMES):
    m=masks[i+1];d={'raw':raw,'state':state,'domain':name,'pixels':int(m.sum()),'mask_sha256':hashlib.sha256(np.ascontiguousarray(m,dtype=np.bool_).tobytes()).hexdigest()}
    if d['pixels']:domain_records.append(d);matrix.append(mass[:,i+1].astype(np.float64))
    else:empty.append(d)
   np.savez_compressed(OUT/f'point_mass/{raw:03d}_{state}.npz',mass=mass,static_row_count=np.int64(NS))
   record={'raw':raw,'state':state,'domain_pixels':[int(m.sum()) for m in masks],'domain_mass_sums':measured.tolist(),'mass_sum_relative_errors':relative.tolist(),'mass_sum_absolute_errors':error.tolist(),
    'dummy_channels_pair_max_abs':pair_error,'dummy_target_group_sum_max_abs':target_error,'baseline_parity_max_abs':parity}
   records.append(record);base_cache.append((raw,state,colours,active));write(OUT/'progress.json',{'rows':records,'joint_forwards':calls,'dummy_backwards':backs,'model_updates':0});print(json.dumps(record),flush=True)
   rt.assert_unchanged();del scene,kw,out,probe,full,loss,readout;torch.cuda.empty_cache()
  base._data.clear()
 candidate=np.flatnonzero(inside_mass>0).astype(np.int64);domain_mass=np.stack(matrix) if matrix else np.zeros((0,N),np.float64)
 counts=np.asarray([d['pixels'] for d in domain_records],np.float64)
 weights=np.max(domain_mass/(3.0*counts[:,None]*EPS),axis=0) if len(counts) else np.zeros(N,np.float64)
 assert weights.shape==(N,) and np.isfinite(weights).all() and weights.min()>=0
 premise=[]
 for raw,state,colours,active in base_cache:
  tested=candidate[active[candidate]];negative=(colours[tested]<0).any(1)
  premise.append({'raw':raw,'state':state,'active_candidate_rows':len(tested),'base_RGB_min':float(colours[tested].min()) if len(tested) else None,'negative_rows':int(negative.sum()),'finite':bool(np.isfinite(colours[tested]).all())})
 nonnegative=all(x['negative_rows']==0 and x['finite'] for x in premise)
 np.savez_compressed(OUT/'bounded_appearance_mass.npz',inside_mass=inside_mass,domain_mass=domain_mass,candidate_row_ids=candidate,weights=weights)
 assert binding==baseline_appearance_binding(rt)
 for path,value in inputs.items():assert ident(path)==value,path
 manifest={'schema':'bounded-road-appearance-qualification-v1','branch_rows':{'native_static':NS,'oldroad':NR},'row_order':'native_static_then_oldroad_branch_local',
  'calibration_raw_ids':RAWS,'states':['keep','delete'],'model_binding':binding,'model_updates':0,'colour_updates':0,
  'mass_matrix_file':'bounded_appearance_mass.npz','mass_matrix_sha256':sha(OUT/'bounded_appearance_mass.npz'),'domain_records':domain_records,'empty_domains':empty,
  'epsilon':EPS,'original_tolerance':1e-6,'candidate_predicate':'inside_mass>0','candidate_rows':len(candidate),'weight_rule':'max domain_mass/(3.0*pixels*epsilon)',
  'feasibility_contract_sha256':sha(contract),'base_RGB_nonnegative_premise':nonnegative,'base_RGB_premise_details':premise,'query_needs_calibration_RGB_or_masks':False}
 write(OUT/'qualification_manifest.json',manifest)
 result={'status':'PASS_BOUNDED_PROTECTION_MASS_FEASIBILITY_NO_FIT' if nonnegative else 'FAIL_NONNEGATIVE_BASE_RGB_PREMISE_NO_FIT','joint_forwards':calls,'dummy_feature_backwards':backs,
  'candidate_rows':len(candidate),'candidate_by_branch':{'native_static':int((candidate<NS).sum()),'oldroad':int((candidate>=NS).sum())},
  'nonempty_protected_domains':len(domain_records),'empty_domains':len(empty),'zero_weight_candidate_rows':int((weights[candidate]==0).sum()),
  'base_RGB_nonnegative_premise':nonnegative,'base_RGB_premise_details':premise,'all_original_model_binding_and_files_unchanged':True,'model_updates':0,'colour_updates':0,'optimizer_steps':0,
  'qualification_manifest_sha256':sha(OUT/'qualification_manifest.json'),'seconds':time.time()-start,'max_reserved_cuda_bytes':int(torch.cuda.max_memory_reserved()),'automatic_fit':False,'automatic_all49':False,'shutdown':False,'reset_card':False}
 write(OUT/'receipt.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()

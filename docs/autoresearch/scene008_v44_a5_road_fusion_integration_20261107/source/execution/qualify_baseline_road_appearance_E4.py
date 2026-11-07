"""Qualify existing static/road colour columns; no geometry or colour update."""
from pathlib import Path
import hashlib,json,os,sys,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');ROOT=B/'scene008_v44_a5_road_fusion_integration_20261107_r1'
SEM=B/'scene008_v44_full5_integration_20261107_r1';OUT=ROOT/'E4_appearance_qualification'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');OBS=B/'scene008_instances_20261105_r1/observations/frames'
REF=B/'scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz'
WORLD=B/'scene008_road_fusion_bake_20261107_r1/red_review/world_masks49_v1'
RAWS=[94,105,128,137];NS=1042776;NR=143248;N=NS+NR
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 import numpy as np
 started=time.time();assert not OUT.exists();OUT.mkdir();(OUT/'point_mass').mkdir()
 sys.path[:0]=[str(ROOT/'source'),str(B/'scene008_v44_rgb_asset_20261107_r1/source'),str(REPO)]
 from fixed_domains import fixed_domains
 from fixed_boundary_domains import registered_pair_union,primary128_pair_union
 assert sha(ROOT/'source/fixed_domains.py')=='a1724b86c90f1ca0b58540fc46182a94e82472679dd84f70ea9fecfdecfbda9d'
 assert sha(ROOT/'source/fixed_boundary_domains.py')=='8a4dd0adf4132c63209558213f669da0e3f8c4da9ca97e670a550072e040f14a'
 registry=json.loads((WORLD/'registry.json').read_text());registered={f['raw']:f for f in registry['frames']}
 primary_path=WORLD/'primary_raw128/fixed_masks_raw128.npz'
 with np.load(primary_path,allow_pickle=False) as z:primary=primary128_pair_union({k:z[k] for k in z.files})
 assert primary.sum()==68
 paths=[ROOT/'contract/fixed_build_support_amendment_v2.json',ROOT/'contract/independent_evaluation_contract_v1.json',WORLD/'registry.json',primary_path,REF]
 paths += [OBS/f'{r:03d}.npz' for r in RAWS]+[WORLD/registered[r]['mask_file'] for r in RAWS]
 paths += [ROOT/f'baseline4/frames/{r:03d}/{s}_float.npz' for r in RAWS for s in ['keep','delete']]
 inputs={str(p):identity(p) for p in paths}
 protocol={'schema':'baseline-existing-road-appearance-E4-protocol','status':'SEALED_BEFORE_QUALIFICATION_GPU','raw_ids':RAWS,'states':['keep','delete'],
  'calibration_scene':'latest_semantic_without_added_road_geometry','branch_rows':{'native_static':NS,'oldroad':NR},
  'inputs':inputs,'source':identity(Path(__file__)),'input_roles':'All four views are BUILD qualification here. No fusion target RGB/fit_mask is read, including the old HELD105/137 targets.',
  'inside':'Same original road as D: KEEP obs.road; DELETE road OR (road_edit & frozen ground)',
  'guard':'Exact V2 mask guard with fixed latest-baseline C2/a5 >0, KEEP37>0; no current candidate alpha selection.',
  'feature_layout':'Only target branches have six ones at f18:24 / extra16:22; original f0:18 unchanged.',
  'probe_domains':['inside','guard','original_road','world_registered_near','world_registered_lateral','primary128_registered_near'],
  'selection_uses_only_first_two_probe_columns':True,'threshold':{'inside_strictly_greater_than':0.,'outside_equal':0.},
  'sum_mass_relative_error_limit':1e-5,'probe_channel_pairwise_max_abs_limit':0.,'probe_vs_original_target_group_sum_max_abs_limit':2e-5,
  'baseline_parity_limits':{'rgb':1e-6,'alpha':1e-6,'depth_m':1e-5,'group_alpha_T':1e-6},
  'planned_joint_forwards':8,'planned_dummy_backwards':8,'optimizer_steps':0,'model_updates':0,'colour_updates':0,'automatic_fit':False,'automatic_all49':False,'shutdown':False,'reset_card':False}
 write(OUT/'qualification_protocol.json',protocol)
 import torch
 from v44_runtime import V44Runtime
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.baseline_road_appearance import baseline_appearance_binding
 torch.set_num_threads(1);assert torch.cuda.is_available()
 base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json')
 rt=SemanticActor5V44Integration(base,semantic_sidecar_path=SEM/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=ROOT/'contract/full5_transform_matrix.txt')
 binding=baseline_appearance_binding(rt);write(OUT/'model_binding_before.json',binding)
 with np.load(REF,allow_pickle=False) as z:ground=np.asarray(z['depth_m'])[:,3]>0
 accumulated=np.zeros((N,2),np.float64);records=[];cached=[];forward_count=0;backward_count=0
 for raw in RAWS:
  with np.load(OBS/f'{raw:03d}.npz',allow_pickle=False) as z:obs={k:z[k].copy() for k in ['actor_labels','visible','road','road_edit','protect']}
  domains=fixed_domains(obs)
  with np.load(WORLD/registered[raw]['mask_file'],allow_pickle=False) as z:world={k:z[k].copy() for k in z.files}
  near=registered_pair_union(world,registered[raw]['edges'],'near_x');lateral=registered_pair_union(world,registered[raw]['edges'],'lateral_y')
  assert (int(near.sum()),int(lateral.sum()))=={94:(0,0),105:(71,96),128:(69,0),137:(166,214)}[raw]
  for state in ['keep','delete']:
   with np.load(ROOT/f'baseline4/frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:baseline_groups=z['group_alpha_T'].copy()
   inside=domains['road'].copy()
   if state=='delete':inside|=domains['road_edit']&ground[raw-89]
   guard=domains['fixed_nonroad_geometry_guard']|domains['a5']|domains['a6']|(baseline_groups[0]>0)|(baseline_groups[4]>0)
   if state=='keep':guard|=domains['a37']|(baseline_groups[1]>0)
   masks=[inside,guard,domains['road'],near,lateral,primary if raw==128 else np.zeros((320,480),bool)]
   scene=rt.scene(raw,delete=state=='delete');original=scene['rows'];kw={k:v.detach() if torch.is_tensor(v) else v for k,v in original.items()}
   idx_s=torch.where(kw['features'][:,5]==1)[0];idx_r=torch.where(kw['features'][:,4]==1)[0]
   assert len(idx_s)==NS and len(idx_r)==NR and not bool((kw['features'][:,18:24]!=0).any())
   idx=torch.cat([idx_s,idx_r]);assert len(torch.unique(idx))==N
   probe=torch.ones((N,6),device=kw['features'].device,dtype=kw['features'].dtype,requires_grad=True)
   full=torch.zeros((len(kw['features']),6),device=probe.device,dtype=probe.dtype).index_copy(0,idx,probe)
   kw['features']=torch.cat([kw['features'][:,:18],full],1)
   assert all(not v.requires_grad for k,v in kw.items() if torch.is_tensor(v) and k!='features')
   out=base.raster(scene['cam'],base.pipe,0,kw,scene['bg']);forward_count+=1
   write(OUT/'kernel_progress.json',{'raw':raw,'state':state,'joint_forwards':forward_count,'dummy_backwards':backward_count,'model_updates':0})
   readout=out['extra'][16:22];assert readout.shape==(6,320,480)
   pair_error=max(float((readout[i].detach()-readout[0].detach()).abs().max()) for i in range(1,6));assert pair_error==0.
   target=out['extra'][2].detach()+out['extra'][3].detach();target_error=float((readout[0].detach()-target).abs().max());assert target_error<=2e-5
   loss=sum(readout[i][torch.as_tensor(m,device=probe.device)].sum() for i,m in enumerate(masks))
   mass=torch.autograd.grad(loss,probe,retain_graph=False,create_graph=False)[0].detach().cpu().numpy();backward_count+=1
   write(OUT/'kernel_progress.json',{'raw':raw,'state':state,'joint_forwards':forward_count,'dummy_backwards':backward_count,'model_updates':0})
   assert mass.shape==(N,6) and np.isfinite(mass).all() and mass.min()>=0
   ro=readout.detach().cpu().numpy().astype(np.float64);expected=np.array([ro[i][m].sum() for i,m in enumerate(masks)]);measured=mass.astype(np.float64).sum(0)
   absolute=np.abs(expected-measured);relative=np.divide(absolute,np.abs(expected),out=np.zeros(6),where=expected!=0)
   assert np.all(measured[expected==0]==0) and relative.max()<=1e-5
   alpha=out['alpha'].detach().cpu().numpy()[0];rgb=(out['rgb']+(1-out['alpha'])*scene['env']).detach().cpu().numpy()
   depth=out['depth_sum'].detach().cpu().numpy()[0]/np.maximum(alpha,1e-8)/base.scale;groups=out['extra'][:7].detach().cpu().numpy();parity={}
   with np.load(ROOT/f'baseline4/frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:
    for k,value in [('rgb',rgb),('alpha',alpha),('depth_m',depth),('group_alpha_T',groups)]:parity[k]=float(np.abs(value-z[k]).max());assert parity[k]<=protocol['baseline_parity_limits'][k]
   accumulated+=mass[:,:2].astype(np.float64)
   np.savez_compressed(OUT/f'point_mass/{raw:03d}_{state}.npz',mass=mass,static_row_count=np.int64(NS))
   record={'raw':raw,'state':state,'domain_pixels':[int(m.sum()) for m in masks],'domain_mass_sums':measured.tolist(),'readout_mass_sums':expected.tolist(),
    'mass_sum_absolute_error':absolute.tolist(),'mass_sum_relative_error':relative.tolist(),'ones_pair_error':pair_error,'ones_target_group_sum_error':target_error,'baseline_parity_max_abs':parity}
   records.append(record);cached.append((record,mass,np.array([alpha[m].astype(np.float64).sum() for m in masks])))
   write(OUT/'progress.json',{'completed':records,'joint_forwards':forward_count,'dummy_backwards':backward_count,'model_updates':0});print(json.dumps(record),flush=True)
   rt.assert_unchanged();del scene,original,kw,probe,full,out,readout,loss;torch.cuda.empty_cache()
  base._data.clear()
 safe=(accumulated[:,0]>0)&(accumulated[:,1]==0);static_ids=np.flatnonzero(safe[:NS]).astype(np.int64);road_ids=np.flatnonzero(safe[NS:]).astype(np.int64)
 np.savez_compressed(OUT/'safe_baseline_road_rows.npz',static_safe_row_ids=static_ids,road_safe_row_ids=road_ids,
  static_inside_mass=accumulated[:NS,0],static_outside_mass=accumulated[:NS,1],road_inside_mass=accumulated[NS:,0],road_outside_mass=accumulated[NS:,1])
 coverage=[]
 for record,mass,alpha_sums in cached:
  fields=[]
  for i,name in enumerate(protocol['probe_domains']):
   selected=float(mass[safe,i].astype(np.float64).sum());allmass=float(record['domain_mass_sums'][i]);supported=record['domain_pixels'][i]>0
   fields.append({'domain':name,'pixels':record['domain_pixels'][i],'supported':supported,'safe_mass':selected,'all_candidate_branch_mass':allmass,'joint_alpha_mass':float(alpha_sums[i]),
    'safe_fraction_of_candidate_branch_mass':selected/allmass if allmass>0 else None,'safe_fraction_of_joint_alpha_mass':selected/float(alpha_sums[i]) if alpha_sums[i]>0 else None,
    'safe_static_mass':float(mass[:NS,i][safe[:NS]].astype(np.float64).sum()),'safe_road_mass':float(mass[NS:,i][safe[NS:]].astype(np.float64).sum())})
  coverage.append({'raw':record['raw'],'state':record['state'],'domains':fields})
 assert binding==baseline_appearance_binding(rt)
 for path,before in inputs.items():assert identity(path)==before,path
 manifest={'schema':'fixed-baseline-road-appearance-qualification-v1','branch_rows':{'native_static':NS,'oldroad':NR},'calibration_raw_ids':RAWS,'states':['keep','delete'],
  'threshold':{'inside_strictly_greater_than':0.,'outside_equal':0.},'model_updates':0,'colour_updates':0,'model_binding':binding,
  'selection_file':'safe_baseline_road_rows.npz','selection_sha256':sha(OUT/'safe_baseline_road_rows.npz'),'safe_rows':{'native_static':len(static_ids),'oldroad':len(road_ids)},
  'calibration_scene':'latest_semantic_without_added_road_geometry','guard_protocol':'fixed_build_support_amendment_v2','guard_protocol_sha256':sha(ROOT/'contract/fixed_build_support_amendment_v2.json'),
  'source_protocol_sha256':sha(OUT/'qualification_protocol.json'),'baseline_float_input_sha256':{str(p):inputs[str(p)]['sha256'] for p in paths if 'baseline4/frames' in str(p)},
  'no_baseline_row_removed':True,'no_fit_or_material_created':True,'QUERY_not_conditioned_on_masks_or_RGB':True}
 write(OUT/'qualification_manifest.json',manifest)
 result={'status':'PASS_EXISTING_SURFACE_QUALIFICATION_ENGINEERING_NO_FIT','joint_forwards':forward_count,'dummy_feature_backwards':backward_count,'model_updates':0,'colour_updates':0,'optimizer_steps':0,
  'safe_rows':manifest['safe_rows'],'branch_rows':manifest['branch_rows'],'coverage':coverage,'rows':records,'source_files_and_model_binding_unchanged':True,
  'selection_uses_only_inside_and_guard_columns':True,'all_other_probe_columns_are_diagnostic_only':True,'manifest_sha256':sha(OUT/'qualification_manifest.json'),
  'max_reserved_cuda_bytes':int(torch.cuda.max_memory_reserved()),'seconds':time.time()-started,'automatic_fit':False,'automatic_all49':False,'shutdown':False,'reset_card':False}
 write(OUT/'receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['coverage','rows']}),flush=True)
if __name__=='__main__':main()

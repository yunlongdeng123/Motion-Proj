"""All49 BUILD visibility qualification for C shared-parent RGB, no optimizer."""
from pathlib import Path
import hashlib,json,os,sys,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_v44_a5_c49_rgb_20261107_r1';OLD=B/'scene008_v44_a5_road_fusion_integration_20261107_r1';SEM=B/'scene008_v44_full5_integration_20261107_r1';REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
OUT=R/'parent_visibility49';PILOT=[94,105,128,137]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident(p):
 p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 import numpy as np,torch
 from PIL import Image
 R.mkdir(exist_ok=True);(R/'contract').mkdir(exist_ok=True);assert not OUT.exists();OUT.mkdir();(OUT/'point_mass').mkdir();warm=R/'C_warm0_pilot';warm.mkdir();(warm/'frames').mkdir()
 sys.path[:0]=[str(REPO),str(B/'scene008_v44_rgb_asset_20261107_r1/source')]
 from v44_runtime import V44Runtime
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration,C_MANIFEST
 manifest=json.loads(C_MANIFEST.read_text());assetpath=C_MANIFEST.parent/manifest['new_road']['file'];assert sha(assetpath)=='3afc6e25b7b152f851b4eb0de4a73e6d219c9526d75453d99097a31ddd5774c5'
 with np.load(assetpath,allow_pickle=False) as z:parentids=z['child_to_parent_ids'].astype(np.int64);colours=z['render_colours_srgb'].copy()
 assert np.array_equal(parentids,np.repeat(np.arange(10098),4));assert colours.shape==(40392,3)
 parent_colours=colours.reshape(10098,4,3);uniform=bool(np.array_equal(parent_colours,np.repeat(parent_colours[:,:1],4,axis=1)))
 source_manifest=B/'scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1/target_manifest.json';assert sha(source_manifest)=='d7cca4167e2bbb68347f63babae356320a1b0ddd60e580bf746e966f2ac8acbe'
 registered={x['raw']:x for x in json.loads(source_manifest.read_text())['rows']};assert set(registered)==set(range(89,138))
 targetroot=B/'scene008_road_fusion_bake_20261107_r1/targets_edgeblend_v1';targets={};target_meta=[]
 paths=[Path(__file__),C_MANIFEST,assetpath,source_manifest,OLD/'contract/full5_transform_matrix.txt']
 for raw in range(89,138):
  p=targetroot/registered[raw]['file'];assert sha(p)==registered[raw]['sha256'];paths.append(p)
  with np.load(p,allow_pickle=False) as z:mask=z['fit_mask'].astype(bool);rgb=z['edgeblend_rgb'];protect=z['protect_mask'].astype(bool)
  assert mask.shape==(320,480) and rgb.shape==(320,480,3) and np.isfinite(rgb).all() and not (mask&protect).any()
  targets[raw]=mask;target_meta.append({'raw':raw,'path':str(p),'sha256':sha(p),'fit_pixels':int(mask.sum()),'original_role':registered[raw]['role'],'current_role':'BUILD_all49'})
 inputs={str(p):ident(p) for p in paths}
 write(OUT/'qualification_protocol.json',{'status':'SEALED_BEFORE_C49_VISIBLE_MASS_QUALIFICATION','raw_ids':list(range(89,138)),'states':['keep','delete'],'source_C_asset_sha256':sha(assetpath),
  'parent_rows':10098,'child_rows':40392,'eligible_predicate':'sum all49 KEEP+DELETE legal-fit-mask joint child alphaT mass, parent-shared derivative, strictly >0',
  'zero_mass_parent_policy':'fixed, no optimizer parameter','parent_RGB_init':'existingC400 warm source, not zero-trained RGB','same_colour_within_each_parent':uniform,
  'targets':target_meta,'inputs':inputs,'old_HELD9_current_role':'BUILD','planned_native_forwards':98,'planned_dummy_backwards':98,'optimizer_updates':0,'automatic_formal_release':False,'GPU_power':'KEEP_ON'})
 torch.set_num_threads(1);assert torch.cuda.is_available();start=time.time()
 base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json')
 core=SemanticActor5V44Integration(base,semantic_sidecar_path=SEM/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=OLD/'contract/full5_transform_matrix.txt')
 rt=RoadFusionV44Integration(core,C_MANIFEST);mapping=torch.as_tensor(parentids,device='cuda');aggregate=np.zeros(10098,np.float64);rows=[];calls=0;backs=0;warmrows=[]
 for raw in range(89,138):
  mask=targets[raw]
  if raw in PILOT:(warm/f'frames/{raw:03d}').mkdir()
  states={}
  for state in ['keep','delete']:
   scene=rt.scene(raw,delete=state=='delete');kw={k:v.detach() if torch.is_tensor(v) else v for k,v in scene['rows'].items()};offset=scene['meta']['latest_semantic_prefix_rows'];assert len(kw['mask'])==offset+40392
   assert not bool((kw['features'][:,16]!=0).any())
   probe=torch.ones((10098,1),device='cuda',dtype=kw['features'].dtype,requires_grad=True)
   extra=torch.cat([probe.new_zeros((offset,1)),probe.index_select(0,mapping)],0);kw['features']=torch.cat([kw['features'][:,:16],extra,kw['features'][:,17:]],1)
   assert all(not v.requires_grad for k,v in kw.items() if torch.is_tensor(v) and k!='features')
   out=base.raster(scene['cam'],base.pipe,0,kw,scene['bg']);calls+=1;write(OUT/'kernel_progress.json',{'raw':raw,'state':state,'actual_native_forwards':calls,'actual_dummy_backwards':backs,'optimizer_updates':0})
   view=out['extra'][14];ones_error=float((view.detach()-out['extra'][7].detach()).abs().max());assert ones_error==0
   loss=view[torch.as_tensor(mask,device='cuda')].sum();mass=torch.autograd.grad(loss,probe)[0].detach().cpu().numpy()[:,0];backs+=1
   write(OUT/'kernel_progress.json',{'raw':raw,'state':state,'actual_native_forwards':calls,'actual_dummy_backwards':backs,'optimizer_updates':0})
   assert np.isfinite(mass).all() and mass.min()>=0;expected=float(view.detach().cpu().numpy()[mask].sum(dtype=np.float64));actual=float(mass.sum(dtype=np.float64));absolute=abs(expected-actual);relative=absolute/max(abs(expected),1e-30)
   assert relative<=1e-5 if expected else actual==0
   aggregate+=mass.astype(np.float64);file=OUT/f'point_mass/{raw:03d}_{state}.npz';np.savez_compressed(file,parent_mass=mass)
   item={'raw':raw,'state':state,'fit_pixels':int(mask.sum()),'positive_parents':int((mass>0).sum()),'mass_sum':actual,'mass_absolute_error':absolute,'mass_relative_error':relative,'ones_readout_error':ones_error,'file':str(file.relative_to(OUT)),'sha256':sha(file)}
   rows.append(item)
   if raw in PILOT:
    alpha=out['alpha'].detach().cpu().numpy()[0];depthsum=out['depth_sum'].detach().cpu().numpy()[0];rgb=(out['rgb']+(1-out['alpha'])*scene['env'].detach()).detach().cpu().numpy();all_extra=out['extra'].detach().cpu().numpy()
    values={'rgb':rgb,'alpha':alpha,'depth_sum':depthsum,'depth_m':depthsum/np.maximum(alpha,1e-8)/core.scale,'group_alpha_T':all_extra[:8],
     'added_owner_rgb_T':np.stack([all_extra[8:11],np.zeros_like(all_extra[8:11]),np.zeros_like(all_extra[8:11])]),'newroad_rgb_T':all_extra[11:14]}
    parity={}
    with np.load(OLD/f'C4_retry1/frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:
     for name,value in values.items():parity[name]=float(np.abs(value-z[name]).max());assert np.array_equal(value,z[name]),(raw,state,name,parity[name])
    folder=warm/f'frames/{raw:03d}';np.savez_compressed(folder/(state+'_float.npz'),**values);Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(folder/(state+'.png'))
    states[state]={'frozen_original_C4_max_abs':parity,'original_C_geometry_and_material_exact':True}
   del scene,kw,probe,extra,out,loss,view
  if raw in PILOT:warmrows.append({'raw':raw,'states':states})
  write(OUT/'progress.json',{'completed_raws':raw-88,'actual_native_forwards':calls,'actual_dummy_backwards':backs,'currently_positive_parents':int((aggregate>0).sum())});print(json.dumps({'raw':raw,'calls':calls,'backs':backs,'currently_positive_parents':int((aggregate>0).sum())}),flush=True)
  rt.assert_unchanged();base._data.clear();torch.cuda.empty_cache()
 assert calls==backs==98;eligible=np.flatnonzero(aggregate>0).astype(np.int64)
 selection=OUT/'qualified_parent_rows.npz';np.savez_compressed(selection,parent_mass=aggregate,eligible_parent_ids=eligible,initial_parent_RGB=parent_colours[:,0],child_to_parent_ids=parentids)
 rt.assert_unchanged(full=True)
 for p,meta in inputs.items():assert ident(p)==meta,p
 manifest={'schema':'C49-parent-visible-mass-qualification-v1','raw_ids':list(range(89,138)),'states':['keep','delete'],'parent_rows':10098,'child_rows':40392,'source_C_asset_sha256':sha(assetpath),
  'selection_file':selection.name,'selection_sha256':sha(selection),'eligible_parents':len(eligible),'eligibility':'aggregate_parent_mass>0','zero_mass_frozen':True,'same_colour_within_each_parent':uniform,
  'targets':target_meta,'model_binding':rt.receipt(),'original_BUILD40_eligible_count':2053,'current_all49_BUILD_eligible_count':len(eligible),'old_HELD9_now_BUILD':True,'optimizer_updates':0,'colour_updates':0}
 write(OUT/'qualification_manifest.json',manifest)
 write(warm/'receipt.json',{'status':'PASS_CWARM0_FOUR_FRAME_EXACT_OLD_C_SOURCE_FROM_QUALIFICATION','frames':warmrows,'native_calls_reused_from_qualification':8,'additional_native_forwards':0,'optimizer_updates':0,'initializer_C_asset_sha256':sha(assetpath),'source_initialization':'previously-trained C400 warm material'})
 result={'status':'PASS_C49_PARENT_VISIBILITY_QUALIFICATION_NO_OPTIMIZER','actual_native_forwards':calls,'actual_dummy_backwards':backs,'optimizer_updates':0,'eligible_parents':len(eligible),'old_BUILD40_eligible':2053,
  'same_colour_within_each_parent':uniform,'qualification_manifest_sha256':sha(OUT/'qualification_manifest.json'),'seconds':time.time()-start,'max_reserved_cuda_bytes':int(torch.cuda.max_memory_reserved()),'automatic_formal_release':False,'shutdown':False,'reset_card':False}
 write(OUT/'receipt.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()

"""Registered H4 only; every prepass/probe/public kernel counted separately."""
from pathlib import Path
import hashlib,json,os,sys,time,traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_v44_a5_road_fusion_integration_20261107_r1';REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');S=B/'scene008_v44_full5_integration_20261107_r1';OUT=R/'H4_model_foreground_culling_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 import numpy as np,torch
 from PIL import Image
 source=R/'source/foreground_road_culling_H4_v2.py';contract=R/'contract/foreground_road_culling_H4_v2.json'
 assert sha(source)=='9eb7fb2f86af8a276d7b1503f119a6551d566b975b430f865af234aa55daabc8' and sha(contract)=='5621868aaf6479fc385b77d0d74f1d8b140ae77d5f69c3d6154609263dfb4532'
 assert not OUT.exists();OUT.mkdir();(OUT/'frames').mkdir();start=time.time();torch.set_num_threads(1)
 sys.path[:0]=[str(R/'source'),str(REPO),str(B/'scene008_v44_rgb_asset_20261107_r1/source')]
 from v44_runtime import V44Runtime
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration,C_MANIFEST
 from foreground_road_culling_H4_v2 import ForegroundRoadCulling
 base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json');core=SemanticActor5V44Integration(base,semantic_sidecar_path=S/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=R/'contract/full5_transform_matrix.txt');road=RoadFusionV44Integration(core,C_MANIFEST);rt=ForegroundRoadCulling(road,max_iterations=64)
 prior=R/'H4_model_foreground_culling';cached={};cache_proof=[]
 assert sha(R/'source/foreground_road_culling_H4.py')=='1632d86d0d1d812bcda3d42a3b2752b29f4b73fd700dafce7aec0e9c8153a532'
 old_progress=json.loads((prior/'progress.json').read_text())
 for raw,state in [(128,'delete'),(94,'keep'),(94,'delete')]:
  matching=[r for r in old_progress['completed'] if r['raw']==raw and r['state']==state];assert len(matching)==1
  record=matching[0]['meta']['foreground_road_culling'];assert record['converged'] and record['raw']==raw and record['delete']==(state=='delete')
  maskfile=prior/f'frames/{raw:03d}/{state}_visibility_mask.npz'
  with np.load(maskfile,allow_pickle=False) as z:mask=z['selected_road_child_rows'].copy();guard=z['model_foreground_guard'].copy()
  assert mask.dtype==np.bool_ and mask.shape==(40392,) and guard.dtype==np.bool_ and guard.shape==(320,480) and int(mask.sum())==record['active_rows']
  rt.cache[(raw,state=='delete')]={'mask':torch.as_tensor(mask,device='cuda'),'guard':torch.as_tensor(guard,device='cuda'),'baseline':None,'record':record}
  cached[(raw,state)]=True;cache_proof.append({'raw':raw,'state':state,'source_mask':str(maskfile),'source_mask_sha256':sha(maskfile),'source_progress_sha256':sha(prior/'progress.json'),'original_iterations':len(record['iterations']),'active_rows':int(mask.sum()),'baseline_and_probe_reused_without_new_backward':True})
 write(OUT/'cached_visibility_reuse.json',{'status':'SEALED_H1_THREE_CONVERGED_CACHE_BINDING_BEFORE_H2_GPU','source_H1_contract_sha256':sha(R/'contract/foreground_road_culling_H4_v1.json'),'source_H2_contract_sha256':sha(contract),'frozen_current_runtime':road.receipt(),'rows':cache_proof})
 calls=0;original=base.raster
 def counted(*a,**kw):
  nonlocal calls
  out=original(*a,**kw);calls+=1;write(OUT/'kernel_progress.json',{'actual_native_forwards':calls,'actual_dummy_backwards_as_reported_by_H':rt.actual_dummy_backwards,'actual_public_forwards_as_reported_by_H':rt.actual_public_forwards,'optimizer_updates':0});return out
 base.raster=counted;records=[]
 order=[(128,'delete'),(94,'keep'),(94,'delete'),(105,'keep'),(105,'delete'),(128,'keep'),(137,'keep'),(137,'delete')]
 try:
  for raw,state in order:
   # Deliberately no enclosing no_grad: H internally probes native feature mass.
   out=rt.render_scene(raw,delete=state=='delete');folder=OUT/f'frames/{raw:03d}';folder.mkdir(exist_ok=True)
   rgb=out['rgb'].detach().cpu().numpy();alpha=out['alpha'].detach().cpu().numpy()[0];depthsum=out['depth_sum'].detach().cpu().numpy()[0];groups=out['group_alpha_T'].detach().cpu().numpy()
   assert groups.shape==(8,320,480) and np.isfinite(rgb).all() and np.isfinite(groups).all()
   np.savez_compressed(folder/(state+'_float.npz'),rgb=rgb,alpha=alpha,depth_sum=depthsum,depth_m=depthsum/np.maximum(alpha,1e-8)/core.scale,group_alpha_T=groups,
    added_owner_rgb_T=out['added_owner_rgb_T'].detach().cpu().numpy(),newroad_rgb_T=out['newroad_rgb_T'].detach().cpu().numpy())
   Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(folder/(state+'.png'))
   if (raw,state) in cached:
    with np.load(prior/f'frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:
     current={'rgb':rgb,'alpha':alpha,'depth_sum':depthsum,'depth_m':depthsum/np.maximum(alpha,1e-8)/core.scale,'group_alpha_T':groups,'added_owner_rgb_T':out['added_owner_rgb_T'].detach().cpu().numpy(),'newroad_rgb_T':out['newroad_rgb_T'].detach().cpu().numpy()}
     for name,value in current.items():assert np.array_equal(value,z[name]),(raw,state,name,float(np.abs(value-z[name]).max()))
   q=rt.cache[(raw,state=='delete')];np.savez_compressed(folder/(state+'_visibility_mask.npz'),selected_road_child_rows=q['mask'].detach().cpu().numpy(),model_foreground_guard=q['guard'].detach().cpu().numpy())
   record={'raw':raw,'state':state,'active_road_children':q['record']['active_rows'],'converged':q['record']['converged'],'iteration_records':q['record']['iterations'],'public_guard_road_mass':out['meta']['public_foreground_road_alpha_mass'],'meta':out['meta']}
   records.append(record);write(OUT/'progress.json',{'completed':records,'actual_native_forwards':calls,'actual_dummy_backwards':rt.actual_dummy_backwards,'actual_public_forwards':rt.actual_public_forwards});print(json.dumps({'raw':raw,'state':state,'active_children':record['active_road_children'],'iterations':len(record['iteration_records']),'actual_native_forwards':calls,'actual_dummy_backwards':rt.actual_dummy_backwards}),flush=True)
   del out;torch.cuda.empty_cache()
  receipt=rt.receipt();assert receipt['actual_native_forwards']==calls and receipt['actual_public_forwards']==8
  write(OUT/'receipt.json',{'status':'COMPLETE_H2_CAP64_REGISTERED_CULLING_PENDING_QUALITY','actual_native_forwards':calls,'actual_dummy_backwards':rt.actual_dummy_backwards,'actual_public_forwards':8,'optimizer_updates':0,'source_sha256':sha(source),'contract_sha256':sha(contract),'records':records,'replayed_cached_three_cases_exact':cache_proof,'runtime_receipt':receipt,'seconds':time.time()-start,'max_reserved_cuda_bytes':int(torch.cuda.max_memory_reserved()),'automatic49':False,'quality_accepted':False,'shutdown':False})
 except Exception:
  write(OUT/'failure.json',{'status':'FAILED_PRESERVED_H_ENGINEERING_OR_CONVERGENCE','actual_native_forwards':calls,'actual_dummy_backwards':rt.actual_dummy_backwards,'actual_public_forwards':rt.actual_public_forwards,'optimizer_updates':0,'traceback':traceback.format_exc(),'completed':records,'runtime_receipt':rt.receipt(),'reused_cache':cache_proof});raise
 finally:base.raster=original
if __name__=='__main__':main()

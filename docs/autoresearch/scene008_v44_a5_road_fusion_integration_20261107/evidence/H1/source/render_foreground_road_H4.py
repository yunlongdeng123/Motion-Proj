"""Registered H4 only; every prepass/probe/public kernel counted separately."""
from pathlib import Path
import hashlib,json,os,sys,time,traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_v44_a5_road_fusion_integration_20261107_r1';REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');S=B/'scene008_v44_full5_integration_20261107_r1';OUT=R/'H4_model_foreground_culling'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 import numpy as np,torch
 from PIL import Image
 source=R/'source/foreground_road_culling_H4.py';contract=R/'contract/foreground_road_culling_H4_v1.json'
 assert sha(source)=='1632d86d0d1d812bcda3d42a3b2752b29f4b73fd700dafce7aec0e9c8153a532' and sha(contract)=='01fbd50470830ad654c9494c7d90ec9e05b074f565e9207df02470326f03638c'
 assert not OUT.exists();OUT.mkdir();(OUT/'frames').mkdir();start=time.time();torch.set_num_threads(1)
 sys.path[:0]=[str(R/'source'),str(REPO),str(B/'scene008_v44_rgb_asset_20261107_r1/source')]
 from v44_runtime import V44Runtime
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration,C_MANIFEST
 from foreground_road_culling_H4 import ForegroundRoadCulling
 base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json');core=SemanticActor5V44Integration(base,semantic_sidecar_path=S/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=R/'contract/full5_transform_matrix.txt');road=RoadFusionV44Integration(core,C_MANIFEST);rt=ForegroundRoadCulling(road)
 calls=0;original=base.raster
 def counted(*a,**kw):
  nonlocal calls
  out=original(*a,**kw);calls+=1;write(OUT/'kernel_progress.json',{'actual_native_forwards':calls,'actual_dummy_backwards_as_reported_by_H':rt.actual_dummy_backwards,'actual_public_forwards_as_reported_by_H':rt.actual_public_forwards,'optimizer_updates':0});return out
 base.raster=counted;records=[]
 order=[(128,'delete')]+[(raw,state) for raw in [94,105,128,137] for state in ['keep','delete'] if (raw,state)!=(128,'delete')]
 try:
  for raw,state in order:
   # Deliberately no enclosing no_grad: H internally probes native feature mass.
   out=rt.render_scene(raw,delete=state=='delete');folder=OUT/f'frames/{raw:03d}';folder.mkdir(exist_ok=True)
   rgb=out['rgb'].detach().cpu().numpy();alpha=out['alpha'].detach().cpu().numpy()[0];depthsum=out['depth_sum'].detach().cpu().numpy()[0];groups=out['group_alpha_T'].detach().cpu().numpy()
   assert groups.shape==(8,320,480) and np.isfinite(rgb).all() and np.isfinite(groups).all()
   np.savez_compressed(folder/(state+'_float.npz'),rgb=rgb,alpha=alpha,depth_sum=depthsum,depth_m=depthsum/np.maximum(alpha,1e-8)/core.scale,group_alpha_T=groups,
    added_owner_rgb_T=out['added_owner_rgb_T'].detach().cpu().numpy(),newroad_rgb_T=out['newroad_rgb_T'].detach().cpu().numpy())
   Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(folder/(state+'.png'))
   q=rt.cache[(raw,state=='delete')];np.savez_compressed(folder/(state+'_visibility_mask.npz'),selected_road_child_rows=q['mask'].detach().cpu().numpy(),model_foreground_guard=q['guard'].detach().cpu().numpy())
   record={'raw':raw,'state':state,'active_road_children':q['record']['active_rows'],'converged':q['record']['converged'],'iteration_records':q['record']['iterations'],'public_guard_road_mass':out['meta']['public_foreground_road_alpha_mass'],'meta':out['meta']}
   records.append(record);write(OUT/'progress.json',{'completed':records,'actual_native_forwards':calls,'actual_dummy_backwards':rt.actual_dummy_backwards,'actual_public_forwards':rt.actual_public_forwards});print(json.dumps({'raw':raw,'state':state,'active_children':record['active_road_children'],'iterations':len(record['iteration_records']),'actual_native_forwards':calls,'actual_dummy_backwards':rt.actual_dummy_backwards}),flush=True)
   del out;torch.cuda.empty_cache()
  receipt=rt.receipt();assert receipt['actual_native_forwards']==calls and receipt['actual_public_forwards']==8
  write(OUT/'receipt.json',{'status':'COMPLETE_H4_REGISTERED_CULLING_PENDING_QUALITY','actual_native_forwards':calls,'actual_dummy_backwards':rt.actual_dummy_backwards,'actual_public_forwards':8,'optimizer_updates':0,'source_sha256':sha(source),'contract_sha256':sha(contract),'records':records,'runtime_receipt':receipt,'seconds':time.time()-start,'max_reserved_cuda_bytes':int(torch.cuda.max_memory_reserved()),'automatic49':False,'quality_accepted':False,'shutdown':False})
 except Exception:
  write(OUT/'failure.json',{'status':'FAILED_PRESERVED_H_ENGINEERING_OR_CONVERGENCE','actual_native_forwards':calls,'actual_dummy_backwards':rt.actual_dummy_backwards,'actual_public_forwards':rt.actual_public_forwards,'optimizer_updates':0,'traceback':traceback.format_exc(),'completed':records});raise
 finally:base.raster=original
if __name__=='__main__':main()

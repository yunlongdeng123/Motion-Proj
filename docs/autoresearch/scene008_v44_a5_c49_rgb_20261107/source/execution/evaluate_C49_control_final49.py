"""Known-imperfect background control: one runtime, Cwarm0 versus formal2K."""
from pathlib import Path
import hashlib,json,os,sys,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_v44_a5_c49_rgb_20261107_r1';OLD=B/'scene008_v44_a5_road_fusion_integration_20261107_r1';SEM=B/'scene008_v44_full5_integration_20261107_r1';REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 import numpy as np,torch
 from PIL import Image
 sys.path[:0]=[str(R/'source'),str(REPO),str(B/'scene008_v44_rgb_asset_20261107_r1/source')]
 from v44_runtime import V44Runtime
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration,C_MANIFEST
 from road_fusion_material_runtime import RoadFusionMaterialRuntime
 result=json.loads((R/'formal2000_r1/returned.json').read_text());assert result['final_step']==2000
 cp=R/'formal2000_r1/state2000.pth';state=torch.load(cp,map_location='cpu',weights_only=False);assert state['step']==2000
 qm=R/'parent_visibility49/qualification_manifest.json';q=json.loads(qm.read_text());assert state['config']['qualification_manifest_sha256']==sha(qm)
 with np.load(qm.parent/q['selection_file'],allow_pickle=False) as z:ids=z['eligible_parent_ids'].copy()
 outdir=R/'control_final49_2000';outdir.mkdir(exist_ok=False)
 for name in ['Cwarm0','material2000']:(outdir/name/'frames').mkdir(parents=True)
 protocol={'status':'SEALED_BEFORE_CONTROL49_NATIVE_FORWARD','purpose':'Known-imperfect C background material control, not default adoption and not H49','raw_ids':list(range(89,138)),'states':['keep','delete'],
  'same_single_loaded_component_runtime_for_two_material_tables':True,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'Cwarm0_asset_sha256':'3afc6e25b7b152f851b4eb0de4a73e6d219c9526d75453d99097a31ddd5774c5',
  'planned_actual_native_forwards':196,'backwards':0,'optimizer_updates':0,'save':'both PNG sets; material2000 full float arrays; all frozen geometry/identity fields checked against warm0 per state',
  'all49_are_BUILD':True,'quality_accepted':False,'H2_49_authorized':False,'shutdown':False}
 write(R/'contract/control_final49_2000_protocol_v1.json',protocol)
 torch.set_num_threads(1);start=time.time();base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json');core=SemanticActor5V44Integration(base,semantic_sidecar_path=SEM/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=OLD/'contract/full5_transform_matrix.txt');road=RoadFusionV44Integration(core,C_MANIFEST);rt=RoadFusionMaterialRuntime(road)
 idx=torch.as_tensor(ids,device='cuda');learned=rt.initial_parent_rgb.index_copy(0,idx,state['parent_RGB'].cuda()).detach();assert torch.isfinite(learned).all()
 calls=0;records=[]
 def render(raw,s,table):
  nonlocal calls
  with torch.no_grad():out=rt.render(raw,rt.scene(raw,delete=s=='delete',parent_rgb=table))
  calls+=1;write(outdir/'kernel_progress.json',{'actual_native_forwards':calls,'backwards':0,'optimizer_updates':0})
  a=out['alpha'].detach().cpu().numpy()[0];ds=out['depth_sum'].detach().cpu().numpy()[0]
  values={'rgb':out['rgb'].detach().cpu().numpy(),'alpha':a,'depth_sum':ds,'depth_m':ds/np.maximum(a,1e-8)/core.scale,'group_alpha_T':out['group_alpha_T'].detach().cpu().numpy(),'added_owner_rgb_T':out['added_owner_rgb_T'].detach().cpu().numpy(),'newroad_rgb_T':out['newroad_rgb_T'].detach().cpu().numpy()}
  assert all(np.isfinite(v).all() for v in values.values());return values
 for raw in range(89,138):
  for name in ['Cwarm0','material2000']:(outdir/name/f'frames/{raw:03d}').mkdir()
  for s in ['keep','delete']:
   warm=render(raw,s,rt.initial_parent_rgb);current=render(raw,s,learned);errors={}
   for key in ['alpha','depth_sum','depth_m','group_alpha_T','added_owner_rgb_T']:
    errors[key]=float(np.abs(warm[key]-current[key]).max());assert np.array_equal(warm[key],current[key]),(raw,s,key,errors[key])
   for name,value in [('Cwarm0',warm),('material2000',current)]:
    f=outdir/name/f'frames/{raw:03d}';Image.fromarray(np.uint8(np.clip(value['rgb'].transpose(1,2,0),0,1)*255+.5)).save(f/(s+'.png'))
    if name=='material2000':np.savez_compressed(f/(s+'_float.npz'),**value)
   change=np.abs(current['rgb']-warm['rgb']);records.append({'raw':raw,'state':s,'frozen_global_geometry_identity_max_abs':errors,'RGB_mean_absolute_change':float(change.mean(dtype=np.float64)),'RGB_max_absolute_change':float(change.max())})
  write(outdir/'progress.json',{'completed_raw':raw,'actual_native_forwards':calls,'records':records});print(json.dumps({'raw':raw,'actual_native_forwards':calls,'global_geometry_and_identity_exact':True}),flush=True);base._data.clear();torch.cuda.empty_cache()
 rt.assert_unchanged(full=True);assert calls==196 and rt.render_calls==196
 receipt={'status':'COMPLETE_KNOWN_IMPERFECT_C_CONTROL49_CWARM0_VS_2000_GEOMETRY_EXACT_PENDING_METRICS','actual_native_forwards':196,'actual_backwards':0,'optimizer_updates':0,'formal_optimizer_updates_represented':2000,'four_scene_components_and_road_geometry_unchanged':True,
  'records':records,'runtime_receipt':rt.receipt(),'seconds':time.time()-start,'max_reserved_cuda_bytes':int(torch.cuda.max_memory_reserved()),'quality_accepted':False,'default_adoption':False,'H49_executed':False,'automatic_further_training':False,'shutdown':False}
 write(outdir/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ['records','runtime_receipt']}),flush=True)
if __name__=='__main__':main()

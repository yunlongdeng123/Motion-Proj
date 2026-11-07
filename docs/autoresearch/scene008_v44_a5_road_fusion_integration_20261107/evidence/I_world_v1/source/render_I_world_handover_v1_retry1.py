"""New RGB-only C asset plus the eight already-qualified H2 masks; 8F/0B."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_v44_a5_c49_rgb_20261107_r1';OLD=B/'scene008_v44_a5_road_fusion_integration_20261107_r1';SEM=B/'scene008_v44_full5_integration_20261107_r1';REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene-manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--step',type=int,choices=[2000],required=True);a=p.parse_args()
 import numpy as np,torch
 from PIL import Image
 sys.path[:0]=[str(OLD/'source'),str(REPO),str(B/'scene008_v44_rgb_asset_20261107_r1/source')]
 from v44_runtime import V44Runtime
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration,C_MANIFEST
 from foreground_road_culling_H4_v2 import ForegroundRoadCulling
 assert sha(OLD/'source/foreground_road_culling_H4_v2.py')=='9eb7fb2f86af8a276d7b1503f119a6551d566b975b430f865af234aa55daabc8'
 old_manifest=json.loads(C_MANIFEST.read_text());new_manifest=json.loads(a.scene_manifest.read_text());oldasset=C_MANIFEST.parent/old_manifest['new_road']['file'];newasset=a.scene_manifest.parent/new_manifest['new_road']['file']
 assert sha(oldasset)=='3afc6e25b7b152f851b4eb0de4a73e6d219c9526d75453d99097a31ddd5774c5' and sha(newasset)==new_manifest['new_road']['sha256']
 with np.load(oldasset,allow_pickle=False) as z:before={k:z[k].copy() for k in z.files}
 with np.load(newasset,allow_pickle=False) as z:after={k:z[k].copy() for k in z.files}
 assert set(before)==set(after);allowed={'parent_render_colours_srgb','render_colours_srgb'}
 for key in before:
  if key not in allowed:assert before[key].dtype==after[key].dtype and before[key].shape==after[key].shape and before[key].tobytes()==after[key].tobytes(),key
 state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert state['step']==a.step
 q=R/'parent_visibility49/qualification_manifest.json';qm=json.loads(q.read_text());assert state['config']['qualification_manifest_sha256']==sha(q)
 with np.load(q.parent/qm['selection_file'],allow_pickle=False) as z:ids=z['eligible_parent_ids'].copy()
 expected=before['parent_render_colours_srgb'].copy();expected[ids]=state['parent_RGB'].numpy();assert np.array_equal(expected,after['parent_render_colours_srgb']) and np.array_equal(expected[before['child_to_parent_ids']],after['render_colours_srgb'])
 outdir=OLD/'I_world_handover_v1/pilot128_137_retry1';outdir.mkdir(exist_ok=False);(outdir/'frames').mkdir();cache_source=OLD/'H4_model_foreground_culling_v2';old_receipt=json.loads((cache_source/'receipt.json').read_text());cache=[]
 for record in old_receipt['records']:
  raw=record['raw'];s=record['state'];path=cache_source/f'frames/{raw:03d}/{s}_visibility_mask.npz';cache.append({'raw':raw,'state':s,'file':str(path),'sha256':sha(path),'record':record['meta']['foreground_road_culling']})
 assert len(cache)==8
 method={'schema':'C49_NODE_MATERIAL_ON_FIXED_H2_VISIBILITY_V1','status':'SEALED_BEFORE_NEW_MATERIAL_PUBLIC_GPU','formal_step':a.step,'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha(a.checkpoint),
  'new_scene_manifest':str(a.scene_manifest),'new_scene_manifest_sha256':sha(a.scene_manifest),'new_C_asset_sha256':sha(newasset),'Cwarm_source_sha256':sha(oldasset),'all_non_RGB_payloads_byte_exact':True,
  'old_H2_policy_source_sha256':sha(OLD/'source/foreground_road_culling_H4_v2.py'),'old_H2_contract_sha256':sha(OLD/'contract/foreground_road_culling_H4_v2.json'),'new_method_does_not_reuse_old_asset_SHA_contract':True,
  'cached_visibility':cache,'cache_validity':'Only RGB changes; positions, scales, opacity, masks, cameras, parent mapping and frozen foreground scene unchanged. Guard/child mass is RGB independent.',
  'planned_new_native_forwards':4,'planned_new_dummy_backwards':0,'optimizer_updates':0,'old_H2_113F100B_not_repeated':True,'quality_accepted':False,'automatic49':False}
 contract=OLD/'contract/I_world_handover_v1_execution_retry1.json';assert not contract.exists();write(contract,method)
 torch.set_num_threads(1);start=time.time();base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json');core=SemanticActor5V44Integration(base,semantic_sidecar_path=SEM/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=OLD/'contract/full5_transform_matrix.txt');road=RoadFusionV44Integration(core,a.scene_manifest);rt=ForegroundRoadCulling(road,max_iterations=64)
 for item in cache:
  raw=item['raw'];s=item['state'];assert sha(item['file'])==item['sha256']
  with np.load(item['file'],allow_pickle=False) as z:mask=z['selected_road_child_rows'].copy();guard=z['model_foreground_guard'].copy()
  assert mask.dtype==np.bool_ and mask.shape==(40392,) and int(mask.sum())==item['record']['active_rows'] and item['record']['converged']
  rt.cache[(raw,s=='delete')]={'mask':torch.as_tensor(mask,device='cuda'),'guard':torch.as_tensor(guard,device='cuda'),'baseline':None,'record':item['record']}
 from world_boundary_road_coverage import WorldBoundaryRoadCoverage
 source_I=OLD/'source/world_boundary_road_coverage.py';primary=OLD/'contract/I_world_handover_v1.json';proof=OLD/'I_world_handover_v1/extent_and_gain_proof.json'
 assert sha(source_I)=='1d2c12393697c4f8dc011395fd05452dc0182cc5e9b9afa727f34a41efb695b3'
 assert sha(primary)=='065cafb3005b8c25cb18e755309131cdedb45e9bce54e0032f36ff9f38e586ad'
 h2=rt;rt=WorldBoundaryRoadCoverage(h2,proof)
 method.update(schema='I_WORLD_HANDOVER_FIRST_TWO_RAW_EXECUTION_V1',source_I_sha256=sha(source_I),primary_I_contract_sha256=sha(primary),proof_sha256=sha(proof),pilot_raw_ids=[128,137],planned_new_native_forwards=4,planned_new_dummy_backwards=0,original_H2_method_only_visibility_reused=True)
 write(contract,method)
 rows=[]
 for raw in [128,137]:
  folder=outdir/f'frames/{raw:03d}';folder.mkdir()
  for s in ['keep','delete']:
   before=h2.scene(raw,delete=s=='delete');scene=rt.scene(raw,delete=s=='delete');oldrows=before['rows'];newrows=scene['rows'];offset=scene['meta']['latest_semantic_prefix_rows']
   for key,value in oldrows.items():
    other=newrows[key]
    if value is None:assert other is None;continue
    if key=='opacities':
     assert torch.equal(value[:offset],other[:offset]) and torch.equal(value[offset:]*rt.gain[:,None],other[offset:])
    else:assert torch.equal(value,other),key
   assert torch.equal(before['env'],scene['env']) and torch.equal(before['bg'],scene['bg'])
   out=rt.render(raw,scene);write(outdir/'kernel_progress.json',{'actual_native_forwards':rt.render_calls,'actual_dummy_backwards':h2.actual_dummy_backwards,'optimizer_updates':0})
   rgb=out['rgb'].detach().cpu().numpy();alpha=out['alpha'].detach().cpu().numpy()[0];ds=out['depth_sum'].detach().cpu().numpy()[0];groups=out['group_alpha_T'].detach().cpu().numpy();added=out['added_owner_rgb_T'].detach().cpu().numpy()
   values={'rgb':rgb,'alpha':alpha,'depth_sum':ds,'depth_m':ds/np.maximum(alpha,1e-8)/core.scale,'group_alpha_T':groups,'added_owner_rgb_T':added,'newroad_rgb_T':out['newroad_rgb_T'].detach().cpu().numpy()};errors={}
   with np.load(cache_source/f'frames/{raw:03d}/{s}_float.npz',allow_pickle=False) as z:
    for key in ['alpha','depth_sum','depth_m','group_alpha_T','added_owner_rgb_T']:errors[key]=float(np.abs(values[key]-z[key]).max())
   np.savez_compressed(folder/(s+'_float.npz'),**values);Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(folder/(s+'.png'))
   rows.append({'raw':raw,'state':s,'reported_H2_material2000_output_differences_not_frozen_requirements':errors,'road_mass_on_original_model_foreground':out['meta']['public_foreground_road_alpha_mass'],'active_children':h2.cache[(raw,s=='delete')]['record']['active_rows']});print(json.dumps(rows[-1]),flush=True);del out,before,scene,oldrows,newrows
  torch.cuda.empty_cache()
 receipt=rt.receipt();assert receipt['actual_native_forwards']==4 and receipt['actual_dummy_backwards']==0 and h2.actual_dummy_backwards==0
 write(outdir/'receipt.json',{'status':'COMPLETE_I_WORLD_BOUNDARY_HANDOVER_FIRST4F_PENDING_QUALITY','step':a.step,'actual_native_forwards':4,'actual_dummy_backwards':0,'optimizer_updates':0,'only_effective_road_opacity_after_original_Mip_changed':True,'all_other_packet_fields_and_original_prefix_exact':True,'later94_105_and49_not_released':True,'method_contract_sha256':sha(contract),'records':rows,'runtime_receipt':receipt,'seconds':time.time()-start,'quality_accepted':False,'automatic49':False,'shutdown':False})
if __name__=='__main__':main()

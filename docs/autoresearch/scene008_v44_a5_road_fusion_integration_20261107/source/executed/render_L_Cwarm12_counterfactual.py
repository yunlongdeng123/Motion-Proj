"""Exactly eight fixed-H2 native forwards for the sealed twelve-parent rollback."""
from pathlib import Path
import hashlib,json,os,sys,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
R=B/'scene008_v44_a5_road_fusion_integration_20261107_r1'
SEM=B/'scene008_v44_full5_integration_20261107_r1'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
CONTRACT=R/'contract/L_Cwarm12_counterfactual_v1.json'
PARENTS=[12,13,14,83,84,85,153,154,155,224,225,226]
OUT=R/'J4_road_parent_operator/L_Cwarm12_render_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
 c=json.loads(CONTRACT.read_text())
 contract_sha256=sha(CONTRACT)
 assert c['status']=='SEALED_BEFORE_L_NATIVE8F' and c['fixed_parent_ids']==PARENTS
 assert c['native_render_source_sha256']==sha(__file__)
 assert c['planned_native_forwards']==8 and c['planned_dummy_backwards']==0 and c['new_solver_calls']==c['Adam_updates']==0
 manifest=Path(c['scene_manifest']);proofpath=Path(c['cpu_proof'])
 assert sha(manifest)==c['scene_manifest_sha256'] and sha(proofpath)==c['cpu_proof_sha256']
 proof=json.loads(proofpath.read_text())
 assert proof['status']=='PASS_CPU_EXACT_CWARM12_ROLLBACK_PENDING_NATIVE8F'
 assert proof['fixed_parent_ids']==PARENTS and proof['full_operator_106_by_12_all_exact_zero']
 assert proof['all_106_K_affine_predictions_byte_exact'] and proof['both_probe_newroad_RGB_T_equal_Cwarm']
 assert proof['only_parent_and_child_RGB_arrays_changed'] and proof['all_other_parent_RGB_byte_exact']
 import numpy as np,torch
 from PIL import Image
 sys.path[:0]=[str(R/'source'),str(REPO),str(B/'scene008_v44_rgb_asset_20261107_r1/source')]
 from v44_runtime import V44Runtime
 from foreground_road_culling_H4_v2 import ForegroundRoadCulling
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration
 assert sha(R/'source/foreground_road_culling_H4_v2.py')=='9eb7fb2f86af8a276d7b1503f119a6551d566b975b430f865af234aa55daabc8'
 for path,expected in proof['source_inputs_sha256'].items():assert sha(path)==expected
 m=json.loads(manifest.read_text());asset=manifest.parent/m['new_road']['file']
 assert sha(asset)==c['asset_sha256']==m['new_road']['sha256']
 OUT.mkdir(exist_ok=False);(OUT/'frames').mkdir();torch.set_num_threads(1)
 start=time.time();base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json')
 sem=SemanticActor5V44Integration(base,semantic_sidecar_path=SEM/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=R/'contract/full5_transform_matrix.txt')
 road=RoadFusionV44Integration(sem,manifest);h2=ForegroundRoadCulling(road,max_iterations=64)
 cache_source=R/'H4_model_foreground_culling_v2'
 for item in c['H2_cached_visibility']:
  assert sha(item['file'])==item['sha256'];record=item['record']
  with np.load(item['file'],allow_pickle=False) as z:mask=z['selected_road_child_rows'].copy();guard=z['model_foreground_guard'].copy()
  assert mask.shape==(40392,) and mask.dtype==np.bool_ and record['converged'] and int(mask.sum())==record['active_rows']
  h2.cache[(item['raw'],item['state']=='delete')]={'mask':torch.as_tensor(mask,device='cuda'),'guard':torch.as_tensor(guard,device='cuda'),'baseline':None,'record':record}
 assert len(h2.cache)==8
 rows=[]
 for raw in [94,105,128,137]:
  folder=OUT/f'frames/{raw:03d}';folder.mkdir()
  for state in ['keep','delete']:
   out=h2.render_scene(raw,delete=state=='delete')
   write(OUT/'kernel_progress.json',{'actual_native_forwards':h2.actual_forwards,'actual_dummy_backwards':h2.actual_dummy_backwards,'Adam_updates':0})
   rgb=out['rgb'].detach().cpu().numpy();alpha=out['alpha'].detach().cpu().numpy()[0];ds=out['depth_sum'].detach().cpu().numpy()[0]
   values={'rgb':rgb,'alpha':alpha,'depth_sum':ds,'depth_m':ds/np.maximum(alpha,1e-8)/sem.scale,'group_alpha_T':out['group_alpha_T'].detach().cpu().numpy(),'added_owner_rgb_T':out['added_owner_rgb_T'].detach().cpu().numpy(),'newroad_rgb_T':out['newroad_rgb_T'].detach().cpu().numpy()}
   errors={}
   with np.load(R/f'J4_road_parent_operator/K_candidate_render_v1/frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:
    for key in ['alpha','depth_sum','depth_m','group_alpha_T','added_owner_rgb_T']:
     errors[key]=float(np.abs(values[key]-z[key]).max());assert np.array_equal(values[key],z[key]),(raw,state,key)
   np.savez_compressed(folder/(state+'_float.npz'),**values)
   Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(folder/(state+'.png'))
   point_check=None
   if state=='delete' and raw in [128,137]:
    p=next(x for x in proof['point_predictions'] if x['raw']==raw);x,y=p['pixel_xy']
    delta=float(np.abs(values['newroad_rgb_T'][:,y,x]-np.asarray(p['L_newroad_RGB_T'])).max())
    assert delta<=1e-6,(raw,'rollback point',delta)
    point_check={'newroad_RGB_T_vs_CPU_Cwarm_max_abs':delta}
   rows.append({'raw':raw,'state':state,'K_geometry_8groups_semantic_RGB_max_abs':errors,'foreground_road_mass':out['meta']['public_foreground_road_alpha_mass'],'two_point_check':point_check})
   print(json.dumps(rows[-1]),flush=True);del out
  torch.cuda.empty_cache()
 receipt=h2.receipt();assert receipt['actual_native_forwards']==8 and receipt['actual_dummy_feature_backwards']==0
 for path,expected in proof['source_inputs_sha256'].items():assert sha(path)==expected
 assert sha(CONTRACT)==contract_sha256
 write(OUT/'receipt.json',{'status':'COMPLETE_L_CWARM12_NATIVE8F_PENDING_ORIGINAL_94_GATES_AND_ROOT_REVIEW','actual_native_forwards':8,'actual_dummy_backwards':0,'new_solver_calls':0,'Adam_updates':0,'original_model_updates':0,'fixed_parent_ids':PARENTS,'source_sha256':sha(__file__),'contract_sha256':sha(CONTRACT),'cpu_proof_sha256':sha(proofpath),'records':rows,'runtime_receipt':receipt,'seconds':time.time()-start,'general_protection_fix_claim':False,'quality_adopted':False,'automatic49':False,'automatic_fit':False,'shutdown':False})

if __name__=='__main__':main()

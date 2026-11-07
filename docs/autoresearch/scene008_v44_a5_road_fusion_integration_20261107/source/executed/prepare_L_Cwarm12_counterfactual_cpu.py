"""Exact, preselected twelve-parent Cwarm rollback from K. CPU only, no solver."""
from pathlib import Path
import copy,hashlib,json,shutil
import numpy as np
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
R=B/'scene008_v44_a5_road_fusion_integration_20261107_r1'
BASE=R/'J4_road_parent_operator'
PARENTS=np.asarray([12,13,14,83,84,85,153,154,155,224,225,226],dtype=np.int64)
K=BASE/'K_candidate_material_v1/scene_manifest.json'
W=B/'scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1/scene_manifest.json'
PROBE=R/'white_point_parent_diagnostic_v1'
OPERATOR=BASE/'road_parent_operator.npz'
OUT=BASE/'L_Cwarm12_counterfactual_material_v1'
GPU_SOURCE=R/'source/render_L_Cwarm12_counterfactual.py'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def main():
 assert sha(K)=='9f768a44d90f7486ee4ad9346e487d80555be85a733a39cb2c6d0fb9b3936beb'
 km=json.loads(K.read_text());wm=json.loads(W.read_text())
 kp=K.parent/km['new_road']['file'];wp=W.parent/wm['new_road']['file']
 assert sha(kp)=='0858f6a263927c0e8a6545db50a2e741dd205618abf39e194ae852e996a9199f'
 assert sha(wp)=='3afc6e25b7b152f851b4eb0de4a73e6d219c9526d75453d99097a31ddd5774c5'
 assert sha(PROBE/'receipt.json')=='8c1d1af0784b8d445df121b037dd572de683717fe74873debcf3e2329b57f50b'
 assert sha(OPERATOR)=='46cee7c1db5e544a076ad65cb52c41e253f3652d281ad77c97d9b0dfe676811b'
 probe=json.loads((PROBE/'receipt.json').read_text())
 assert probe['status']=='PASS_TWO_POINT_FULL_PARENT_VISIBILITY_DIAGNOSTIC_NO_OPTIMIZATION'
 assert probe['actual_native_forwards']==probe['actual_dummy_backwards']==2
 assert [(x['raw'],x['pixel_xy']) for x in probe['records']]==[(128,[301,221]),(137,[300,227])]
 k=arrays(kp);warm=arrays(wp);operator=arrays(OPERATOR)
 mapping=k['child_to_parent_ids'];assert same(mapping,warm['child_to_parent_ids'])
 assert np.array_equal(mapping,np.repeat(np.arange(10098),4))
 masses={};positive=[]
 inputs=[Path(__file__),GPU_SOURCE,K,W,kp,wp,PROBE/'receipt.json',OPERATOR,R/'contract/K_single_LP_candidate_actual8F_v1.json']
 for rec in probe['records']:
  p=PROBE/rec['mass_file'];assert sha(p)==rec['mass_sha256'];inputs.append(p)
  z=arrays(p);mass=z['parent_alphaT']
  assert mass.shape==(10098,) and mass.dtype==np.float32 and np.isfinite(mass).all() and mass.min()>=0
  positive.extend(np.flatnonzero(mass>0).tolist());masses[rec['raw']]=mass
 assert np.array_equal(np.unique(positive),PARENTS)
 full_A=operator['full_parent_A'];assert full_A.shape==(106,10098)
 assert np.array_equal(full_A[:,PARENTS],np.zeros((106,12),dtype=full_A.dtype))
 before=k['parent_render_colours_srgb'];colours=before.copy()
 colours[PARENTS]=warm['parent_render_colours_srgb'][PARENTS]
 assert colours.dtype==np.float32 and colours.shape==(10098,3) and np.isfinite(colours).all()
 assert colours.min()>=0 and colours.max()<=1
 nonselected=np.ones(10098,bool);nonselected[PARENTS]=False
 assert same(colours[nonselected],before[nonselected])
 assert same(colours[PARENTS],warm['parent_render_colours_srgb'][PARENTS])
 delta=colours.astype(np.float64)-before.astype(np.float64)
 assert np.array_equal(full_A@delta,np.zeros((106,3),dtype=np.float64))
 c0=operator['full_parent_RGB'].astype(np.float64)
 predicted_K=operator['baseline_J']+full_A@(before.astype(np.float64)-c0)
 predicted_L=operator['baseline_J']+full_A@(colours.astype(np.float64)-c0)
 assert same(predicted_L,predicted_K)
 point_predictions=[]
 for rec in probe['records']:
  mass=masses[rec['raw']].astype(np.float64)
  reconstructed_L=mass@colours.astype(np.float64)
  reconstructed_W=mass@warm['parent_render_colours_srgb'].astype(np.float64)
  reconstructed_K=mass@before.astype(np.float64)
  assert same(reconstructed_L,reconstructed_W)
  assert np.abs(reconstructed_L-np.asarray(rec['material_reconstructions']['Cwarm']['cached_newroad_RGB_T'])).max()<=1e-6
  point_predictions.append({'raw':rec['raw'],'state':'delete','pixel_xy':rec['pixel_xy'],
   'L_newroad_RGB_T':reconstructed_L.tolist(),'Cwarm_newroad_RGB_T':reconstructed_W.tolist(),
   'K_newroad_RGB_T':reconstructed_K.tolist(),'L_minus_K_newroad_RGB_T':(reconstructed_L-reconstructed_K).tolist(),
   'all_positive_parent_mass_within_fixed12':True,'not_a_query_mask':True})
 result={key:value.copy() for key,value in k.items()}
 result['parent_render_colours_srgb']=colours
 result['render_colours_srgb']=colours[mapping]
 allowed={'parent_render_colours_srgb','render_colours_srgb'}
 assert set(k)==set(result)
 for key in k:
  if key not in allowed:assert same(result[key],k[key]),key
 changed_ids=np.flatnonzero(np.any(colours!=before,axis=1))
 assert np.isin(changed_ids,PARENTS).all()
 changed_children=np.flatnonzero(np.any(result['render_colours_srgb']!=k['render_colours_srgb'],axis=1))
 assert np.array_equal(changed_children,np.flatnonzero(np.isin(mapping,changed_ids)))
 fixed=np.ones(10098,bool);fixed[operator['eligible_parent_ids']]=False
 assert same(colours[fixed],before[fixed])
 replay=[]
 for item in km['replay_sources'].values():
  p=K.parent/item['file'];assert sha(p)==item['sha256'];inputs.append(p);replay.append((p,item['file']))
 h2=R/'H4_model_foreground_culling_v2/receipt.json';inputs.append(h2)
 cached=json.loads((R/'contract/K_single_LP_candidate_actual8F_v1.json').read_text())['cached_visibility']
 for item in cached:
  p=Path(item['file']);assert sha(p)==item['sha256'];inputs.append(p)
 inputs_sha={str(p):sha(p) for p in inputs}
 OUT.mkdir(exist_ok=False);(OUT/'provenance').mkdir()
 path=OUT/'new_road_L_Cwarm12_counterfactual.npz';np.savez_compressed(path,**result)
 saved=arrays(path)
 for key in k:
  assert same(saved[key],result[key])
  if key not in allowed:assert same(saved[key],k[key])
 rgb_path=OUT/'provenance/L_parent_RGB.npz'
 np.savez_compressed(rgb_path,full_parent_RGB=colours,restored_parent_ids=PARENTS,
  actual_changed_parent_ids=changed_ids,delta_from_K=delta,
  K_signed_predictions=predicted_K,L_signed_predictions=predicted_L,
  full_parent_A_fixed12=full_A[:,PARENTS])
 for source,name in replay:shutil.copyfile(source,OUT/name)
 for source,name in [(K,'K_scene_manifest.json'),(W,'Cwarm_scene_manifest.json'),(PROBE/'receipt.json','two_point_probe_receipt.json'),(Path(__file__),'prepare_L_Cwarm12_counterfactual_cpu.py')]:shutil.copyfile(source,OUT/'provenance'/name)
 proof={'schema':'L_EXACT_TWELVE_PARENT_CWARM_CAUSAL_COUNTERFACTUAL_CPU_V1',
  'status':'PASS_CPU_EXACT_CWARM12_ROLLBACK_PENDING_NATIVE8F','fixed_parent_ids':PARENTS.tolist(),
  'actual_changed_parent_ids':changed_ids.tolist(),'actual_changed_child_rows':changed_children.tolist(),
  'selection_was_fixed_by_root_before_export':True,'fixed_ids_equal_original_two_probe_positive_union':True,
  'full_operator_106_by_12_all_exact_zero':True,'all_106_K_affine_predictions_byte_exact':True,
  'both_probe_newroad_RGB_T_equal_Cwarm':True,'point_predictions':point_predictions,
  'only_parent_and_child_RGB_arrays_changed':True,'all_other_array_dtype_shape_bytes_exact_to_K':True,
  'all_other_parent_RGB_byte_exact':True,'frozen7979_RGB_byte_exact':True,
  'source_inputs_sha256':inputs_sha,'K_parent_asset_sha256':sha(kp),'Cwarm_asset_sha256':sha(wp),
  'operator_npz_sha256':sha(OPERATOR),'full_parent_RGB_provenance_file':'provenance/L_parent_RGB.npz',
  'full_parent_RGB_provenance_sha256':sha(rgb_path),'new_asset_sha256':sha(path),
  'CPU_solver_calls':0,'Adam_updates':0,'original_model_updates':0,'native_forwards_by_preparer':0,
  'general_protection_fix_claim':False,'quality_adopted':False,'scope':'Only tests causal impact of the fixed twelve parents after C49 RGB changes; no generic protection rule.'}
 proofpath=OUT/'cpu_counterfactual_proof.json';write(proofpath,proof)
 manifest=copy.deepcopy(km)
 manifest['new_road'].update(file=path.name,sha256=sha(path),material_variant='L_fixed12_Cwarm_rollback_from_K_no_new_optimization')
 manifest['lineage']={'K_parent_manifest_file':'provenance/K_scene_manifest.json','K_parent_manifest_sha256':sha(K),
  'L_Cwarm12_counterfactual':{'fixed_parent_ids':PARENTS.tolist(),'actual_changed_parent_ids':changed_ids.tolist(),
   'K_source_asset_sha256':sha(kp),'Cwarm_source_asset_sha256':sha(wp),'cpu_proof_file':proofpath.name,'cpu_proof_sha256':sha(proofpath),
   'new_solver_calls':0,'Adam_updates':0,'original_model_updates':0,'only_two_RGB_arrays_changed':True,'all_other_fields_byte_exact':True,'general_protection_fix_claim':False}}
 manifest['material_writeback'].update(L_new_CPU_solver_calls=0,L_Adam_updates=0,L_fixed_Cwarm_parent_ids=PARENTS.tolist())
 manifest['known_unresolved']+=['L is a posthoc fixed12-parent causal rollback, not a reusable general protection fix. Original94 gates and images remain pending native8F.']
 manifest.update(exporter_sha256=sha(__file__),GPU_parity_state_at_export='L_NATIVE8F_NOT_AUTHORIZED_YET',GPU_parity_receipt='native_validation_pending.json',adoption_claim=False,human_verdict=None)
 mp=OUT/'scene_manifest.json';write(mp,manifest)
 write(OUT/'native_validation_pending.json',{'status':'NO_NATIVE_FORWARD_BY_CPU_PREPARATION','actual_F':0,'actual_B':0,'planned_only_after_root_source_contract_review':8})
 draft={'schema':'L_CWARM12_CAUSAL_COUNTERFACTUAL_NATIVE8F_PROTOCOL_V1','status':'DRAFT_CPU_PREPARATION_COMPLETE_PENDING_ROOT_GPU_REVIEW',
  'fixed_parent_ids':PARENTS.tolist(),'actual_changed_parent_ids':changed_ids.tolist(),
  'scene_manifest':str(mp),'scene_manifest_sha256':sha(mp),'asset_sha256':sha(path),
  'cpu_proof':str(proofpath),'cpu_proof_sha256':sha(proofpath),
  'native_render_source':str(GPU_SOURCE),'native_render_source_sha256':sha(GPU_SOURCE),
  'CPU_preparation_source':str(Path(__file__)),'CPU_preparation_source_sha256':sha(__file__),
  'raw_ids':[94,105,128,137],'states':['keep','delete'],'fixed_base':'K material on original fixed H2/21K semantic scene, no I/H3',
  'H2_cached_visibility':cached,'no_new_visibility_probe':True,
  'original_evaluation_contract_sha256':'4560bb91f4cc3b27b89693acd0918f939d2ac07e9bcd7cbbc6e0574e885516cc',
  'all_original_94_quality_gates_remain_unchanged':True,'six_pool_and_106_J_operator_residuals_expected_same_as_K':True,
  'two_point_newroad_RGB_T_expected_back_to_Cwarm':True,'planned_native_forwards':8,'planned_dummy_backwards':0,
  'new_solver_calls':0,'Adam_updates':0,'original_model_updates':0,'no_clamp_no_projection_no_subset_or_ratio_search':True,
  'general_protection_fix_claim':False,'quality_adopted':False,'automatic49':False,'automatic_fit':False,'shutdown':False}
 dp=R/'contract/L_Cwarm12_counterfactual_v1.draft.json';assert not dp.exists();write(dp,draft)
 for source,expected in inputs_sha.items():assert sha(source)==expected
 print(json.dumps({'status':proof['status'],'scene_manifest':str(mp),'scene_manifest_sha256':sha(mp),'asset_sha256':sha(path),'cpu_proof_sha256':sha(proofpath),'draft_contract_sha256':sha(dp),'native_render_source_sha256':sha(GPU_SOURCE),'actual_changed_parent_ids':changed_ids.tolist(),'actual_changed_children':len(changed_children),'GPU_calls':0},indent=2),flush=True)

if __name__=='__main__':main()

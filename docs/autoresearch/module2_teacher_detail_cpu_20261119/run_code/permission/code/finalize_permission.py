"""最终CPU映射契约/清单；不修改积分系数、门、幅度、旧P或任何原生场景。"""
from pathlib import Path
import json,shutil
import numpy as np
from teacher_detail_permission import Q,N,load,sha
def main():
 p=Q/'permission';f=p/'final_r1';s=p/'avg_sparse_gh31_r1'
 d=load(f/'teacher_evidence.npz',['active_world_rows','common_write_active'])
 m=load(s/'row_mapping.npz');op=load(s/'operator/constants.npz',['row_ids'])
 with np.load(N/'road.npz')as old:ctx_active=np.flatnonzero(old['active'])
 idx=np.flatnonzero(d['common_write_active']);expected=d['active_world_rows'][idx]
 checks=dict(full_active_order_exact_parent=np.array_equal(d['active_world_rows'],ctx_active),
  operator_global_world_ids_exact_scope=np.array_equal(op['row_ids'],expected),
  operator_world_ids_exact_mapping_npz=np.array_equal(op['row_ids'],m['query_world_rows']),
  active_local_indices_exact=np.array_equal(m['active_local_indices'],idx),
  full_active_mapping_exact=np.array_equal(m['full_active_world_rows'],ctx_active),
  future_scatter_index_exact=np.array_equal(np.searchsorted(ctx_active,op['row_ids']),idx))
 if not all(checks.values()):raise AssertionError(str(checks))
 contract=dict(schema='average_operator_row_id_contract.v1',row_id_semantics='global_road_topology_world_row_id',
  operator_array='operator/constants.npz::row_ids',not_active_local_positions=True,
  query_rows=len(expected),full_active_rows=len(ctx_active),checks=checks,
  context_snapshot='CPU np.flatnonzero(parent road.active), same definition as prior actual native context; no new native context loaded',
  future_runtime_check='idx=searchsorted(ctx.active_world_rows,op.row_ids); assert ctx.active_world_rows[idx]==op.row_ids; assert idx==flatnonzero(common_write_active)',
  CPU_only=True,native_calls=0,CUDA_calls=0,optimizer_updates=0)
 (s/'row_id_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
 recipe=json.loads((f/'recipe.json').read_text())
 recipe.update(average_row_id_semantics=contract['row_id_semantics'],average_row_id_contract=str(s/'row_id_contract.json'),
  average_row_id_contract_sha256=sha(s/'row_id_contract.json'))
 before=f/'recipe_before_row_id_contract.json'
 if not before.exists():shutil.copy2(f/'recipe.json',before)
 (f/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
 evidence=['old_audit_r1/receipt.json','ledger_r1/receipt.json','ledger_r2/receipt.json','numeric_r1/receipt.json',
  'numeric21_r1/receipt.json','numeric31_r1/receipt.json','numeric31_r1/reference61_spot.json',
  'avg_plan_gh31_r1/recipe.json','avg_sparse_gh31_r1/receipt.json','avg_sparse_gh31_r1/gradient_validation.json',
  'avg_sparse_gh31_r1/upload_verified.json','avg_sparse_gh31_r1/row_id_contract.json','final_r1/recipe.json','final_r1/cpu_tests.json']
 inventory=dict(schema='permission_final_source_inventory.v2',frozen=True,
  code={v.name:dict(path=str(v),sha256=sha(v))for v in sorted((p/'code').glob('*.py'))},
  evidence={name:dict(path=str(p/name),sha256=sha(p/name))for name in evidence},
  note='Local builder/compiler/chart snapshots are actual executed offload sources. Scientific oldP never modified; later recipe additions are only sparse paths/row-id metadata.',
  models_and_large_arrays_not_Git_sources=True,CUDA_calls=0,native_calls=0,optimizer_updates=0)
 for name in ['source_inventory.json','results.json']:
  if (p/name).exists()and not (p/(name+'.before_final_sparse')).exists():shutil.copy2(p/name,p/(name+'.before_final_sparse'))
 (p/'source_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
 result=dict(schema='teacher_H_permission_CPU_final.v2',status='CPU_PREPARED_STOP_BEFORE_GPU',
  common_high_bound=.34,scope_candidate_rows=len(expected),old_rows_retained=84266,new_rows=5621,
  all_arms_common_base_valid_scope_latent_reg=True,prior_source_known_unknown_not_changed=True,
  row_id_semantics=contract['row_id_semantics'],mapping_checks=checks,CPU_actual_tests=9,
  new_GH31_numeric_pass=True,independent_GH61_spot_pass=True,full_average_UV_plan_complete=True,
  sparse_average_operator_uploaded_and_verified=True,sparse_actual150_RGB_and_L_H_gradients_exact=True,
  mode='Gaussian_footprint_average',kernel_order=31,KH_includes_gate=True,
  avg_operator=str(s/'operator'),permission_recipe=str(f/'recipe.json'),source_inventory=str(p/'source_inventory.json'),
  CPU_only=True,GPU_authorized=False,CUDA_calls=0,native_calls=0,optimizer_updates=0,no_controller_created=True,
  native_scope_K_sparse_ABI_and_lifecycle_still_require_explicit_future_GPU_authorization=True)
 (p/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()


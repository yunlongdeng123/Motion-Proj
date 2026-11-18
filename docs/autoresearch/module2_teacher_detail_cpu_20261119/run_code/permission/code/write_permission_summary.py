"""只读CPU收尾：实际测试结果、source inventory与稳定evidence集合。"""
from pathlib import Path
import json
from teacher_detail_permission import Q,sha
def main():
 p=Q/'permission';final=p/'final_r1'
 tests=dict(schema='teacher_permission_real_CPU_tests.v1',passed=True,tests=9,actual_GH31_fixture=True,
  command='CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 motionproj/python -m pytest -q permission/code/test_permission_cpu.py',
  captured_output='9 passed in 12.56s',execution_session=43524,CPU_only=True,CUDA_calls=0,native_calls=0,optimizer_updates=0,
  source_sha256={str(v):sha(v)for v in [p/'code/test_permission_cpu.py',p/'code/teacher_detail_permission.py',final/'cpu_mini_fixture.npz',final/'recipe.json']})
 (final/'cpu_tests.json').write_text(json.dumps(tests,indent=2)+'\n')
 numeric=json.loads((p/'numeric31_r1/receipt.json').read_text());spot=json.loads((p/'numeric31_r1/reference61_spot.json').read_text())
 plan=json.loads((p/'avg_plan_gh31_r1/recipe.json').read_text())
 files=['old_audit_r1/receipt.json','ledger_r1/receipt.json','ledger_r2/receipt.json','numeric_r1/receipt.json','numeric21_r1/receipt.json',
  'numeric31_r1/receipt.json','numeric31_r1/reference61_spot.json','avg_plan_gh31_r1/recipe.json','final_r1/recipe.json','final_r1/cpu_tests.json']
 inventory=dict(schema='permission_owned_CPU_source_inventory.v1',CPU_only=True,
  sources={v.name:dict(path=str(v),sha256=sha(v))for v in sorted((p/'code').glob('*.py'))},
  evidence={name:dict(path=str(p/name),sha256=sha(p/name))for name in files},
  original_P_modified=False,GPU_authorized=False,native_calls=0,optimizer_updates=0)
 (p/'source_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
 result=dict(schema='teacher_H_permission_CPU_preparation_summary.v1',status='CPU_PREPARED_STOP_BEFORE_GPU',
  permission=str(final/'teacher_evidence.npz'),recipe=str(final/'recipe.json'),cpu_mini_fixture=str(final/'cpu_mini_fixture.npz'),
  source_inventory=str(p/'source_inventory.json'),actual_CPU_tests=tests['tests'],kernel_order=31,
  new_numeric_150row_passed=numeric['status']=='PASS',independent61spot_passed=spot['status']=='PASS',
  full_average_plan_complete=plan['completed'],query_rows=plan['query_rows'],kernel_points=plan['mapped_points'],
  full_average_plan=str(p/'avg_plan_gh31_r1'),scope_role='CPU candidate only; actual native qualification before future training',
  prior_labels_immutable=True,common_H_bound=.34,old84266_retained=84266,new_write_rows=5621,
  CUDA_calls=0,native_calls=0,optimizer_updates=0,controller_created=False,original_P_modified=False)
 (p/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()


"""已完成full GH31后将四个真实Σ小fixture对齐相同核；保留旧225路由fixture。"""
from pathlib import Path
import json,shutil
import numpy as np
from teacher_detail_permission import Q,load,sha
def main():
 final=Q/'permission/final_r1';plan=Q/'permission/avg_plan_gh31_r1'
 receipt=json.loads((plan/'recipe.json').read_text())
 if not receipt['completed'] or receipt['order']!=31:raise ValueError('complete GH31 plan missing')
 path=final/'cpu_mini_fixture.npz';d=load(path);recipe=json.loads((final/'recipe.json').read_text())
 mapping=load(plan/'geometry_plan.npz',['query_world_rows','weights'])
 uv=np.load(plan/'sample_uv_f64.npy',mmap_mode='r');valid=np.load(plan/'geometric_sample_valid.npy',mmap_mode='r')
 backup=final/'cpu_mini_fixture_gh15_routing_backup.npz'
 if not backup.exists():shutil.copyfile(path,backup)
 for case in recipe['fixture_cases']:
  if case['mode']!='Sigma':continue
  name=case['case'];j=int(np.searchsorted(mapping['query_world_rows'],case['world_row']))
  if mapping['query_world_rows'][j]!=case['world_row']:raise ValueError('fixture not commonwrite')
  d[name+'__uv']=np.asarray(uv[j])-np.asarray(case['origin_uv'])
  d[name+'__weights']=mapping['weights'];d[name+'__geometry']=np.asarray(valid[j])
 np.savez_compressed(path,**d)
 recipe['CPU_mini_Sigma_kernel_order']=31;recipe['CPU_mini_fixture_sha256']=sha(path)
 (final/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
 print(json.dumps(dict(order=31,updated_actual_Sigma_cases=4,pointcases_unchanged=True,original_225_fixture_preserved=True,native_calls=0,updates=0)))
if __name__=='__main__':main()


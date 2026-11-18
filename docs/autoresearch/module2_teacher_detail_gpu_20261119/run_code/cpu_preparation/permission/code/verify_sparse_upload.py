"""核对local编译算子传输原字节，不导入Torch/native。"""
from pathlib import Path
import hashlib,json
Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 out=Q/'permission/avg_sparse_gh31_r1';r=json.loads((out/'receipt.json').read_text());checks={}
 for name,expected in r['operator_files'].items():
  p=out/'operator'/name;checks[name]=p.stat().st_size==expected['bytes']and sha(p)==expected['sha256']
 if not all(checks.values()):raise AssertionError(str(checks))
 g=json.loads((out/'gradient_validation.json').read_text());m=json.loads((out/'operator/read.json').read_text())
 if not g['passed']or m['rows']!=89887 or m['metadata']['mode']!='Gaussian_footprint_average':raise AssertionError('actual validation/mode missing')
 result=dict(completed=True,all_source_bytes_verified=True,checks=checks,rows=m['rows'],order=31,
  operator_dir=str(out/'operator'),KH_already_contains_gate=True,
  actual_150row_forward_and_grad_passed=True,CPU_only=True,native_calls=0,CUDA_calls=0,optimizer_updates=0,GPU_authorized=False,
  complete_remote_UV_plan_preserved=str(Q/'permission/avg_plan_gh31_r1'),readonly_worldrows_require_parent_RGB_override=True)
 (out/'upload_verified.json').write_text(json.dumps(result,indent=2)+'\n')
 recipe=Q/'permission/final_r1/recipe.json';d=json.loads(recipe.read_text())
 d['average_sparse_read_operator']=str(out/'operator');d['average_sparse_upload_verified']=str(out/'upload_verified.json')
 d['average_sparse_mode']='Gaussian_footprint_average';d['KH_already_contains_gate_do_not_apply_again']=True
 recipe.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()


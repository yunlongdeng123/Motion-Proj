"""同一22点独立61参考交叉，不重算核。"""
import json
from teacher_detail_permission import Q,load
from repair_quadrature_small import metric
def main():
 output=Q/'permission/numeric31_r1/reference61_spot.json';rows=[]
 for case in ['base_H0','H_nyquist_u','H_nyquist_diag45','teacher_HF_seed']:
  d=load(Q/'permission/numeric31_r1'/(case+'.npz'));r=load(Q/'permission/numeric21_r1'/(case+'_61spot.npz'))
  m=metric(d['candidate_RGB'][r['sample_indices']],d['candidate_mass'][r['sample_indices']],r['reference61_RGB'],r['reference61_mass'])
  m['case']=case;rows.append(m)
 result=dict(status='PASS' if all(r['passed'] for r in rows) else 'FAIL',candidate_order=31,reference_order=61,rows=22,cases=rows,
  CPU_only=True,new_world_queries=0,native_calls=0,optimizer_updates=0)
 output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()


"""冻结 QEM 规则的预算匹配资产；不读取射线或质量结果。"""
import argparse,json,time
from pathlib import Path
import numpy as np
import fast_simplification
def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[]
    plan={'task':'WS-V74-H2-NKSR-BUDGET-01','max_faces':4096,'algorithm':'quadric error decimation fast-simplification 0.1.12',
        'aggressiveness':7,'rule':'exact-coordinate vertex merge and repeated-vertex zero-area face removal for every mesh; leave <=4096 unchanged thereafter, otherwise QEM target_count=4096 agg=7; no quality-based changes',
        'scope':'adapted physical asset reference; native results and assets retained; not NKSR original paper result',
        'quality_or_ray_values_read':False,'rule_selection':'standard fixed QEM settings; FIT processed before DEV; no metric tuning',
        'failure_ledger_refs':['V74-F06','V74-F11'],'failure_ledger_delta':'pending'}
    (a.output/'manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
    for row in json.loads(a.index.read_text()):
        with np.load(a.inputs/row['input']) as z:v=z['vertices_actor_m'];f=z['faces']
        n=len(f);t=time.monotonic();native_v=v;aggressiveness=7
        v,inverse=np.unique(v,axis=0,return_inverse=True);f=inverse[f]
        take=(f[:,0]!=f[:,1])&(f[:,0]!=f[:,2])&(f[:,1]!=f[:,2]);removed=int((~take).sum());f=f[take]
        merged_vertices=len(native_v)-len(v)
        if len(f)>4096:v,f=fast_simplification.simplify(v,f,target_count=4096,agg=7)
        out=a.output/row['input'];out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,vertices_actor_m=v,faces=f)
        rows.append({**row,'output':str(out),'native_faces':n,'faces':len(f),'within_budget':len(f)<=4096,'empty':not len(f),'seconds':time.monotonic()-t,'aggressiveness':aggressiveness,'merged_exact_vertices':merged_vertices,'removed_repeated_vertex_faces':removed})
    result={'task':plan['task'],'status':'done','assets':rows,'modified':sum(r['native_faces']>4096 for r in rows),'empty':sum(r['empty'] for r in rows),
        'over_budget':sum(not r['within_budget'] for r in rows),'wall_s':time.monotonic()-start,'quality_evaluated':False}
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='assets'}))
if __name__=='__main__':main()

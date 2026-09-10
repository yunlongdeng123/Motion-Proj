"""WEX 首次解析机制实验：固定共享域析取系统，精确枚举与成熟 MILP 对照。"""
import argparse,itertools,json,sys,time,resource
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.a_wex import DomainProblem
from motion_proj.worldsim_v74.common import write_json

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
rng=np.random.default_rng(7401);records=[];start=time.monotonic()
for family in ['joint_exchange','multiple_front_constraints','empty_correct_candidate','duplicate_and_scale']:
    for trial in range(20):
        # x1,x2 是原责任，y1,y2 是替换责任；三条共享自由约束使单换不可行、双换可行。
        mat=np.array([[.5,.5,0,0],[.5,0,0,.5],[0,.5,.5,0],
                      [1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],float)
        free=np.array([0,1,2]);groups={0:np.array([3,4]),1:np.array([5,6])}
        if family=='multiple_front_constraints':
            weights=rng.uniform(.35,.65,size=3)
            for row,w in zip(range(3),weights):
                nz=np.flatnonzero(mat[row]);mat[row,nz]=[w,1-w]
        if family=='empty_correct_candidate':groups[2]=np.array([],int)
        if family=='duplicate_and_scale':
            mat=np.vstack([mat,mat[4],mat[6]]);groups={0:np.array([3,4,7]),1:np.array([5,6,8])}
        # 平移/旋转不改变线性域系数，此处只声明约束子系统证据，尚非三维几何 80 例。
        order=rng.permutation(4);mat=mat[:,order]
        prior=np.array([1.,1.,0.,0.])[order]
        nonempty={r:ids for r,ids in groups.items() if len(ids)}
        folder=args.output/f'{family}_{trial:02d}';folder.mkdir()
        np.savez_compressed(folder/'problem.npz',matrix=mat,free=free,prior=prior)
        results={}
        for method in ['A1_fixed','A2_greedy','WEX','A3_milp','enumeration']:
            t=time.monotonic();problem=DomainProblem(sp.csr_matrix(mat),free,nonempty,prior,sp.eye(4,format='csc'))
            if method=='A3_milp':state,trace,states=problem.mixed_integer(5)
            elif method=='enumeration':
                candidates=[problem.solve(list(y)) for y in itertools.product(*nonempty.values())]
                state=min(candidates,key=lambda s:s['score']);trace=[];states=[state['x']]
            else:state,trace,states=problem.search(method,max_outer=10,seconds=5)
            result={'feasible_listed_constraints':bool(state['score'][0]==0),'all_observed_rays_explained':bool(state['score'][0]==0 and len(nonempty)==len(groups)),
                    'empty_G':len(groups)-len(nonempty),'slack_sum':state['score'][1],'objective':state['score'][2],
                    'wall_seconds':time.monotonic()-t,'qp_calls':problem.calls,'events':trace}
            np.savez_compressed(folder/f'{method}.npz',domain_values=state['x'],responsibility=state['y'])
            results[method]=result
        write_json(folder/'result.json',results);records.append({'family':family,'trial':trial,'results':results})
    print(f'{family}: {len(records)} complete',flush=True)
summary={}
for family in sorted({r['family'] for r in records}):
    summary[family]={method:{'feasible':sum(r['results'][method]['feasible_listed_constraints'] for r in records if r['family']==family),
                            'all_explained':sum(r['results'][method]['all_observed_rays_explained'] for r in records if r['family']==family),
                            'wall_seconds':sum(r['results'][method]['wall_seconds'] for r in records if r['family']==family)} for method in records[0]['results']}
write_json(args.output/'summary.json',{'task':'WS-V74-A-MECHANISM-01','run':'20260910__domain-subsystem-r1-s7401','seed':7401,
    'scope':'80 analytic shared-domain disjunction instances; not the required end-to-end geometric mechanism cohort or real validation',
    'families':summary,'wall_seconds':time.monotonic()-start,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
    'failure_ledger_refs':['V73-F02','V73-F03'],'verdict':'not_decided'})
print(json.dumps(summary,indent=2),flush=True)

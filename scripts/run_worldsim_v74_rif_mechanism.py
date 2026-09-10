"""RIF 的已知三维场机制：真实网格首交点、区间端点和保场细分。"""
import argparse,json,sys,time,itertools
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v74.b_rif import TetField,regular_tets,field_solve
from motion_proj.worldsim_v74.common import write_json,save_mesh,all_intersections

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);rng=np.random.default_rng(7401);records=[];start=time.monotonic()
for family in ['hidden_zero_crossing','narrow_gap','coarse_root_conflict','grazing_noise']:
    for trial in range(20):
        field=TetField(*regular_tets([-1,-1,-1],[1,1,1],5))
        shift=float(rng.uniform(-.08,.08));angle=float(rng.uniform(-.1,.1))
        # 连续 P1 场有解析可查的薄层/开口；不把整个未知后段当实体。
        signed=field.v[:,2]+angle*field.v[:,0]-shift
        if family=='hidden_zero_crossing':c=np.abs(signed)-.15
        elif family=='narrow_gap':c=np.abs(signed)-.04
        else:c=signed.copy()
        o=np.stack([rng.uniform(-.7,.7,12),rng.uniform(-.7,.7,12),np.full(12,2.)],1)
        d=np.tile([0.,0.,-1.],(len(o),1))
        if family=='grazing_noise':d[:]=[.98,0,-.2];d/=np.linalg.norm(d,axis=1,keepdims=True);o[:,0]=-1.5;o[:,2]=.3
        ends=np.full(len(o),4.)
        F,interval=field.intervals(o,d,ends);v,f,zero=field.extract(c)
        hits=all_intersections(v,f,o,d)
        # 根据 P1 每单元的仿射值直接计算根，完全独立于 marching tetrahedra 的拓扑。
        vv=(F@c).reshape(-1,2);roots=[]
        for i,(a,b) in enumerate(vv):
            if a*b<0:roots.append((int(interval['ray'][i]),float(interval['lo'][i]+(-a)/(b-a)*(interval['hi'][i]-interval['lo'][i]))))
        mismatch=[]
        for r,t in roots:
            actual=hits['t'][hits['ray']==r]
            mismatch.append(float(np.min(np.abs(actual-t))) if len(actual) else 1e3)
        # 全部区间端点正且无松弛时，不允许声称空段却导出交点。
        contrad=0
        for r in range(len(o)):
            take=np.flatnonzero(interval['ray']==r)
            positive=bool(len(take) and (vv[take]>1e-7).all())
            contrad+=int(positive and np.any(hits['ray']==r))
        chosen=np.arange(min(20,len(field.t)))
        refined,cc,mapping=field.refine(chosen,c)
        p=field.v[field.t[chosen]].mean(1);M,_=refined.point_rows(p)
        prolong=float(np.max(np.abs(M@cc-c[field.t[chosen]].mean(1))))
        # 固定粗网格的同完整区间控制与普通采样控制；不是自适应方法获胜证明。
        folder=args.output/f'{family}_{trial:02d}';folder.mkdir()
        save_mesh(folder/'initial.npz',v,f,field_vertices=field.v,tetrahedra=field.t,field_values=c)
        np.savez_compressed(folder/'ray_interval.npz',o=o,d=d,values=vv,**interval)
        r={'family':family,'trial':trial,'field_roots':len(roots),'mesh_intersections':len(hits['ray']),
            'max_mesh_field_root_error_m':max(mismatch,default=0.),'false_free_claims':contrad,
            'prolongation_max_error':prolong,'zero_tetra_count':zero}
        records.append(r);write_json(folder/'result.json',r)
    print(family,'complete',flush=True)
write_json(args.output/'summary.json',{'task':'WS-V74-B-MECHANISM-01','seed':7401,'scope':'80 fixed-field geometry numerical mechanisms; learned/adaptive solving comparisons not yet completed',
    'instances':records,'wall_seconds':time.monotonic()-start,'false_free_claims':sum(r['false_free_claims'] for r in records),
    'max_mesh_field_root_error_m':max(r['max_mesh_field_root_error_m'] for r in records),'max_prolongation_error':max(r['prolongation_max_error'] for r in records),'verdict':'not_decided'})
print('RIF primitive mechanisms complete',flush=True)

"""r27有界工程检查：地图片段末端是否被错误当作轨迹末端。

只连接官方相邻车道，固定选择切向变化最小的分支；保持原中点/速度和全部几何门槛。
该脚本不准入训练，不跑生成模型，不调整已冻结r23产物。
"""
from pathlib import Path
import sys, math
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))
import reveal_factory as f
from audit_reveal_rejections import full_support
from scipy.spatial import cKDTree
import numpy as np

O=f.T/'r27'


def turn(a,b):
    return abs(math.atan2(math.sin(a-b),math.cos(a-b)))


def connected(q, api, speed):
    """保留同一个中点，前后只补足1.5秒所需的官方拓扑长度。"""
    p=q['path'].copy();anchor=q['s'];tokens=[q['token']];current=q['token']
    for direction in ['incoming','outgoing']:
        current=tokens[0] if direction=='incoming' else tokens[-1]
        for _ in range(5):
            ss=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1))]
            available=anchor if direction=='incoming' else ss[-1]-anchor
            if available>=speed*1.5+1:break
            ids=api.get_incoming_lane_ids(current) if direction=='incoming' else api.get_outgoing_lane_ids(current)
            options=[]
            for token in sorted(set(ids)-set(tokens)):
                raw=api.discretize_lanes([token],1.)[token]
                a=np.asarray(raw,float)
                if len(a)<2:continue
                end,start=(a[-1],p[0]) if direction=='incoming' else (p[-1],a[0])
                distance=np.linalg.norm(end[:2]-start[:2]);angle=turn(end[2],start[2])
                if distance>1.5 or angle>math.radians(20):continue
                options.append((angle,distance,token,a))
            if not options:break
            _,_,token,a=min(options,key=lambda v:v[:3]);current=token
            if direction=='incoming':
                old_len=ss[-1];p=np.r_[a,p];tokens.insert(0,token)
            else:p=np.r_[p,a];tokens.append(token)
            keep=np.r_[True,np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)>1e-6]
            p=p[keep];p[:,2]=np.unwrap(p[:,2])
            new_len=float(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).sum())
            if direction=='incoming':anchor+=new_len-old_len
    ss=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1))]
    old=np.array([np.interp(q['s'],q['distances'],q['path'][:,i]) for i in range(3)])
    now=np.array([np.interp(anchor,ss,p[:,i]) for i in range(3)])
    assert np.linalg.norm(old[:2]-now[:2])<1e-5 and turn(old[2],now[2])<1e-5
    return q|{'path':p,'distances':ss,'s':anchor,'connected_tokens':tokens}


def main():
    O.mkdir(exist_ok=True)
    assert not (O/'connected_lane_result.json').exists(),'已有结果不可覆盖或重复跑'
    records=[f.read(p) for p in (f.O/'lane_candidates').glob('*.json')]
    selected=sorted(records,key=lambda r:(-r['rejects'].get('lane_window_exhausted',0),r['source_id']))[:3]
    ids=[r['source_id'] for r in selected]
    f.dump(O/'run.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r27',
        'purpose':'bounded_source_engineering_probe','sources':ids,
        'selection':'top3 r23 lane_window_exhausted counts; before new outputs',
        'same_midpoints_speeds_thresholds':True,'max_connection_hops_each_direction':5,
        'branches':'minimum tangent difference then gap then token; no branch search',
        'support_control':'same old fitted plane, same .08m band/2.5m distance; full in-window observed points',
        'training_steps':0,'training_admission':0,'failure_ledger_refs':['V77-F02'],'human_verdict':None})
    f.legacy.O=f.O;f.legacy.ROOT=f.ROOT;g=f.legacy.geometry()
    api=f.NuScenesMap(dataroot=str(f.ROOT),map_name='boston-seaport');rows=[]
    for sid in ids:
        c=g.sources[sid];g.prepare(sid);f.legacy.support(g,sid);oldtree=g.ground[sid]['_tree']
        points,scans=full_support(g,sid);fulltree=cKDTree(points[:,:2])
        asset=f.POLICY['split_shape'][c['source_split']];size=[1.85,4.5,1.5] if asset=='sedan' else [1.9,4.6,1.7]
        results=[]
        for q in f.centers(g,api,c,size):
            for speed in f.POLICY['speeds_mps']:
                old,reason=f.trajectory(g,c,q,speed,asset)
                if reason!='lane_window_exhausted':continue
                joined=connected(q,api,speed)
                topo,topo_reason=f.trajectory(g,c,joined,speed,asset)
                g.ground[sid]['_tree']=fulltree
                full,full_reason=f.trajectory(g,c,joined,speed,asset)
                g.ground[sid]['_tree']=oldtree
                results.append({'lane_token':q['token'],'old_s':q['s'],'speed_mps':speed,
                    'connected_tokens':joined['connected_tokens'],'old_result':reason,
                    'topology_only':'geometry_pass' if topo else topo_reason,
                    'topology_plus_full_observed_ground':'geometry_pass' if full else full_reason})
        row={'source_id':sid,'scene':c['scene'],'trials':results,'in_window_scans':scans,
             'topology_counts':dict(Counter(r['topology_only'] for r in results)),
             'combined_counts':dict(Counter(r['topology_plus_full_observed_ground'] for r in results))}
        rows.append(row);print('CONNECTED',sid,row['topology_counts'],row['combined_counts'],flush=True)
        f.dump(O/'connected_lane_partial.json',{'sources':rows,'training_admission':0})
    f.dump(O/'connected_lane_result.json',{'sources':rows,'training_admission':0,
        'boundary':'geometry acceptance only; exact silhouette, visible instance and reveal QA not yet run'})


if __name__=='__main__':main()

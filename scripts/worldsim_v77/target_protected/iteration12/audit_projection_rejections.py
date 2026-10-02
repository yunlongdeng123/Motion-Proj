"""把r33的复合投影拒绝拆开，判断是否真因合法出入画而缺样本；不改任何门槛。"""
from pathlib import Path
import sys,os
from collections import Counter
sys.path.insert(0,str(Path(__file__).parent))
import target_linked_proposals as p
f=p.f;np=p.np;O=f.T/'r35'

def main():
    O.mkdir(exist_ok=False)
    f.dump(O/'run.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r35',
        'source':'r33 fixed solutions rejected by size_border_or_ego only','change':'diagnosis only, no new positions or thresholds',
        'question':'horizontal entry/exit vs ego/top/scale constraints','training_steps':0,'GPU_jobs':0,
        'stop_rule':'one decomposition of existing failures','failure_ledger_refs':['V77-F02']})
    f.legacy.O=f.O;f.legacy.ROOT=f.ROOT;g=f.legacy.geometry();rows=[]
    for r in f.read(f.T/'r33/proposal_audit.json')['rows']:
        if r['result']!='size_border_or_ego':continue
        sid=r['source_id'];c=g.sources[sid]
        if sid not in g.obstacles:g.prepare(sid)
        pm=f.legacy.protections(g,sid);asset=f.POLICY['split_shape'][c['source_split']]
        size=[1.85,4.5,1.5] if asset=='sedan' else [1.9,4.6,1.7]
        q,why=p.solve(g,c,r['intended_B'],r['anchor'],r['side'],size,pm[r['intended_B']][r['anchor']]);assert q is not None
        a=f.actor_on_lane(q['path'],q['distances'],0.,size,g.ground[sid]['plane']);frames=[]
        for i,fr in enumerate(c['frames']):
            pr=f.projection(a,fr);fail=[]
            if pr is None:fail=['near_camera_plane']
            else:
                bb=pr['box'];wh=bb[2:]-bb[:2];vw=np.maximum(0,np.minimum(bb[2:],[1024,576])-np.maximum(bb[:2],[0,0]));fraction=float(np.prod(vw)/max(1,np.prod(wh)))
                if bb[1]<8:fail.append('top')
                if bb[3]+32>=512:fail.append('ego_bottom')
                if wh[0]<72 or wh[1]<40 or not .006<=np.prod(wh)/(1024*576)<=.18:fail.append('amodal_scale_or_area')
                if fraction<.60:fail.append('visible_fraction')
            if fail:frames.append({'frame':i,'failures':fail,'box':pr['box'].tolist() if pr else None})
        reasons=sorted({w for fr in frames for w in fr['failures']})
        rows.append({**r,'projection_reasons':reasons,'horizontal_visibility_only':reasons==['visible_fraction'],'frames':frames})
    only=[r for r in rows if r['horizontal_visibility_only']]
    result={'cases':rows,'count':len(rows),'constraint_counts':dict(Counter(w for r in rows for w in r['projection_reasons'])),
        'horizontal_visibility_only':len(only),'horizontal_visibility_only_scenes':sorted({r['scene'] for r in only}),
        'new_training_admission':0,'boundary':'later collision/ground and actual silhouette quality not implied by a projected-box diagnosis'}
    f.dump(O/'projection_diagnosis.json',result)
    f.dump(O/'controller_state.json',{'stage':'complete','pid':os.getpid(),'training_steps':0})
    print({k:v for k,v in result.items() if k!='cases'},flush=True)

if __name__=='__main__':main()

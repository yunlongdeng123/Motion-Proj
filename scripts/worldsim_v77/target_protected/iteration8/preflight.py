"""先真实几何配对，再投入SAM2；有限位移集合，全局门槛不变。"""
from pathlib import Path
import sys,os,time,copy
from collections import Counter
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
from geometry_factory import Geometry,read,dump
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r8';F=O/'factory'
OFFSETS=[(z,x) for z in [2.,4.,6.,8.,10.] for x in [-8.,-6.,-4.,-2.,0.,2.,4.,6.,8.]]
def main():
    geo=Geometry(F);rows=[];total=Counter();(O/'preflight').mkdir(exist_ok=True)
    for sid,c in geo.sources.items():
        out=O/'preflight'/f'{sid}.json'
        if out.exists():rows.append(read(out));continue
        start=time.time();prov=c['reuse_provenance'];old=Path(prov['factory'])/'geometry'/f"{prov['source_id']}.json"
        # 已有地面缓存来自相同真实LiDAR/keyframe上下文；不复用历史配对结论。
        if old.exists() and (old.parent/(old.stem+'_ground.npz')).exists():
            g=read(old);g.update(source_id=sid,reused_ground_from=str(old));dump(F/'geometry'/f'{sid}.json',g)
            dst=F/'geometry'/f'{sid}_ground.npz'
            if not dst.exists():dst.symlink_to((old.parent/(old.stem+'_ground.npz')).resolve())
        reject=Counter();plans=[]
        try:
            ground=geo.prepare(sid)
            if ground['pass']:
                for z,x in OFFSETS:
                    traj,why=geo.trajectory(sid,z,x,sid,'world_offset',0)
                    if traj is None:reject[why]+=1;continue
                    if max(f['box'][3] for f in traj['frames'])+6>=512:reject['ego_band']+=1;continue
                    plans.append(traj)
            else:reject['ground_fit']+=1
        except (ValueError,IndexError,KeyError) as e:
            ground={'pass':False,'error':repr(e)};reject['source_context_engineering']+=1
        row={'source_id':sid,'scene':c['scene'],'source_actor_count':len(c['actors']),'ground':{k:v for k,v in ground.items() if not k.startswith('_')},'plans':plans,'rejections':dict(reject),'seconds':time.time()-start,'source_role':'geometric proposal only; not visual/instance QA'}
        dump(out,row);rows.append(row);total.update(reject)
        dump(O/'preflight_state.json',{'completed':len(rows),'total':len(geo.sources),'sources_with_plans':sum(bool(r['plans']) for r in rows),'scenes_with_plans':len({r['scene'] for r in rows if r['plans']}),'pid':os.getpid()})
        print('PREFLIGHT',sid,c['scene'],len(plans),'seconds',round(row['seconds'],1),flush=True)
    dump(O/'preflight_summary.json',{'sources':[{k:v for k,v in r.items() if k!='plans'} for r in rows],'sources_with_plans':sum(bool(r['plans']) for r in rows),'scenes_with_plans':len({r['scene'] for r in rows if r['plans']}),'rejections':dict(total),'offsets_m':OFFSETS})
if __name__=='__main__':main()

"""一次扩展供体候选；复用已独立合格SAM2轮廓，不更改物理门槛。"""
from pathlib import Path
import sys,os,copy
from collections import Counter
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
import numpy as np
from geometry_factory import Geometry,read,dump,view_angles,wrap
from assemble_sources import link
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r8';F=O/'factory';D=T/'r3/factory'
def main():
    geo=Geometry(F);dsrc=read(D/'source_manifest.json')['clips'];ad=read(D/'donor_admission.json')['sources']
    sq={c['source_id']:c for c in read(D/'subagent_source_reviews.json')['clips']};mq={c['source_id']:c for c in read(D/'mask_review/independent_mask_reviews.json')['clips']}
    donors=[];manifest=read(F/'source_manifest.json');known={c['source_id'] for c in manifest['clips']}
    for c in dsrc:
        sid=c['source_id']
        if len(c['frames'])!=10 or not ad.get(sid,{}).get('eligible_for_pairing') or sq.get(sid,{}).get('donor_status')!='pass' or mq.get(sid,{}).get('donor_mask_status')!='pass':continue
        cc=copy.deepcopy(c);cc['source_id']='D_'+sid;cc['reuse_provenance']={'factory':str(D),'source_id':sid,'slice':[0,10],'already_independent_donor_pass':True};donors.append(cc)
        if cc['source_id'] not in known:manifest['clips'].append(cc)
        for f in cc['frames']:link(D/'rgb'/f['filename'],F/'rgb'/f['filename'])
        link(D/'segmented'/sid,F/'segmented'/cc['source_id'])
        geo.sources[cc['source_id']]=copy.deepcopy(cc)
        for f in geo.sources[cc['source_id']]['frames']:f['_w2c']=np.linalg.inv(f['camera_to_world'])
    dump(F/'source_manifest.json',manifest);(O/'cross_preflight').mkdir(exist_ok=True)
    rows=[];offsets=[(z,x) for z in [-6.,0.,4.5,6.,8.] for x in [-6.,-3.,-1.5,0.,1.5,3.,6.]]
    for sid,c in list(geo.sources.items()):
        if not sid.startswith('N'):continue
        out=O/'cross_preflight'/f'{sid}.json'
        if out.exists():rows.append(read(out));continue
        g=geo.prepare(sid);plans=[];reject=Counter()
        if g['pass']:
            ranking=[]
            for d in donors:
                if d['scene']==c['scene']:continue
                yaw=np.mean([abs(wrap(view_angles(c['frames'][i]['actors'][0],c['frames'][i])[0]-view_angles(d['frames'][i]['actors'][0],d['frames'][i])[0])) for i in [0,5,9]])
                if yaw<=35:ranking.append((yaw,d['source_id']))
            for _,ds in sorted(ranking)[:8]:
                for z,x in offsets:
                    tr,why=geo.trajectory(sid,z,x,ds,'donor_camera_replay',0)
                    if tr is None:reject[why]+=1;continue
                    if max(f['box'][3] for f in tr['frames'])+6>=512:reject['ego_band']+=1;continue
                    plans.append(tr)
        row={'source_id':sid,'scene':c['scene'],'plans':plans,'rejections':dict(reject),'proposal_only':True};dump(out,row);rows.append(row)
        print('CROSS',sid,len(plans),flush=True)
        dump(O/'cross_state.json',{'completed':len(rows),'receivers_with_plans':sum(bool(r['plans']) for r in rows),'scenes_with_plans':len({r['scene'] for r in rows if r['plans']}),'pid':os.getpid()})
    dump(O/'cross_summary.json',{'donors':len(donors),'receivers_with_plans':sum(bool(r['plans']) for r in rows),'scenes_with_plans':len({r['scene'] for r in rows if r['plans']}),'rejections':dict(sum((Counter(r['rejections']) for r in rows),Counter()))})
if __name__=='__main__':main()

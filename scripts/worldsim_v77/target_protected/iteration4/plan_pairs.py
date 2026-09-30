"""真实LiDAR/地图/GT下先配对，再排新receiver的SAM2队列；包络不冒充mask。"""
import argparse,copy,os,sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import cv2,numpy as np
from scipy.ndimage import median_filter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import Geometry,read,dump,source_masks,view_angles,wrap
from build_pairs import warp_matrix,actor_contact
from iteration2.donor_gate import check_masks

def envelope_precheck(geo,traj,dmasks,contacts):
    """只认证候选空间门槛；B的遮挡比例必须等真实实例mask验证。"""
    c=geo.sources[traj['source_id']];d=geo.sources[traj['donor_source_id']]
    intended={a['instance_token'] for a in c['actors']};ratios={t:[] for t in intended};hits=set()
    for i,(f,df,p) in enumerate(zip(c['frames'],d['frames'],traj['frames'])):
        T=warp_matrix(df,p,contacts[i]);A=cv2.warpAffine(dmasks[i].astype('uint8'),T,(1024,576),flags=cv2.INTER_NEAREST)>0
        yy,xx=np.where(A)
        if len(yy)<200 or xx.max()-xx.min()+1<72 or yy.max()-yy.min()+1<40:return None,'actual_actor_too_small'
        H=cv2.dilate(A.astype('uint8'),np.ones((13,13),'uint8'))>0
        if H[512:].any():return None,'conservative_ego_image_band'
        rows=np.flatnonzero(A.sum(1)>=3)
        if abs(rows[-1]-p['box'][3])>3:return None,'contact_anchor_jitter'
        bytoken={ob['instance_token']:ob for ob in geo.obstacles[c['source_id']][i]}
        for tok in intended:
            ob=bytoken.get(tok);pr=ob.get('_projection') if ob else None
            if pr is None:return None,'protected_projection_missing'
            b=pr['box'];x0,y0=np.maximum(np.floor(b[:2]).astype(int),0);x1,y1=np.minimum(np.ceil(b[2:]).astype(int),[1024,576])
            area=max(1,(x1-x0)*(y1-y0));hit=int(H[y0:y1,x0:x1].sum())
            ratios[tok].append(hit/area)
            if hit>max(12,.01*int(H.sum())):
                hits.add(tok)
                if p['depth_interval'][1]>.3+pr['near_depth']:return None,'depth_order_uncertain'
        for tok,ob in bytoken.items():
            pr=ob['_projection']
            if pr is None or tok in intended:continue
            b=pr['box'];x0,y0=np.maximum(np.floor(b[:2]).astype(int),0);x1,y1=np.minimum(np.ceil(b[2:]).astype(int),[1024,576])
            if x1>x0 and y1>y0 and H[y0:y1,x0:x1].sum()>max(12,.01*int(H.sum())):return None,'additional_unreviewed_envelope'
    active=[t for t,v in ratios.items() if max(v)>.01]
    # 这里只以GT包络做有希望的候选排序，不能宣称真实实例遮挡比例通过。
    if not active:kind='background'
    elif len(active)==1:
        v=np.array(ratios[active[0]])
        if v.min()<.12 or v.max()>.75:return None,'weak_or_excessive_envelope_overlap'
        kind='single_actor'
    else:
        v=np.array([ratios[t] for t in active])
        if v.min()<.10 or v.max()>.70:return None,'weak_dense_envelope_overlap'
        kind='dense_actors'
    score=min([min(ratios[t]) for t in active] or [0])
    return {'planned_type':kind,'required_protected_instances':sorted(intended),
            'overlapped_envelopes':active,'envelope_hole_fraction':ratios,'ranking_score':score,
            'exact_protected_occlusion_verified':False,'render_allowed':False,'training_ready':False},None

def main(root,donor_root,ready_only=False,distinct_instances=False):
    cv2.setNumThreads(1);factory=root/'native10_factory';geo=Geometry(factory)
    receivers=list(geo.sources)
    old={c['source_id']:c for c in read(donor_root/'source_manifest.json')['clips']}
    sq={r['source_id']:r for r in read(donor_root/'subagent_source_reviews.json')['clips']}
    mq={r['source_id']:r for r in read(donor_root/'mask_review/independent_mask_reviews.json')['clips']}
    numeric={r['source_id']:r['numeric_gate_pass'] for r in read(donor_root/'mask_review/mask_audit.json')['clips']}
    adm=read(donor_root/'donor_admission.json')['sources']
    donors=[s for s in old if len(old[s]['frames'])==10 and numeric.get(s) and sq.get(s,{}).get('donor_status')=='pass' and mq.get(s,{}).get('donor_mask_status')=='pass' and adm.get(s,{}).get('eligible_for_pairing')]
    for sid in donors:
        geo.sources[sid]=old[sid]
        for f in old[sid]['frames']:f['_w2c']=np.linalg.inv(f['camera_to_world'])
    @lru_cache(maxsize=8)
    def donor_data(sid):
        masks=source_masks(donor_root,sid);assert check_masks(old[sid]['actors'][0]['instance_token'],masks)['eligible_for_pairing']
        contacts=median_filter(np.array([actor_contact(m,np.array(f['actors'][0]['projection']['box_xyxy'])) for m,f in zip(masks,old[sid]['frames'])]),size=5,mode='nearest')
        return masks,contacts
    offsets=[[z,x] for z in [-6.,0.,4.5,6.,8.] for x in [-6.,-3.,-1.5,0.,1.5,3.,6.]]
    config={'run_id':'r4','placement_mode':'donor_camera_replay','offsets_m':offsets,'max_ranked_donors':6,
            'geometry_thresholds':'unchanged r3','protected_envelopes':'planning only; exact SAM2 overlap not available',
            'cpu_only':True,'actual_synthetic_cases':0,'donor_root':str(donor_root)}
    if distinct_instances:config['donor_selection']='best yaw-ranked window per distinct instance; still at most six donors'
    if (root/'planner_config.json').exists():assert read(root/'planner_config.json')==config,'配置已变，须新run，不能混用旧缓存'
    dump(root/'planner_config.json',config)
    chosen=[];logs=[];total=Counter();(root/'plans').mkdir(exist_ok=True)
    for sid in receivers:
        required=[f['filename'] for f in geo.sources[sid]['frames']]+[geo.context[sid]['frames'][i]['sensors']['LIDAR_TOP']['filename'] for i in [0,3,6]]
        missing=[n for n in required if not (factory/'rgb'/n).is_file()]
        if missing:
            if ready_only:continue
            raise FileNotFoundError(f'{sid}: missing {len(missing)} actual source files')
        saved=root/'plans'/f'{sid}.json'
        if saved.exists():
            row=read(saved);logs.append(row);total.update(row['rejection_counts']);chosen+=row['chosen'];continue
        c=geo.sources[sid];reject=Counter();ranked=[];best={};attempts=0
        if any(len(f['actors'])!=len(c['actors']) or any(a['projection'] is None for a in f['actors']) for f in c['frames']):
            reject['receiver_projection_missing']+=1;ground={'pass':False,'reason':'receiver_projection_missing'}
        else:
            ground=geo.prepare(sid)
            if not ground['pass']:reject['real_lidar_ground_fail']+=1
            else:
                for ds in donors:
                    d=old[ds]
                    if d['scene']==c['scene'] or d['actors'][0]['instance_token']==c['actors'][0]['instance_token']:continue
                    yaw=np.mean([abs(wrap(view_angles(c['frames'][i]['actors'][0],c['frames'][i])[0]-view_angles(d['frames'][i]['actors'][0],d['frames'][i])[0])) for i in [0,5,9]])
                    if yaw<=35:ranked.append((yaw,ds))
                order=sorted(ranked)
                if distinct_instances:
                    seen=set();deduplicated=[]
                    for rank,ds in order:
                        tok=old[ds]['actors'][0]['instance_token']
                        if tok in seen:continue
                        seen.add(tok);deduplicated.append((rank,ds))
                    order=deduplicated
                for _,ds in order[:6]:
                    masks,contacts=donor_data(ds)
                    for lo,la in offsets:
                        attempts+=1;traj,why=geo.trajectory(sid,lo,la,ds,'donor_camera_replay',0)
                        if traj is None:reject[why]+=1;continue
                        q,why=envelope_precheck(geo,traj,masks,contacts)
                        if q is None:reject[why]+=1;continue
                        typ=q['planned_type'];p=traj|q
                        if typ not in best or p['ranking_score']>best[typ]['ranking_score']:best[typ]=p
        # 每receiver仅一例进入GPU前检，优先密集；不靠同背景多offset凑scene数。
        pick=next((best[t] for t in ['dense_actors','single_actor','background'] if t in best),None)
        picks=[pick] if pick else []
        for p in picks:p.update(case_id=sid,human_verdict=None,independent_source_review='pending')
        row={'source_id':sid,'scene':c['scene'],'camera':c['camera'],'metadata_actor_count':len(c['actors']),
             'ground':{k:v for k,v in ground.items() if not k.startswith('_')},'ranked_donors':len(ranked),'attempted_placements':attempts,
             'rejection_counts':dict(reject),'possible_types':list(best),'chosen':picks}
        dump(saved,row);logs.append(row);chosen+=picks;total.update(reject)
        print('PLANNED',sid,c['scene'],'ground',ground['pass'],'chosen',pick['planned_type'] if pick else None,'reasons',dict(reject),flush=True)
        dump(root/'planner_progress.json',{'processed':len(logs),'planned':len(chosen),'rejection_counts':dict(total)})
    summary={'receiver_windows':len(logs),'total_receiver_windows':len(receivers),'complete':len(logs)==len(receivers),'receiver_scenes':len({r['scene'] for r in logs}),'planned_cases':len(chosen),
             'planned_scenes':len({geo.sources[p['source_id']]['scene'] for p in chosen}),
             'planned_types':dict(Counter(p['planned_type'] for p in chosen)),'real_lidar_ground_pass':sum(r['ground']['pass'] for r in logs),
             'rejection_counts':dict(total),'actual_synthetic_cases':0,'training_ready':0,'GPU_model_calls':0}
    dump(root/'pair_plan.json',{'config':config,'summary':summary,'plans':chosen});print('COMPLETE',summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--donor-root',type=Path,required=True);p.add_argument('--ready-only',action='store_true');p.add_argument('--distinct-instances',action='store_true');a=p.parse_args();main(a.root,a.donor_root,a.ready_only,a.distinct_instances)

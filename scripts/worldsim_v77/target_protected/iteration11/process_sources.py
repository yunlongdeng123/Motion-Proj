"""按投影过程选新来源，以端点视差拟合道路方向速度；不看模型输出。"""
from pathlib import Path
import os,sys,math,copy,time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(P/'iteration9'))
from temporal_factory import geometry,protections,trajectory,exact,old,R8,F,T,read,dump
from temporal_metrics import process
from geometry_factory import footprint,ground_orientation,projection
from pyquaternion import Quaternion
import numpy as np,cv2
from collections import Counter,defaultdict
O=T/'r11'

def b_at(c,t,i):return next(a for a in c['frames'][i]['actors'] if a['instance_token']==t)

def anchor_at(geo,sid,t,d,x):
    c=geo.sources[sid];b=b_at(c,t,5);f=c['frames'][5]
    camera=np.asarray(f['camera_to_world'])[:3,3]
    toward=camera[:2]-np.asarray(b['translation'])[:2];toward/=np.linalg.norm(toward)
    right=np.array([-toward[1],toward[0]]);xy=np.asarray(b['translation'])[:2]+d*toward+x*right
    plane=np.array(geo.ground[sid]['plane']);r=Quaternion(b['rotation']).rotation_matrix
    yaw=math.atan2(r[1,0],r[0,0]);R=ground_orientation(yaw,plane)
    size=[1.85,4.5,1.5];center=np.r_[xy,np.dot(np.r_[xy,1],plane)]+R[:,2]*size[2]/2
    a={'translation':center.tolist(),'rotation':Quaternion(matrix=R).elements.tolist(),'size':size}
    foot=footprint(a)
    if not geo.road.covers(foot):return None,'mid_outside_road'
    if any(foot.distance(o['_foot'])<.3 for o in geo.obstacles[sid][5]):return None,'mid_collision'
    if geo.ground[sid]['_tree'].query(np.array(foot.exterior.coords)[:4])[0].max()>2.5:return None,'mid_ground'
    poses=[{'actor':a} for _ in range(10)]
    return {'source_id':sid,'frames':poses,'asset':'sedan','offset_longitudinal_m':d,'offset_lateral_m':x,'protected_anchor_token':t},None

def sweep_proxy(c,anchor,t,v):
    a=copy.deepcopy(anchor['frames'][5]['actor']);r=Quaternion(a['rotation']).rotation_matrix
    direction=r[:2,0];origin=np.array(a['translation']);mid=c['frames'][5]['timestamp'];vals=[]
    for i in [0,9]:
        f=c['frames'][i];aa=copy.deepcopy(a);aa['translation']=(origin+np.r_[direction*v*((f['timestamp']-mid)/1e6),0]).tolist()
        q=projection(aa,f);box=np.asarray(b_at(c,t,i)['projection']['box_xyxy'])
        if q is None:return None
        vals.append(((q['box'][0]+q['box'][2])/2-box[0])/(box[2]-box[0]))
    return vals[1]-vals[0]

def fitted_speeds(c,anchor,t):
    zero=sweep_proxy(c,anchor,t,0);one=sweep_proxy(c,anchor,t,1)
    speeds=[0.]
    if zero is not None and one is not None and abs(one-zero)>.015:
        for target in [-.5,.5]:
            v=(target-zero)/(one-zero)
            if .25<=abs(v)<=8:speeds.append(float(round(v,3)))
    return speeds

def select_sources(geo):
    path=O/'source_selection.json'
    if path.exists():return read(path)['sources']
    used={p.stem for name in ['temporal_candidates','expansion_candidates'] for p in (T/'r9'/name).glob('*.json')}
    train={c['receiver_scene'] for r in ['r8','r10'] for c in read(T/r/'dataset_catalog.json')['cases'] if c['split']=='train'}
    train|=set(read(T/'r8/source_split.json')['old_r7_training_scenes'])
    val={c['receiver_scene'] for r in ['r8','r10'] for c in read(T/r/'dataset_catalog.json')['cases'] if c['split']=='validation'}
    rows=[]
    for sid,c in geo.sources.items():
        if not sid.startswith('N') or sid in used or sid not in geo.context or c['scene'] in val:continue
        pm=protections(geo,sid)
        for t in pm:
            bs=[b_at(c,t,i) for i in range(10)];boxes=np.array([b['projection']['box_xyxy'] for b in bs]);wh=boxes[:,2:]-boxes[:,:2]
            if np.median(wh[:,0])<90:continue
            center=(boxes[:,:2]+boxes[:,2:])/2;pixel=float(np.linalg.norm(center[-1]-center[0]))
            e=np.array([f['camera_to_world'] for f in c['frames']])[:,:3,3];ego=float(np.linalg.norm(np.diff(e,axis=0),axis=1).sum())
            bpath=float(np.linalg.norm(np.diff(np.array([b['translation'] for b in bs]),axis=0),axis=1).sum())
            if max(ego,bpath)<.5:continue
            rows.append({'source_id':sid,'scene':c['scene'],'camera':c['camera'],'B':t,'prior_train':c['scene'] in train,'ego_path_m':ego,'B_path_m':bpath,'B_centroid_change_px':pixel,'priority':pixel+12*ego,'near_static_ego_moving_B':ego<.2 and bpath>.5})
    chosen=[];scenes=set()
    # 各16个新验证可能世界和旧训练世界；世界内选实际投影变化最大的新窗。
    for blocked in [False,True]:
        subset=sorted([r for r in rows if r['prior_train']==blocked],key=lambda r:(not r['near_static_ego_moving_B'],-r['priority'],r['source_id'],r['B']))
        n=0
        for r in subset:
            if r['scene'] in scenes or n>=16:continue
            chosen.append(r);scenes.add(r['scene']);n+=1
    dump(path,{'sources':chosen,'candidate_rows':rows,'budget':32,'prior_train_worlds':sorted(train),'prior_validation_worlds_excluded':sorted(val),'new_to_r9_factory':True,'uses_model_output':False,'rule':'one new window/world; near-static ego + moving B priority, then actual B image change +12*ego path; median B width>=90px'})
    return chosen

def main(shard=0,count=1):
    cv2.setNumThreads(1);O.mkdir(exist_ok=True);geo=geometry();sources=select_sources(geo);mesh=dict(np.load(R8/'assets/sedan.npz'));out=O/'planned';out.mkdir(exist_ok=True)
    for row in sources[shard::count]:
        sid=row['source_id'];dest=out/(sid+'.json')
        if dest.exists():continue
        start=time.time();g=geo.prepare(sid);pm=protections(geo,sid);reject=Counter();chosen=defaultdict(list)
        if g['pass'] and row['B'] in pm:
            c=geo.sources[sid]
            # 仅九个前后/横向中点，世界速度由当前相机端点推算，不继续旧来源位置网格。
            for d in [6.,8.,10.]:
                for x in [0.,-1.75,1.75]:
                    anchor,why=anchor_at(geo,sid,row['B'],d,x)
                    if anchor is None:reject[why]+=1;continue
                    for speed in fitted_speeds(c,anchor,row['B']):
                        tr,why=trajectory(geo,anchor,speed)
                        if tr is None:reject[why]+=1;continue
                        aa=[old.silhouette(mesh['vertices'],mesh['faces'],p['actor'],f) for p,f in zip(tr['frames'],c['frames'])]
                        q,why=exact(geo,tr,aa,pm)
                        if q is None:reject[why]+=1;continue
                        active={t:pm[t] for t in q['protected_instances']};ba={t:[b_at(c,t,i) for i in range(10)] for t in active}
                        hs=[cv2.dilate(a.astype('uint8'),np.ones((7,7),'uint8'))>0 for a in aa];proc=process(c['frames'],[p['actor'] for p in tr['frames']],hs,active,ba)
                        family='sweep_B' if proc['sweep_over_any_B'] else 'static_A_moving_ego' if proc['static_A_moving_ego_image_change'] else 'visibility_transition' if proc['visibility_transition_any_B'] else None
                        if family is None:reject['no_actual_process']+=1;continue
                        if len(chosen[family])>=2:continue
                        chosen[family].append(tr|q|{'process_family':family,'temporal_process':proc,'edge_mode':'near_hard','mask_dilation_px':3,'human_verdict':None,'source_process_priority':row,'velocity_method':'endpoint projected-centre derivative fit; actual SAM2 sweep measured afterwards'})
        pp=[p for v in chosen.values() for p in v]
        dump(dest,{'source':row,'candidates':pp,'counts':dict(Counter(p['process_family'] for p in pp)),'rejects':dict(reject),'seconds':time.time()-start,'ground_pass':g['pass']})
        print(sid,len(pp),dict(Counter(p['process_family'] for p in pp)),dict(reject),flush=True)
    allrows=[read(p) for p in sorted(out.glob('*.json'))];pp=[c for r in allrows for c in r['candidates']]
    dump(O/f'pool_{shard}.json',{'complete_shard':True,'candidates':pp,'counts':dict(Counter(c['process_family'] for c in pp)),'source_completed':len(allrows),'sources_expected':len(sources)})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--count',type=int,default=1);a=p.parse_args();main(a.shard,a.count)

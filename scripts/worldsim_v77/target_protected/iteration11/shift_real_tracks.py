"""真实同世界轨迹的时间平移产生虚拟遮挡；Y保持当前真实视频。"""
from pathlib import Path
import sys,os,math,bisect,time,copy
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(P/'iteration9'))
from temporal_factory import geometry,protections,exact,old,R8,F,T,read,dump
from temporal_metrics import process
from geometry_factory import footprint,ground_orientation,projection,wrap,transform
from pyquaternion import Quaternion
import numpy as np,cv2
from collections import Counter,defaultdict
from scipy.spatial import cKDTree
O=T/'r12'

def recover_ground_support(geo,sid):
    """平面不变；近地原始返回也能提供支持，不能整车框删掉地面。"""
    dest=O/'ground_support_recovery';dest.mkdir(exist_ok=True);path=dest/(sid+'.npz')
    g=geo.ground[sid]
    if path.exists():points=np.load(path)['points']
    else:
        clouds=[];plane=np.asarray(g['plane']);anchor=np.array([f['actors'][0]['translation'] for f in geo.sources[sid]['frames']]);low=anchor[:,:2].min(0)-16;high=anchor[:,:2].max(0)+16
        for k in [0,3,6]:
            d=geo.context[sid]['frames'][k]['sensors']['LIDAR_TOP'];raw=np.fromfile(F/'rgb'/d['filename'],np.float32).reshape(-1,5)[:,:3];cal=d['calibrated_sensor'];ego=d['ego_pose'];m=transform(ego['translation'],ego['rotation'])@transform(cal['translation'],cal['rotation']);pts=raw@m[:3,:3].T+m[:3,3]
            height=pts[:,2]-np.c_[pts[:,:2],np.ones(len(pts))]@plane
            clouds.append(pts[(abs(height)<=.08)&np.all((pts[:,:2]>=low)&(pts[:,:2]<=high),axis=1)])
        points=np.concatenate(clouds);np.savez_compressed(path,points=points)
    assert len(points)>0;g['_tree']=cKDTree(points[:,:2]);g['support_evidence_recovery']={'raw_scans':[0,3,6],'near_plane_band_m':.08,'plane_unchanged':True,'support_threshold_unchanged_m':2.5,'points':len(points),'cache':str(path),'support_is_not_empty_space_certification':True}

def interpolate_track(c,ctx,tok,t):
    stamps=c['keyframe_timestamps'];j=bisect.bisect_right(stamps,t)
    if not 0<j<len(stamps):return None
    a=next((a for a in ctx[j-1]['annotations'] if a['instance_token']==tok),None)
    b=next((a for a in ctx[j]['annotations'] if a['instance_token']==tok),None)
    if a is None or b is None or a['category']!='vehicle.car' or b['category']!='vehicle.car':return None
    w=(t-stamps[j-1])/(stamps[j]-stamps[j-1])
    return {'translation':((1-w)*np.asarray(a['translation'])+w*np.asarray(b['translation'])).tolist(),'rotation':Quaternion.slerp(Quaternion(a['rotation']),Quaternion(b['rotation']),amount=w).elements.tolist(),'size':a['size']}

def shifted(geo,sid,tok,delta):
    c=geo.sources[sid];ctx=geo.context[sid]['frames'];plane=np.asarray(geo.ground[sid]['plane']);rows=[];min_gap=1e9;max_support=0;ground_correction=0;prev=None
    for f,obs in zip(c['frames'],geo.obstacles[sid]):
        a=interpolate_track(c,ctx,tok,f['timestamp']+round(delta*1e6))
        if a is None:return None,'shifted_track_missing'
        pos=np.asarray(a['translation']);rr=Quaternion(a['rotation']).rotation_matrix;yaw=math.atan2(rr[1,0],rr[0,0]);R=ground_orientation(yaw,plane);size=a['size']
        ground=float(np.dot(np.r_[pos[:2],1],plane));correction=abs(pos[2]-size[2]/2-ground)
        if correction>.35:return None,'GT_ground_inconsistent'
        ground_correction=max(ground_correction,correction);pos=np.r_[pos[:2],ground]+R[:,2]*size[2]/2
        a={'translation':pos.tolist(),'rotation':Quaternion(matrix=R).elements.tolist(),'size':size};foot=footprint(a)
        if not geo.road.covers(foot):return None,'outside_mapped_road'
        support=geo.ground[sid]['_tree'].query(np.asarray(foot.exterior.coords)[:4])[0].max();max_support=max(max_support,float(support))
        if support>2.5:return None,'ground_support_gap'
        p=projection(a,f)
        if p is None:return None,'behind_camera'
        bb=p['box'];wh=bb[2:]-bb[:2]
        if min(bb[0],bb[1],1024-bb[2],576-bb[3])<8 or wh[0]<72 or wh[1]<40 or not .006<=np.prod(wh)/(1024*576)<=.18:return None,'size_border'
        if bb[3]+6>=512:return None,'ego_band'
        for ob in obs:
            gap=foot.distance(ob['_foot']);min_gap=min(min_gap,gap)
            if gap<.3:return None,'current_actor_collision'
        if prev:
            fac=100000/(f['timestamp']-prev['timestamp']);scale=(wh/prev['wh'])**fac
            if np.linalg.norm(pos-prev['pos'])*fac>2 or abs(wrap(math.degrees(yaw-prev['yaw'])))*fac>5 or scale.min()<.85 or scale.max()>1.18:return None,'trajectory_continuity'
        rows.append({'frame':f['frame'],'timestamp':f['timestamp'],'actor':a,'box':bb.tolist(),'depth_interval':[p['near_depth'],p['far_depth']],'real_track_time_us':f['timestamp']+round(delta*1e6)})
        prev={'timestamp':f['timestamp'],'pos':pos,'yaw':yaw,'wh':wh}
    path=np.array([p['actor']['translation'] for p in rows]);duration=(rows[-1]['timestamp']-rows[0]['timestamp'])/1e6;speed=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()/duration)
    if speed<.5:return None,'stationary_shift_occupied'
    return {'source_id':sid,'scene':c['scene'],'frames':rows,'asset':'suv' if rows[5]['actor']['size'][2]>=1.65 else 'sedan','min_GT_clearance_m':float(min_gap),'max_ground_support_distance_m':max_support,'ground':{k:v for k,v in geo.ground[sid].items() if not k.startswith('_')},'trajectory_policy':'same_world_observed_real_track_time_shift','time_shift_s':delta,'real_track_instance':tok,'max_GT_bottom_ground_correction_m':ground_correction,'speed_signed_mps':speed,'world_velocity_xy_mps':((path[-1,:2]-path[0,:2])/duration).tolist(),'anchor_offsets_m':[None,None]},None

def main(shard=0,count=1):
    cv2.setNumThreads(1);O.mkdir(exist_ok=True);geo=geometry();sources=read(T/'r11/source_selection.json')['sources'];assets={n:dict(np.load(R8/'assets'/(n+'.npz'))) for n in ['sedan','suv']};out=O/'planned_ground_recovered';out.mkdir(exist_ok=True)
    for source in sources[shard::count]:
        sid=source['source_id'];dest=out/(sid+'.json')
        if dest.exists():continue
        start=time.time();g=geo.prepare(sid);reject=Counter();chosen=defaultdict(list);pm=protections(geo,sid)
        if g['pass'] and pm:
            recover_ground_support(geo,sid)
            c=geo.sources[sid];ctx=geo.context[sid]['frames'];tokens=sorted({a['instance_token'] for f in ctx for a in f['annotations'] if a['category']=='vehicle.car'})
            # 真实有限时间支撑内四个固定偏移；不为过关改速度或空间位置。
            for tok in tokens:
                for delta in [-1.,-.5,.5,1.]:
                    tr,why=shifted(geo,sid,tok,delta)
                    if tr is None:reject[why]+=1;continue
                    mesh=assets[tr['asset']];aa=[old.silhouette(mesh['vertices'],mesh['faces'],p['actor'],f) for p,f in zip(tr['frames'],c['frames'])];q,why=exact(geo,tr,aa,pm)
                    if q is None:reject[why]+=1;continue
                    active={t:pm[t] for t in q['protected_instances']};ba={t:[next(a for a in f['actors'] if a['instance_token']==t) for f in c['frames']] for t in active};hs=[cv2.dilate(a.astype('uint8'),np.ones((7,7),'uint8'))>0 for a in aa];proc=process(c['frames'],[p['actor'] for p in tr['frames']],hs,active,ba)
                    family='sweep_B' if proc['sweep_over_any_B'] else 'visibility_transition' if proc['visibility_transition_any_B'] else 'moving_A_near_static_ego' if proc['ego_camera_motion']['path_m']<.2 and proc['hole_centroid_diameter_px']>=50 else None
                    if family is None:reject['no_new_actual_process']+=1;continue
                    if len(chosen[family])>=2:continue
                    chosen[family].append(tr|q|{'process_family':family,'temporal_process':proc,'edge_mode':'near_hard','mask_dilation_px':3,'human_verdict':None,'source_process_priority':source})
        pp=[p for v in chosen.values() for p in v];dump(dest,{'source':source,'candidates':pp,'counts':dict(Counter(p['process_family'] for p in pp)),'rejects':dict(reject),'seconds':time.time()-start,'ground_pass':g['pass']});print(sid,len(pp),dict(Counter(p['process_family'] for p in pp)),flush=True)
    allrows=[read(p) for p in sorted(out.glob('*.json'))];pp=[c for r in allrows for c in r['candidates']];dump(O/f'pool_ground_recovered_{shard}.json',{'complete_shard':True,'candidates':pp,'counts':dict(Counter(p['process_family'] for p in pp)),'source_completed':len(allrows),'sources_expected':len(sources)})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--count',type=int,default=1);a=p.parse_args();main(a.shard,a.count)

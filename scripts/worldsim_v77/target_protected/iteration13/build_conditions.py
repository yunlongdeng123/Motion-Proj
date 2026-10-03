"""CPU粗语义几何条件：车辆框内核O，实测非actor背景返回N，其余U。"""
from common import *
import numpy as np, cv2
from PIL import Image
from geometry_factory import transform
from actor_state import Observation, cuboid_front_depth, project_world
cv2.setNumThreads(1)

def build(c):
    x=images(c,'rgb');holes=images(c,'hole')>0
    obs=[Observation(np.where(h[...,None],127,im).astype('uint8'),h,np.array(fr['camera_to_world']),
        np.array(fr['intrinsics_1024']),fr['timestamp']) for im,h,fr in zip(x,holes,c['frames'])]
    retained=[[a for a in fr['actors'] if a['instance_token']!=c['target_token']] for fr in c['frames']]
    front=[cuboid_front_depth(o,a) for o,a in zip(obs,retained)]
    sourcefront=[cuboid_front_depth(o,fr['actors']) for o,fr in zip(obs,c['frames'])]
    background=[];provenance=[]
    for scan in c['scans']:
        p=scan.get('path')
        if not p or not Path(p).is_file():continue
        raw=np.fromfile(p,np.float32).reshape(-1,5)[:,:3];cal=scan['calibrated_sensor'];ego=scan['ego_pose']
        m=transform(ego['translation'],ego['rotation'])@transform(cal['translation'],cal['rotation'])
        world=raw@m[:3,:3].T+m[:3,3]
        i=min(range(len(obs)),key=lambda i:abs(obs[i].timestamp-scan['timestamp']))
        # 避免把不同时间的道路返回当成当前已经看见。
        if abs(obs[i].timestamp-scan['timestamp'])>100000 or not c['frames'][i]['actors']:continue
        # 条件只描述背景/车辆，不预测路面高度；墙、护栏等实测静态返回也是背景。
        # 没有实测返回仍为U，绝不把GT框外整片刷成N。
        pts=world[np.linalg.norm(world-obs[i].camera_to_world[:3,3],axis=1)<40]
        uv,z,good=project_world(pts,obs[i])
        ids=np.flatnonzero(good);u,v=uv[ids].T
        blocked=cv2.dilate(np.isfinite(sourcefront[i]).astype('uint8'),np.ones((5,5),'uint8'))>0
        ids=ids[(~holes[i][v,u])&(~blocked[v,u])]
        # 相机像素下只保留最近实测返回，避免累加其后的不可见点。
        order=np.argsort(z[ids],kind='stable');ids=ids[order];flat=uv[ids,1]*1024+uv[ids,0]
        _,first=np.unique(flat,return_index=True);ids=ids[first]
        background.append((pts[ids],scan['timestamp']))
        provenance.append({'filename':scan['filename'],'timestamp':scan['timestamp'],'source_frame':i,
                           'visible_background_points':len(ids),'source_actor_envelopes_dilated_px':2})
    dest=O/'conditions'/c['case_id'];dest.mkdir(parents=True,exist_ok=True);stats=[]
    for i,(ob,depth) in enumerate(zip(obs,front)):
        # 整个3D框投影包含可见道路/车身外角。只给各实例的保守内核，边缘U。
        occupied=np.zeros_like(holes[i])
        for actor in retained[i]:
            if not actor['category'].startswith('vehicle.'):continue
            cd=cuboid_front_depth(ob,[actor]);mask=np.isfinite(cd)&(cd<=depth+.05)
            yy,xx=np.where(mask)
            if not len(yy):continue
            margin=max(2.,.25*min(np.ptp(xx)+1,np.ptp(yy)+1))
            inner=cv2.distanceTransform(np.pad(mask.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]>margin
            occupied|=inner
        n=np.zeros_like(occupied);confidence=np.zeros_like(depth);confidence[occupied]=.5
        for pts,stamp in background:
            if not c['frames'][i]['actors']:continue # annotation边界外不能当空世界
            uv,z,good=project_world(pts,ob);ids=np.flatnonzero(good);u,v=uv[ids].T
            # 所有保留实体包络均阻止N，包含非车辆。没有返回的区域仍为U。
            blocked=cv2.dilate(np.isfinite(depth).astype('uint8'),np.ones((5,5),'uint8'))>0
            ids=ids[~blocked[v,u]];u,v=uv[ids].T
            n[v,u]=True;confidence[v,u]=np.maximum(confidence[v,u],.8*np.exp(-abs(ob.timestamp-stamp)/3e6))
        n&=~occupied;u=~(occupied|n);confidence[u]=0
        assert not np.any(occupied&n) and np.all(occupied|n|u)
        np.savez_compressed(dest/f'{i:05}.npz',O=occupied,N=n,U=u,Q=confidence)
        h=holes[i];stats.append({'frame':i,'H':int(h.sum()),'O_H':int((occupied&h).sum()),'N_H':int((n&h).sum()),'U_H':int((u&h).sum())})
        if i in (0,5,9):
            color=np.zeros_like(x[i])+40;color[occupied]=[45,210,110];color[n]=[45,130,250]
            contours,_=cv2.findContours(h.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color,contours,-1,(255,195,45),2) # 仅审核图，不进入条件tensor
            Image.fromarray(np.concatenate([ob.rgb,color],1)).save(dest/f'review_{i:02}.jpg',quality=90)
    result={'case_id':c['case_id'],'frames':len(obs),'statistics':stats,'lidar':provenance,
            'condition_source':'GT retained cuboid conservative inner core Q=.5; observed in-window non-actor LiDAR returns Q<=.8',
            'O_margin':'per-instance distance to envelope boundary >25% of its shorter visible extent, minimum2px; no per-case tuning',
            'unknown_geometry_frames':[i for i,fr in enumerate(c['frames']) if not fr['actors']],
            'Y_read':False,'no_appearance_condition':True,'vehicle_shape_is_proxy':True,
            'missing_lidar':sum(not d.get('path') for d in c['scans']),'human_verdict':None}
    dump(dest/'result.json',result);return result

def main():
    manifest=read(O/'manifest.json');results=[];pending=[]
    for c in manifest['cases']:
        if any(not d.get('path') or not Path(d['path']).is_file() for d in c['scans']):
            pending.append(c['case_id']);continue
        p=O/'conditions'/c['case_id']/'result.json'
        results.append(read(p) if p.exists() else build(c))
        dump(O/'condition_state.json',{'stage':'building','completed':len(results),'total':len(manifest['cases']),'pid':os.getpid()})
        print('CONDITION',c['case_id'],flush=True)
    dump(O/'condition_summary.json',{'cases':results,'model_value_not_tested':True})
    dump(O/'condition_state.json',{'stage':'waiting_lidar' if pending else 'CPU_conditions_complete',
        'completed':len(results),'pending':pending,'human_verdict':None})

if __name__=='__main__':main()

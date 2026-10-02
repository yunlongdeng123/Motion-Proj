"""数据工厂的真实相机/GT几何/地图/LiDAR支持；不生成GT，不以框充当mask。"""
from __future__ import annotations
import json,math,bisect
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union
from shapely.prepared import prep
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

def read(p):return json.loads(Path(p).read_text())
def dump(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def transform(t,q):
    m=np.eye(4);m[:3,:3]=Quaternion(q).rotation_matrix;m[:3,3]=t;return m
def wrap(v):return (v+180)%360-180
def footprint(a):
    w,l,h=a['size'];r=Quaternion(a['rotation']).rotation_matrix
    corners=np.array([[l/2,w/2,-h/2],[l/2,-w/2,-h/2],[-l/2,-w/2,-h/2],[-l/2,w/2,-h/2]])@r.T+np.array(a['translation'])
    return Polygon(corners[:,:2])
def projection(a,frame,clip_near=False):
    b=Box(a['translation'],a['size'],Quaternion(a['rotation']))
    w2c=frame['_w2c'];p=w2c[:3,:3]@b.corners()+w2c[:3,3:4]
    if p[2].min()<=.5:
        if not clip_near or p[2].max()<=.5:return None
        pts=[p[:,i] for i in range(8) if p[2,i]>.5]
        for i,j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            if (p[2,i]>.5)!=(p[2,j]>.5):pts.append(p[:,i]+(p[:,j]-p[:,i])*((.5-p[2,i])/(p[2,j]-p[2,i])))
        p=np.stack(pts,axis=1)
    pix=np.array(frame['intrinsics_1024'])@p;pix=(pix[:2]/pix[2]).T
    bb=np.r_[pix.min(0),pix.max(0)]
    return {'box':bb,'corners':pix,'near_depth':float(p[2].min()),'far_depth':float(p[2].max()),'center_depth':float((w2c@np.r_[a['translation'],1])[2])}
def view_angles(a,frame):
    v=(np.array(frame['camera_to_world'])[:3,3]-np.array(a['translation']))@Quaternion(a['rotation']).rotation_matrix
    return np.array([math.degrees(math.atan2(v[1],v[0])),math.degrees(math.atan2(v[2],np.hypot(v[0],v[1])))])
def ground_orientation(yaw,plane):
    n=np.array([-plane[0],-plane[1],1.]);n/=np.linalg.norm(n)
    f=np.array([math.cos(yaw),math.sin(yaw),0.]);f-=n*np.dot(f,n);f/=np.linalg.norm(f)
    return np.column_stack([f,np.cross(n,f),n])
def source_masks(root,sid,secondary=None):
    import cv2
    folder=root/'segmented'/sid if secondary is None else root/'segmented_secondary'/secondary
    paths=sorted((folder/'sam2_raw').glob('*.png'))
    if not paths:raise ValueError(f'no real masks: {folder}')
    return [cv2.imread(str(p),0)>0 for p in paths]

class Geometry:
    def __init__(self,root,include_parking=False):
        self.root=Path(root);self.sources={c['source_id']:c for c in read(self.root/'source_manifest.json')['clips']}
        self.context={c['source_id']:c for c in read(self.root/'source_context.json')['clips']}
        for c in self.sources.values():
            for f in c['frames']:f['_w2c']=np.linalg.inv(f['camera_to_world'])
        self.locations={sid:c['location'] for sid,c in self.sources.items()}
        self.roads={};self.map_notes={}
        for location in sorted(set(self.locations.values())):
            if location not in {'boston-seaport','singapore-hollandvillage','singapore-onenorth','singapore-queenstown'}:
                raise ValueError(f'未知nuScenes地图: {location}')
            raw=read(self.root/'maps/expansion'/f'{location}.json');nodes={n['token']:(n['x'],n['y']) for n in raw['node']}
            polygons={p['token']:Polygon([nodes[t] for t in p['exterior_node_tokens']],[[nodes[t] for t in hole['node_tokens']] for hole in p['holes']]) for p in raw['polygon']}
            tokens=set();missing=[]
            for layer in ['drivable_area']+(['carpark_area'] if include_parking else []):
                for row in raw.get(layer,[]):
                    refs=row['polygon_tokens'] if layer=='drivable_area' else [row['polygon_token']]
                    for token in refs:
                        # 官方v1.3 Holland有两条[null]道路记录：无几何不能提供道路支持。
                        # 仅忽略显式null并记账；非null的悬空引用是输入损坏，必须报错。
                        if token is None:missing.append({'layer':layer,'record':row['token']});continue
                        if token not in polygons:raise ValueError(f'地图polygon引用缺失: {location}/{layer}/{token}')
                        tokens.add(token)
            if not tokens:raise ValueError(f'地图没有可用道路polygon: {location}')
            self.map_notes[location]={'explicit_null_polygon_records':missing,'usable_polygon_count':len(tokens)}
            self.roads[location]=prep(unary_union([polygons[t] for t in sorted(tokens)]).buffer(.02))
        self.obstacles={};self.ground={}

    def road_for(self,sid):
        return self.roads[self.locations[sid]]

    @property
    def road(self):
        # 旧单地图实验兼容；混合城市调用必须明确source，不能套Boston地图。
        if len(self.roads)!=1:raise ValueError('多地图数据必须使用 road_for(source_id)')
        return next(iter(self.roads.values()))

    def prepare(self,sid):
        c=self.sources[sid];ctx=self.context[sid];stamps=c['keyframe_timestamps'];frames=ctx['frames'];road=self.road_for(sid)
        bytime=[{a['instance_token']:a for a in f['annotations']} for f in frames]
        cal=frames[0]['sensors'][c['camera']]['calibrated_sensor'];e2cam=np.linalg.inv(transform(cal['translation'],cal['rotation']))
        obstacles=[]
        for f in c['frames']:
            t=f['timestamp'];i=bisect.bisect_right(stamps,t);lo,hi=bytime[i-1:i+1];frac=(t-stamps[i-1])/(stamps[i]-stamps[i-1])
            obs=[]
            # 顺序固定，避免不同进程给同一候选记录不同的首个拒绝原因。
            for tok in sorted(lo.keys()|hi.keys()):
                if tok in lo and tok in hi:
                    a,b=lo[tok],hi[tok];obj={'instance_token':tok,'category':a['category'],
                        'translation':((1-frac)*np.array(a['translation'])+frac*np.array(b['translation'])).tolist(),
                        'size':a['size'],'rotation':Quaternion.slerp(Quaternion(a['rotation']),Quaternion(b['rotation']),amount=frac).elements.tolist()}
                else:
                    # 短缺失附近保守保留邻接框，不把没有插值当作没有障碍物。
                    obj=dict(lo.get(tok,hi.get(tok)));obj['interpolation_uncertain']=True
                obj['_foot']=footprint(obj);obj['_projection']=projection(obj,f,clip_near=True);obs.append(obj)
            ego=np.array(f['camera_to_world'])@e2cam
            corners=np.array([[3.5,1.25,0,1],[3.5,-1.25,0,1],[-2.5,-1.25,0,1],[-2.5,1.25,0,1]])@ego.T
            obs.append({'instance_token':'ego_conservative_proxy','category':'ego_proxy','_foot':Polygon(corners[:,:2]),'_projection':None})
            obstacles.append(obs)
        self.obstacles[sid]=obstacles
        cache=self.root/'geometry'/f'{sid}.json';cloudfile=self.root/'geometry'/f'{sid}_ground.npz'
        if cache.exists():
            meta=read(cache)
            if meta.get('map_location','boston-seaport')!=self.locations[sid]:
                raise ValueError(f'缓存地图与source不一致: {sid}')
            if meta['pass']:
                points=np.load(cloudfile)['points'];meta['_tree']=cKDTree(points[:,:2])
            self.ground[sid]=meta;return meta
        clouds=[]
        anchor=np.array([f['actors'][0]['translation'] for f in c['frames']]);xmin,ymin=anchor[:,:2].min(0)-16;xmax,ymax=anchor[:,:2].max(0)+16
        for k in [0,3,6]:
            datum=frames[k]['sensors']['LIDAR_TOP'];path=self.root/'rgb'/datum['filename']
            raw=np.fromfile(path,dtype=np.float32)
            if raw.size%5:raise ValueError(f'LiDAR binary malformed: {path}')
            points=raw.reshape(-1,5)[:,:3];cal=datum['calibrated_sensor'];ego=datum['ego_pose']
            m=transform(ego['translation'],ego['rotation'])@transform(cal['translation'],cal['rotation'])
            points=points@m[:3,:3].T+m[:3,3];sel=(points[:,0]>xmin)&(points[:,0]<xmax)&(points[:,1]>ymin)&(points[:,1]<ymax)
            points=points[sel]
            for a in frames[k]['annotations']:
                local=(points-np.array(a['translation']))@Quaternion(a['rotation']).rotation_matrix
                w,l,h=a['size'];inside=np.all(abs(local)<np.array([l/2+.2,w/2+.2,h/2+.2]),axis=1);points=points[~inside]
            # 地图只约束平面区域；高度仍由真实LiDAR拟合。
            points=points[[road.covers(Point(p[:2])) for p in points]]
            clouds.append(points)
        points=np.concatenate(clouds);rng=np.random.default_rng(42)
        # 半米格保留低点，避免把树冠/车顶主平面当路面。
        bins=np.floor(points[:,:2]/.5).astype(int);order=np.lexsort((points[:,2],bins[:,1],bins[:,0]));b=bins[order]
        keep=np.r_[True,np.any(b[1:]!=b[:-1],axis=1)] if len(order) else np.empty(0,bool)
        low=points[order[keep]]
        if len(low)>8000:low=low[rng.choice(len(low),8000,replace=False)]
        best=None;best_count=0
        if len(low)>=50:
            A=np.c_[low[:,:2],np.ones(len(low))]
            for _ in range(256):
                ix=rng.choice(len(low),3,replace=False)
                try:coef=np.linalg.solve(A[ix],low[ix,2])
                except np.linalg.LinAlgError:continue
                if np.linalg.norm(coef[:2])>math.tan(math.radians(10)):continue
                ins=np.abs(A@coef-low[:,2])<.08;n=int(ins.sum())
                if n>best_count:best=ins;best_count=n
        passed=best is not None and best_count>=50 and best_count/max(1,len(low))>=.55
        meta={'source_id':sid,'map_location':self.locations[sid],'pass':bool(passed),'lidar_keyframes':[0,3,6],'low_grid_points':len(low),
              'ransac_inlier_count':best_count,'ransac_fraction':best_count/max(1,len(low)),
              'source':'real_LiDAR_after_GT_actor_removal_and_drivable_map_filter','ground_not_from_GT_box_bottom':True}
        if passed:
            plane=np.linalg.lstsq(A[best],low[best,2],rcond=None)[0];ground=points[np.abs(np.c_[points[:,:2],np.ones(len(points))]@plane-points[:,2])<.08]
            meta.update(plane=plane.tolist(),ground_points=len(ground),fit_residual_p90=float(np.percentile(abs(A[best]@plane-low[best,2]),90)))
            cloudfile.parent.mkdir(exist_ok=True);np.savez_compressed(cloudfile,points=ground.astype(np.float32));meta['_tree']=cKDTree(ground[:,:2])
        dump(cache,{k:v for k,v in meta.items() if not k.startswith('_')});self.ground[sid]=meta;return meta

    def trajectory(self,sid,longitudinal,lateral,donor_sid,mode='world_offset',min_ratio=.65):
        c=self.sources[sid];d=self.sources[donor_sid];g=self.ground[sid];road=self.road_for(sid)
        if not g['pass']:return None,'ground_fit'
        assert len(c['frames'])==len(d['frames'])
        mid=len(c['frames'])//2
        ref=c['frames'][mid];b=ref['actors'][0];cam=np.array(ref['camera_to_world'])[:3,3]
        toward=cam[:2]-b['translation'][:2];toward/=np.linalg.norm(toward);right=np.array([-toward[1],toward[0]])
        offset=toward*longitudinal+right*lateral
        if mode=='donor_camera_replay':
            dm=d['frames'][mid];donor_mid=dm['_w2c']@np.r_[dm['actors'][0]['translation'],1]
            base_mid=ref['_w2c']@np.r_[b['translation'],1]
            replay_scale=(base_mid[2]-longitudinal)/donor_mid[2]
            replay_shift=base_mid[0]+lateral-replay_scale*donor_mid[0]
            if replay_scale<=0:return None,'replay_behind_camera'
        elif mode!='world_offset':raise ValueError(mode)
        size=np.median([f['actors'][0]['size'] for f in d['frames']],axis=0).tolist();h=size[2];plane=np.array(g['plane'])
        rows=[];min_gap=1e9;max_support=0.;max_yaw=0.;max_pitch=0.;prev=None
        for f,df,obs in zip(c['frames'],d['frames'],self.obstacles[sid]):
            base=f['actors'][0];r=Quaternion(base['rotation']).rotation_matrix;base_yaw=math.atan2(r[1,0],r[0,0])
            if mode=='donor_camera_replay':
                da=df['actors'][0];dc=df['_w2c']@np.r_[da['translation'],1]
                dc[:3]*=replay_scale;dc[0]+=replay_shift
                xy=(np.array(f['camera_to_world'])@dc)[:2]
                transported=np.array(f['camera_to_world'])[:3,:3]@df['_w2c'][:3,:3]@Quaternion(da['rotation']).rotation_matrix
                yaw=math.atan2(transported[1,0],transported[0,0])
                if abs(wrap(math.degrees(yaw-base_yaw)))>20:return None,'replay_heading_mismatch'
            else:yaw=base_yaw;xy=np.array(base['translation'])[:2]+offset
            R=ground_orientation(yaw,plane);bottom=np.r_[xy,np.dot(np.r_[xy,1],plane)];center=bottom+R[:,2]*(h/2)
            a={'translation':center.tolist(),'size':size,'rotation':Quaternion(matrix=R).elements.tolist()};foot=footprint(a)
            if not road.covers(foot):return None,'outside_drivable'
            support=g['_tree'].query(np.array(foot.exterior.coords)[:4])[0].max();max_support=max(max_support,float(support))
            if support>2.5:return None,'ground_support_gap'
            p=projection(a,f)
            if p is None:return None,'behind_camera'
            bb=p['box'];w,hh=bb[2:]-bb[:2]
            if min(bb[0],bb[1],1024-bb[2],576-bb[3])<8 or w<72 or hh<40 or not .006<=w*hh/(1024*576)<=.18:return None,'projected_size_border'
            db=np.array(df['actors'][0]['projection']['box_xyxy']);ratio=np.array([w,hh])/(db[2:]-db[:2])
            if min(ratio)<min_ratio:return None,'downscale_proxy_gate'
            if max(ratio)>1.5:return None,'upscale_gt1p5'
            if max(ratio)/min(ratio)>1.12:return None,'aspect_deformation_gt1p12'
            angle=view_angles(a,f);donor_angle=view_angles(df['actors'][0],df);diff=np.abs(wrap(angle-donor_angle));max_yaw=max(max_yaw,float(diff[0]));max_pitch=max(max_pitch,float(diff[1]))
            if diff[0]>12 or diff[1]>8:return None,'view_angle'
            for ob in obs:
                gap=foot.distance(ob['_foot']);min_gap=min(min_gap,gap)
                if gap<.3:return None,'collision_or_clearance'
            if prev is not None:
                dt=(f['timestamp']-prev['timestamp'])/1e6;factor=.1/dt
                if np.linalg.norm(center-np.array(prev['actor']['translation']))*factor>2:return None,'position_continuity'
                if abs(wrap(math.degrees(yaw-prev['yaw'])))*factor>5:return None,'yaw_continuity'
                scale=np.array([w,hh])/prev['wh'];scale=scale**factor
                if scale.min()<.85 or scale.max()>1.18:return None,'scale_continuity'
            row={'frame':f['frame'],'timestamp':f['timestamp'],'actor':a,'box':bb.tolist(),'scale_xy':ratio.tolist(),
                 'view_delta_deg':diff.tolist(),'depth_interval':[p['near_depth'],p['far_depth']],'center_depth':p['center_depth']}
            rows.append(row);prev={'timestamp':f['timestamp'],'actor':a,'yaw':yaw,'wh':np.array([w,hh])}
        return {'source_id':sid,'donor_source_id':donor_sid,'offset_longitudinal_m':longitudinal,'offset_lateral_m':lateral,'placement_mode':mode,
                'offset_semantics':'midframe camera depth/lateral anchoring' if mode=='donor_camera_replay' else 'fixed world offset toward midframe camera',
                'frames':rows,'min_GT_clearance_m':float(min_gap),'max_ground_support_distance_m':max_support,
                'max_view_yaw_delta_deg':max_yaw,'max_view_pitch_delta_deg':max_pitch,'ground':{k:v for k,v in g.items() if not k.startswith('_')}},None

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();geo=Geometry(a.root)
    qa=read(a.root/'subagent_source_reviews.json')['clips'];rows=[]
    for r in qa:
        if r['receiver_status']!='pass':continue
        result=geo.prepare(r['source_id']);rows.append({k:v for k,v in result.items() if not k.startswith('_')});print('GROUND',r['source_id'],result['pass'],result['ransac_fraction'],flush=True)
    dump(a.root/'geometry/ground_summary.json',rows)

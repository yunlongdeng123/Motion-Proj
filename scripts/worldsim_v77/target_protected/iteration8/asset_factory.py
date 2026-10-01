"""真实相机下显式车辆遮挡：3D轮廓、实际地面、连续位姿；RGB仅诊断。"""
from pathlib import Path
import sys,os,math,copy,time
from collections import Counter,defaultdict
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
import numpy as np,cv2
from pyquaternion import Quaternion
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union
from shapely.prepared import prep
from geometry_factory import Geometry,read,dump,footprint,ground_orientation,projection,wrap,source_masks
from iteration5.planning import normalized_masks
from exact_pairs import O,F,T
from build_pairs import runlen

class ParkingGeometry(Geometry):
    def __init__(self,root):
        super().__init__(root)
        raw=read(self.root/'maps/expansion/boston-seaport.json');nodes={n['token']:(n['x'],n['y']) for n in raw['node']}
        polys={p['token']:Polygon([nodes[t] for t in p['exterior_node_tokens']],[[nodes[t] for t in h['node_tokens']] for h in p['holes']]) for p in raw['polygon']}
        tokens={t for r in raw['drivable_area'] for t in r['polygon_tokens']}|{r['polygon_token'] for r in raw.get('carpark_area',[])}
        self.road=prep(unary_union([polys[t] for t in tokens]).buffer(.02))
        self.cache=O/'asset_geometry';self.cache.mkdir(exist_ok=True)
        rgb=self.cache/'rgb'
        if not rgb.exists():rgb.symlink_to((root/'rgb').resolve(),target_is_directory=True)
    def prepare(self,sid):
        old=self.root;self.root=self.cache
        try:return super().prepare(sid)
        finally:self.root=old

def silhouette(vertices,faces,actor,frame):
    w,l,h=actor['size'];R=Quaternion(actor['rotation']).rotation_matrix
    world=(vertices*np.array([l,w,h]))@R.T+np.array(actor['translation'])
    cam=world@frame['_w2c'][:3,:3].T+frame['_w2c'][:3,3]
    assert cam[:,2].min()>.5
    uv=cam@np.array(frame['intrinsics_1024']).T;uv=uv[:,:2]/uv[:,2:]
    rounded=np.rint(uv).astype('int32');polys=rounded[faces];m=np.zeros((576,1024),'uint8')
    inside=(rounded[:,0]>=0)&(rounded[:,0]<1024)&(rounded[:,1]>=0)&(rounded[:,1]<576)
    points=rounded[inside];m[points[:,1],points[:,0]]=1
    delta=polys[:,1:]-polys[:,:1];area=delta[:,0,0]*delta[:,1,1]-delta[:,0,1]*delta[:,1,0]
    extent=np.ptp(polys,axis=1).max(1)
    polys=polys[(area!=0)|(extent>1)]
    # 每个三角形独立填充，不能用一次fillPoly的奇偶规则互相抵消重叠三角形。
    for p in polys:cv2.fillConvexPoly(m,p,1)
    return m>0

def trajectory(geo,sid,z,x,size):
    c=geo.sources[sid];g=geo.ground[sid];plane=np.array(g['plane']);mid=c['frames'][5];b=mid['actors'][0];cam=np.array(mid['camera_to_world'])[:3,3]
    toward=cam[:2]-np.array(b['translation'])[:2];toward/=np.linalg.norm(toward);right=np.array([-toward[1],toward[0]])
    offset=toward*z+right*x;rows=[];min_gap=1e9;max_support=0;prev=None
    for f,obs in zip(c['frames'],geo.obstacles[sid]):
        base=f['actors'][0];r=Quaternion(base['rotation']).rotation_matrix;yaw=math.atan2(r[1,0],r[0,0]);xy=np.array(base['translation'])[:2]+offset
        R=ground_orientation(yaw,plane);bottom=np.r_[xy,np.dot(np.r_[xy,1],plane)];center=bottom+R[:,2]*(size[2]/2)
        a={'translation':center.tolist(),'size':size,'rotation':Quaternion(matrix=R).elements.tolist()};foot=footprint(a)
        if not geo.road.covers(foot):return None,'outside_mapped_drivable_or_parking'
        support=g['_tree'].query(np.array(foot.exterior.coords)[:4])[0].max();max_support=max(max_support,float(support))
        if support>2.5:return None,'ground_support_gap'
        p=projection(a,f)
        if p is None:return None,'behind_camera'
        bb=p['box'];wh=bb[2:]-bb[:2]
        if min(bb[0],bb[1],1024-bb[2],576-bb[3])<8 or wh[0]<72 or wh[1]<40 or not .006<=np.prod(wh)/(1024*576)<=.18:return None,'size_border'
        if bb[3]+6>=512:return None,'ego_band'
        for ob in obs:
            gap=foot.distance(ob['_foot']);min_gap=min(min_gap,gap)
            if gap<.3:return None,'collision_clearance'
        if prev is not None:
            factor=100000/(f['timestamp']-prev['timestamp'])
            if np.linalg.norm(center-prev['center'])*factor>2 or abs(wrap(math.degrees(yaw-prev['yaw'])))*factor>5:return None,'trajectory_continuity'
            scale=(wh/prev['wh'])**factor
            if scale.min()<.85 or scale.max()>1.18:return None,'scale_continuity'
        rows.append({'frame':f['frame'],'timestamp':f['timestamp'],'actor':a,'box':bb.tolist(),'depth_interval':[p['near_depth'],p['far_depth']]})
        prev={'timestamp':f['timestamp'],'center':center,'yaw':yaw,'wh':wh}
    return {'source_id':sid,'frames':rows,'min_GT_clearance_m':min_gap,'max_ground_support_distance_m':max_support,'offset_longitudinal_m':z,'offset_lateral_m':x,'ground':{k:v for k,v in g.items() if not k.startswith('_')}},None

def exact(geo,tr,alphas,protected):
    sid=tr['source_id'];ratios={tok:[] for tok in protected};hole_ratios={tok:[] for tok in protected};static=set()
    for i,(aa,p,obs) in enumerate(zip(alphas,tr['frames'],geo.obstacles[sid])):
        H=cv2.dilate(aa.astype('uint8'),np.ones((13,13),'uint8'))>0;yy,xx=np.where(aa)
        if len(yy)<200 or xx.max()-xx.min()+1<72 or yy.max()-yy.min()+1<40:return None,'actual_size'
        if H[512:].any():return None,'ego_band'
        for tok,mm in protected.items():
            m=mm[i];ratio=float((aa&m).sum()/max(1,m.sum()));ratios[tok].append(ratio);hole_ratios[tok].append(float((H&m).sum()/max(1,m.sum())))
        for ob in obs:
            pr=ob['_projection'];tok=ob['instance_token']
            if pr is None:continue
            bb=pr['box'];x0,y0=np.maximum(np.floor(bb[:2]).astype(int),0);x1,y1=np.minimum(np.ceil(bb[2:]).astype(int),[1024,576])
            hit=x1>x0 and y1>y0 and H[y0:y1,x0:x1].sum()>max(12,.01*H.sum())
            if not hit:continue
            if tok in protected:
                if ratios[tok][-1]>.01 and p['depth_interval'][1]>.3+pr['near_depth']:return None,'protected_depth_order'
            elif ob['category'] in {'movable_object.barrier','movable_object.trafficcone','static_object.bicycle_rack'}:
                if ob.get('interpolation_uncertain') or p['depth_interval'][1]>.3+pr['near_depth']:return None,'static_foreground'
                static.add(tok)
            else:return None,'unreviewed_dynamic_envelope'
    active=[tok for tok,v in ratios.items() if max(v)>.01]
    if not active:kind='background'
    elif len(active)==1:
        v=np.array(ratios[active[0]]);h=np.array(hole_ratios[active[0]])
        if not np.all((v>=.3)&(v<=.8)) or max(v)>.85 or max(h)>.85:return None,'single_occlusion'
        kind='single_actor'
    else:
        v=np.array([ratios[t] for t in active]);h=np.array([hole_ratios[t] for t in active])
        if not np.all((v>=.2)&(v<=.7)) or v.max()>.8 or h.max()>.85:return None,'dense_occlusion'
        kind='dense_actors'
    stats=normalized_masks(alphas,[p['box'] for p in tr['frames']])
    if any(s['normalized_iou']<.8 or not .85<=s['normalized_area_ratio']<=1.15 for s in stats):return None,'mask_continuity'
    return {'type':kind,'protected_instances':active,'occlusion_fraction':ratios,'hole_overlap_upper':hole_ratios,'silhouette_stats':stats,'static_background_annotations_behind_A':sorted(static)},None

def main(shard_index=0,shard_count=1,policy='coarse'):
    cv2.setNumThreads(1);geo=ParkingGeometry(F);out=O/('asset_candidates' if policy=='coarse' else 'asset_focus_candidates');out.mkdir(exist_ok=True)
    assets={name:dict(np.load(O/'assets'/f'{name}.npz')) for name in ['sedan','suv']}
    offsets=[(z,x) for z in [-6.,-4.,-2.,0.,2.,4.,6.,8.,10.] for x in [-8.,-6.,-4.,-2.,0.,2.,4.,6.,8.]]
    if policy=='focused':offsets=[(z,x) for z in [4.5,5.5,7.,9.,12.,15.] for x in [-3.,-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.,3.]]
    rows=[];pools=[]
    sources=[(sid,c) for sid,c in geo.sources.items() if sid.startswith('N')][shard_index::shard_count]
    for sid,c in sources:
        if not sid.startswith('N'):continue
        saved=out/(sid+'.json')
        if saved.exists():row=read(saved);rows.append(row['summary']);pools+=row['candidates'];continue
        g=geo.prepare(sid);protected={};rejections=Counter();chosen=defaultdict(list);start=time.time()
        if g['pass']:
            for j,a in enumerate(c['actors']):
                jid=sid if j==0 else sid+'_'+a['instance_token'][:8];folder=F/('segmented' if j==0 else 'segmented_secondary')/jid/'sam2_raw'
                if not folder.exists():continue
                mm=source_masks(F,sid,None if j==0 else jid);stats=normalized_masks(mm,[next(aa for aa in f['actors'] if aa['instance_token']==a['instance_token'])['projection']['box_xyxy'] for f in c['frames']])
                if all(s['pixels']>0 and s['normalized_iou']>=.8 and .85<=s['normalized_area_ratio']<=1.15 for s in stats):protected[a['instance_token']]=mm
            if not protected:rejections['no_stable_observed_protected_masks']+=1
            else:
                for asset,size in [('sedan',[1.85,4.5,1.5]),('suv',[1.95,4.7,1.75])]:
                    data=assets[asset]
                    for z,x in offsets:
                        tr,why=trajectory(geo,sid,z,x,size)
                        if tr is None:rejections[why]+=1;continue
                        # 单帧先拒绝不合法遮挡，避免在明确失败时完整渲染十帧。
                        aa=silhouette(data['vertices'],data['faces'],tr['frames'][0]['actor'],c['frames'][0])
                        # 仍用完整十帧最终关卡；首帧只做包络/当前已满类别的便宜筛选。
                        roughactive=[tok for tok,m in protected.items() if (aa&m[0]).sum()/max(1,m[0].sum())>.01]
                        likely='background' if not roughactive else 'single_actor' if len(roughactive)==1 else 'dense_actors'
                        if len(chosen[likely])>=3:continue
                        alphas=[aa]+[silhouette(data['vertices'],data['faces'],p['actor'],f) for p,f in zip(tr['frames'][1:],c['frames'][1:])]
                        q,why=exact(geo,tr,alphas,protected)
                        if q is None:rejections[why]+=1;continue
                        if len(chosen[q['type']])>=3:continue
                        if any(p['asset']==asset and abs(p['offset_longitudinal_m']-z)<2.1 and abs(p['offset_lateral_m']-x)<2.1 for p in chosen[q['type']]):continue
                        chosen[q['type']].append(tr|q|{'asset':asset,'scene':c['scene'],'silhouette_method':'full explicit mesh triangle projection at actual camera/metric SE3, no donor affine view restriction'})
        pp=[p for ar in chosen.values() for p in ar];pools+=pp
        row={'source_id':sid,'scene':c['scene'],'ground_pass':g['pass'],'counts':{k:len(v) for k,v in chosen.items()},'rejects':dict(rejections),'seconds':time.time()-start};rows.append(row)
        dump(saved,{'summary':row,'candidates':pp});dump(O/f'asset_{policy}_state_{shard_index}.json',{'completed':len(rows),'total':len(sources),'candidate_count':len(pools),'candidate_scene_count':len({p['scene'] for p in pools}),'type_counts':dict(Counter(p['type'] for p in pools)),'pid':os.getpid()});print('ASSET',sid,row['counts'],'seconds',round(row['seconds'],1),flush=True)
    dump(O/f'all_asset_{policy}_candidates_{shard_index}.json',{'stage':'complete','policy':policy,'candidates':pools,'rows':rows,'type_counts':dict(Counter(p['type'] for p in pools)),'scene_count':len({p['scene'] for p in pools}),'physical_offsets':offsets})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);p.add_argument('--policy',choices=['coarse','focused'],default='coarse');a=p.parse_args();main(a.shard_index,a.shard_count,a.policy)

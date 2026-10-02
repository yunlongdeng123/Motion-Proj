"""真实合法输入的状态投影；Y只由独立评价脚本读取。"""
from pathlib import Path
import json,sys,os,time
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path[:0]=[str(P),str(P/'iteration11'),str(P/'iteration12')]
import numpy as np,cv2
from PIL import Image,ImageDraw
from actor_state import Observation,build_actor_samples,build_background_samples,render_state,cuboid_front_depth
from long_factory import geometry,ROOT
from geometry_factory import transform
from shapely.geometry import Point
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r22'
def read(p):return json.loads(p.read_text())
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main(ground_patches=False):
    cv2.setNumThreads(2);g=geometry();summary=[]
    assert read(O/'segmentation_state.json')['stage']=='complete_pending_identity_review'
    jobs=read(O/'observed_queue.json')['jobs']
    for cid in read(O/'run.json')['probe_cases']:
        start=time.monotonic();inp=O/'observed'/cid;meta=read(inp/'observations.json');sid=meta['source_id'];frames=meta['frames']
        dest=O/('state_patches' if ground_patches else 'state')/cid;dest.mkdir(parents=True,exist_ok=True)
        ground=g.prepare(sid);assert ground['pass']
        observations=[Observation(np.asarray(Image.open(inp/'rgb'/f'{i:05}.png')).copy(),np.asarray(Image.open(inp/'H'/f'{i:05}.png'))>0,np.array(f['camera_to_world']),np.array(f['intrinsics_1024']),f['timestamp']) for i,f in enumerate(frames)]
        tracks={};masks={}
        for job in [j for j in jobs if j['case_id']==cid]:
            tok=job['instance_token'];tracks[tok]=[next(a for a in f['actors'] if a['instance_token']==tok) for f in frames]
            masks[tok]=[np.asarray(Image.open(O/'observed_masks'/job['job_id']/f'{i:05}.png'))>0 for i in range(30)]
        samples=build_actor_samples(observations,tracks,masks)
        front=[];excluded=[]
        for ob,obstacles in zip(observations,g.obstacles[sid]):
            # 合成A从未存在于真实GT列表，故这里天然是删除A后的保留世界。
            d=cuboid_front_depth(ob,[a for a in obstacles if a['instance_token']!='ego_conservative_proxy'])
            front.append(d);excluded.append(np.isfinite(d))
        lidar=[];clouds={};sources=[];ta,tb=meta['window_start'],meta['window_end']
        in_window=[r['sensors']['LIDAR_TOP'] for r in g.context[sid]['frames'] if ta<=r['sensors']['LIDAR_TOP']['timestamp']<=tb]
        available=[d for d in in_window if (ROOT/'rgb'/d['filename']).is_file()]
        # 不沿用含窗口外扫描的旧地面平面。重新只用当前窗口扫描拟合。
        for datum in available:
            pts=np.fromfile(ROOT/'rgb'/datum['filename'],np.float32).reshape(-1,5)[:,:3]
            cal=datum['calibrated_sensor'];ego=datum['ego_pose'];m=transform(ego['translation'],ego['rotation'])@transform(cal['translation'],cal['rotation']);pts=pts@m[:3,:3].T+m[:3,3]
            cam=np.array(frames[len(frames)//2]['camera_to_world'])[:3,3]
            pts=pts[np.linalg.norm(pts[:,:2]-cam[:2],axis=1)<40]
            clouds[datum['token']]=pts
        full=np.concatenate(list(clouds.values())) if clouds else np.empty((0,3))
        road=full[[g.road.covers(Point(p[:2])) for p in full]] if len(full) else full
        plane=None;plane_info={'input_scans':len(available),'window_only':True,'pass':False}
        if len(road)>=50:
            bins=np.floor(road[:,:2]/.5).astype(int);order=np.lexsort((road[:,2],bins[:,1],bins[:,0]));b=bins[order];keep=np.r_[True,np.any(b[1:]!=b[:-1],axis=1)];low=road[order[keep]]
            A=np.c_[low[:,:2],np.ones(len(low))];rng=np.random.default_rng(42);best=np.zeros(len(low),bool)
            for _ in range(128):
                ix=rng.choice(len(low),3,replace=False)
                try:coef=np.linalg.solve(A[ix],low[ix,2])
                except np.linalg.LinAlgError:continue
                if np.linalg.norm(coef[:2])>np.tan(np.deg2rad(10)):continue
                fit=abs(A@coef-low[:,2])<.08
                if fit.sum()>best.sum():best=fit
            if best.sum()>=50 and best.mean()>=.55:
                plane=np.linalg.lstsq(A[best],low[best,2],rcond=None)[0];plane_info.update(pass_=True,inlier_count=int(best.sum()),fraction=float(best.mean()),plane=plane.tolist());plane_info['pass']=True
        for i,ob in enumerate(observations):
            if not available or plane is None:lidar.append(np.empty((0,3)));sources.append(None);continue
            datum=min(available,key=lambda d:abs(d['timestamp']-ob.timestamp))
            if abs(datum['timestamp']-ob.timestamp)>100000:
                lidar.append(np.empty((0,3)));sources.append(None);continue
            key=datum['token']
            pts=clouds[key];height=pts[:,2]-np.c_[pts[:,:2],np.ones(len(pts))]@plane;pts=pts[abs(height)<=.08]
            lidar.append(pts);sources.append({'camera_frame':i,'lidar_timestamp':datum['timestamp'],'delta_ms':abs(datum['timestamp']-ob.timestamp)/1000,'within_RGB_window':True})
        background=build_background_samples(observations,lidar,excluded)
        patch_stats=None
        if ground_patches and plane is not None:
            from ground_patches import observed_ground_patches
            background,patch_stats=observed_ground_patches(observations,lidar,excluded,plane)
        results=render_state(observations,tracks,samples,background,retained_box_masks=excluded,retained_front_depth=front)
        withheld=render_state(observations,tracks,samples,background,exclude_source=lambda t:t,retained_box_masks=excluded,retained_front_depth=front)
        metrics=[];contacts=[]
        for i,(ob,r,lo) in enumerate(zip(observations,results,withheld)):
            np.savez_compressed(dest/f'{i:05}.npz',**r)
            np.savez_compressed(dest/f'{i:05}_leave_self.npz',**lo)
            h=ob.hole
            metrics.append({'frame':i,'hole_pixels':int(h.sum()),'O_in_H':int((r['O']&h).sum()),'N_in_H':int((r['N']&h).sum()),'U_in_H':int((r['U']&h).sum()),'known_fraction_in_H':float(((r['O']|r['N'])&h).sum()/max(1,h.sum())),'leave_self_O':int(lo['O'].sum()),'leave_self_N':int(lo['N'].sum()),'mean_Q_in_H':float(r['Q'][h].mean()),'source_actor_visible_pixels':{t:int(ms[i].sum()) for t,ms in masks.items()}})
            rgb=ob.rgb.copy();overlay=rgb.copy()
            for t,ms in masks.items():overlay[ms[i]]=np.rint(overlay[ms[i]]*.4+np.array([45,235,120])*.6).astype('uint8')
            states=np.full_like(rgb,40);states[r['O']]=[40,225,100];states[r['N']]=[40,120,245];states[r['U']]=[55,55,55]
            cond=r['F'].copy();cond[r['U']]=48
            for name,img in [('input',rgb),('visible_sam',overlay),('state',states),('projected_rgb',cond)]:Image.fromarray(img).save(dest/f'{i:05}_{name}.jpg',quality=95)
            if i in [0,15,29]:contacts.append((i,[rgb,overlay,states,cond]))
        canvas=Image.new('RGB',(1600,820),(18,23,29));draw=ImageDraw.Draw(canvas)
        for j,(i,panels) in enumerate(contacts):
            for k,img in enumerate(panels):
                draw.text((400*k+6,j*270+5),f'{cid} f{i} '+['masked input','SAM visible only','O green / N blue / U gray','observed RGB projection'][k],fill='white')
                canvas.paste(Image.fromarray(img).resize((400,225)),(k*400,j*270+30))
        canvas.save(dest/'contact.jpg',quality=95)
        for token,s in samples.items():np.savez_compressed(dest/(token+'_samples.npz'),**{k:v for k,v in s.items() if isinstance(v,np.ndarray)})
        np.savez_compressed(dest/'background_samples.npz',**{k:v for k,v in background.items() if isinstance(v,np.ndarray)})
        row={'case_id':cid,'scene':meta['scene'],'source_id':sid,'frames':30,'window_s':(tb-ta)/1e6,'actors':list(tracks),'actor_samples':{t:len(s['local_xyz']) for t,s in samples.items()},'background_samples':len(background['world_xyz']),'lidar_provenance':sources,'missing_in_window_lidar_files':len(in_window)-len(available),'background_plane':plane_info,'geometry_proxy':'cuboid actor surfaces + observed near-ground LiDAR only','state_input':'masked RGB and masked-input SAM; no Y file read','metrics':metrics,'seconds':time.monotonic()-start,'method_quality':'pending_observed_identity_and_projection_review','human_verdict':None}
        row['background_patch_stats']=patch_stats
        dump(dest/'result.json',row);summary.append(row);print('STATE',cid,row['actor_samples'],row['background_samples'],flush=True)
    dump(O/('state_patches_summary.json' if ground_patches else 'state_summary.json'),{'cases':summary,'stage':'complete_pending_QA','condition_is_sparse':True,'no_training':True,'human_verdict':None})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--ground-patches',action='store_true');main(p.parse_args().ground_patches)

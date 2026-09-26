"""固定地面查询的GT视线与LiDAR支持盘点；未知区域不伪装成已观测背景。"""
import argparse,json,pathlib,sys,time
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from geometry import transform,box_mask,resized_intrinsics
from actor_command_audit import box

def segment_box_occlusion(camera,points,pose,size,endpoint_margin=1e-5):
    """相机到查询点的开线段是否穿过有向框；端点接触不算前景遮挡。"""
    inv=np.linalg.inv(pose);origin=transform(np.asarray(camera)[None],inv)[0]
    end=transform(points,inv);direction=end-origin
    half=np.asarray(size)/2;parallel=np.abs(direction)<1e-12
    outside=parallel & ((origin < -half)|(origin > half))
    safe=np.where(parallel,1.,direction)
    a=(-half-origin)/safe;b=(half-origin)/safe
    low=np.where(parallel,-np.inf,np.minimum(a,b));high=np.where(parallel,np.inf,np.maximum(a,b))
    enter=np.maximum(low.max(1),1e-6);leave=np.minimum(high.min(1),1.-endpoint_margin)
    return (~outside.any(1)) & (leave>enter)

def ground_fit(points_local,nominal_bottom,seed=7704):
    """周围GT框外LiDAR拟合局部路面；仅诊断近似平面，不宣称真值。"""
    p=np.asarray(points_local);p=p[np.abs(p[:,2]-nominal_bottom)<.4]
    if len(p)<50:raise ValueError('insufficient near-ground LiDAR for conditional query plane')
    rng=np.random.default_rng(seed);sub=p[rng.choice(len(p),min(16000,len(p)),replace=False)]
    best=None
    for _ in range(128):
        xyz=sub[rng.choice(len(sub),3,replace=False)];X=np.c_[xyz[:,:2],np.ones(3)]
        if abs(np.linalg.det(X))<.001:continue
        coef=np.linalg.solve(X,xyz[:,2])
        if np.linalg.norm(coef[:2])>.2:continue
        err=np.abs(np.c_[sub[:,:2],np.ones(len(sub))]@coef-sub[:,2]);keep=err<.08
        if best is None or keep.sum()>best[0]:best=(int(keep.sum()),coef,keep)
    if best is None:raise ValueError('no supported near-horizontal plane')
    q=sub[best[2]];coef=np.linalg.lstsq(np.c_[q[:,:2],np.ones(len(q))],q[:,2],rcond=None)[0]
    error=np.abs(np.c_[q[:,:2],np.ones(len(q))]@coef-q[:,2])
    return coef,{'candidate_points':len(p),'sampled_points':len(sub),'inlier_points':len(q),'median_inlier_error_m':float(np.median(error)),'plane_z_ax_by_c':coef.tolist(),'nominal_gt_bottom_local_z':float(nominal_bottom)}

def close_box_to_ground(pose,size,reference_pose,plane,padding=.10):
    """只下延框到局部地面以下，保留上表面，避免悬浮框漏出车底假背景。

    这是车辆/对象到地面的保守遮挡包络，并非真实形状或可见性真值。
    """
    p=np.array(pose,copy=True);s=np.array(size,dtype=float,copy=True)
    n=reference_pose[:3,:3]@np.array([-plane[0],-plane[1],1.])
    rhs=float(n@reference_pose[:3,3]+plane[2]-n@p[:3,3]);nl=p[:3,:3].T@n
    if abs(nl[2])<.5:raise ValueError('ground plane nearly vertical in object coordinates')
    corners=np.array([[x,y] for x in [-s[0]/2,s[0]/2] for y in [-s[1]/2,s[1]/2]])
    floor=float(((rhs-corners@nl[:2])/nl[2]).min()-padding)
    bottom=min(-s[2]/2,floor);top=s[2]/2
    p[:3,3]+=p[:3,:3]@np.array([0.,0.,(top+bottom)/2]);s[2]=top-bottom
    return p,s

def core_reference_gate(core,clear,support):
    """双证据也只是候选；缺证据则禁止复制到所谓已观测背景。"""
    candidates=np.asarray(core)&(np.asarray(clear)>0)&np.asarray(support)
    return {'joint_candidate_queries':int(candidates.sum()),'copy_as_observed_background_admitted':False,
            'status':'candidate_requires_static_occlusion_check' if candidates.any() else 'no_jointly_supported_core_reference',
            'reason':'GT贴地框视线与近邻LiDAR不足以证明RGB射线未碰围栏等静态遮挡；无联合候选时停止参考复制，不生成伪观测。'}

def main():
    from PIL import Image
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=pathlib.Path,default=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-PIPELINE-R3-20260926/r1'))
    parser.add_argument('--data-root',type=pathlib.Path,default=pathlib.Path('/root/autodl-tmp/data/v76_vadgs'))
    args=parser.parse_args();OUT=args.out;OUT.mkdir(parents=True,exist_ok=True)
    registration={'task_id':'WS-V77-PIPELINE-R3-20260926','run_id':'r1','source_commit':'a0a18b3a','training_steps':0,'new_model_forwards':0,'seed':7704,'query_spacing_m':.2,'ground_closure_padding_m':.10,'scene_scope':'same two development scenes and actors','input_roles':'诊断使用GT相机/全部对象框与原LiDAR；不声称RGB自动或真实背景GT','failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02','human_verdict':None}
    (OUT/'registration.json').write_text(json.dumps(registration,ensure_ascii=False,indent=2)+'\n')
    summaries=[]
    for name,aid,ref in [('scene_0230','22',5),('scene_0255','25',20)]:
        started=time.monotonic();data=args.data_root/name
        inst=json.loads((data/'instances/instances_info.json').read_text());target=inst[aid]['frame_annotations'];frames=target['frame_idx'];pose,size=box(inst[aid],ref)
        # Do not infer disappearance when a GT track ends: only frames with
        # known target pose enter visibility or LiDAR evidence denominators.
        points=[];scans=[]
        for f in frames:
            xyz=np.fromfile(data/'lidar'/f'{f:03d}.bin',np.float32).reshape(-1,4)[:,:3]
            xyz=transform(xyz,np.loadtxt(data/'lidar_pose'/f'{f:03d}.txt'))
            xyz=xyz[np.linalg.norm(xyz[:,:2]-pose[:2,3],axis=1)<10]
            keep=np.ones(len(xyz),bool)
            for row in inst.values():
                item=box(row,f)
                if item is not None:
                    p,s=item
                    if np.linalg.norm(p[:2,3]-pose[:2,3])<17:keep &= ~box_mask(xyz,p,s+.2)
            points.append(xyz[keep]);scans.append(np.full(keep.sum(),f))
        lidar=np.concatenate(points);local=transform(lidar,np.linalg.inv(pose));scanids=np.concatenate(scans)
        coef,plane=ground_fit(local,-size[2]/2)
        step=.2;xs=np.arange(-size[0]/2-1.6,size[0]/2+1.6+step/2,step);ys=np.arange(-size[1]/2-1.6,size[1]/2+1.6+step/2,step)
        xx,yy=np.meshgrid(xs,ys);qx,qy=xx.ravel(),yy.ravel();qz=coef[0]*qx+coef[1]*qy+coef[2]
        qlocal=np.stack([qx,qy,qz],axis=1);query=transform(qlocal,pose)
        core=(np.abs(qx)<size[0]/2-.2)&(np.abs(qy)<size[1]/2-.2)
        ground_error=np.abs(local[:,2]-(coef[0]*local[:,0]+coef[1]*local[:,1]+coef[2]));gp=lidar[ground_error<.10];gf=scanids[ground_error<.10]
        distance,index=cKDTree(gp).query(query,k=1);support=distance<=.15
        count=len(query);fov=np.zeros(count,int);target_clear=np.zeros(count,int);all_clear=np.zeros(count,int);same_stream_clear=np.zeros(count,int);raw_all_clear=np.zeros(count,int);score=np.full(count,-np.inf);best=np.full((count,2),-1,int);frame_rows=[];bottom_heights=[]
        keys=[]
        for c in range(6):
            v=np.loadtxt(data/'intrinsics'/f'{c}.txt');keys.append(resized_intrinsics([[v[0],0,v[2]],[0,v[1],v[3]],[0,0,1]],(1600,900),(384,688)))
        for f in frames:
            raw_tp,raw_ts=box(inst[aid],f);raw_others=[b for i,row in inst.items() if i!=aid and (b:=box(row,f)) is not None]
            bottom_heights.append(float(transform(np.array([[0.,0.,-raw_ts[2]/2]]),raw_tp)[0,2]))
            tp,ts=close_box_to_ground(raw_tp,raw_ts,pose,coef)
            others=[close_box_to_ground(op,os,pose,coef) for op,os in raw_others]
            for c in range(6):
                c2w=np.loadtxt(data/'extrinsics'/f'{f:03d}_{c}.txt');cp=transform(query,np.linalg.inv(c2w));uv=cp@keys[c].T;uv=uv[:,:2]/np.maximum(uv[:,2:],1e-9)
                seen=(cp[:,2]>.5)&(cp[:,2]<60)&(uv[:,0]>=0)&(uv[:,0]<687)&(uv[:,1]>=0)&(uv[:,1]<383)
                if not seen.any():continue
                blocked_target=segment_box_occlusion(c2w[:3,3],query,tp,ts)
                clear=seen & ~blocked_target;fov+=seen;target_clear+=clear
                raw_clear=seen&~segment_box_occlusion(c2w[:3,3],query,raw_tp,raw_ts)
                for op,os in raw_others:
                    if not raw_clear.any():break
                    ids=np.flatnonzero(raw_clear);raw_clear[ids]&=~segment_box_occlusion(c2w[:3,3],query[ids],op,os)
                raw_all_clear+=raw_clear
                for op,os in others:
                    if not clear.any():break
                    ids=np.flatnonzero(clear);clear[ids]&=~segment_box_occlusion(c2w[:3,3],query[ids],op,os)
                all_clear+=clear
                if c==(2 if name=='scene_0230' else 3):same_stream_clear+=clear
                candidate=np.abs(c2w[2,3]-query[:,2])/np.maximum(cp[:,2],.1)**3
                choose=clear&(candidate>score);score[choose]=candidate[choose];best[choose]=[f,c]
                frame_rows.append({'frame':f,'camera':c,'core_in_fov':int((seen&core).sum()),'core_target_box_clear':int((seen&~blocked_target&core).sum()),'core_all_boxes_clear':int((clear&core).sum()),'core_raw_boxes_clear':int((raw_clear&core).sum())})
        # Sample only geometrically selected RGB; no fill of unknown cells.
        colors=np.zeros((count,3),np.uint8)
        for f,c in np.unique(best[best[:,0]>=0],axis=0):
            chosen=(best[:,0]==f)&(best[:,1]==c);ids=np.flatnonzero(chosen)
            img=np.array(Image.open(data/'images'/f'{f:03d}_{c}.jpg').convert('RGB').resize((688,384),Image.Resampling.BICUBIC))
            c2w=np.loadtxt(data/'extrinsics'/f'{f:03d}_{c}.txt');cp=transform(query[ids],np.linalg.inv(c2w));uv=cp@keys[c].T;uv=np.rint(uv[:,:2]/uv[:,2:]).astype(int)
            colors[ids]=img[uv[:,1].clip(0,383),uv[:,0].clip(0,687)]
        def coverage(region):
            return {'queries':int(region.sum()),'ever_in_fov':int((region&(fov>0)).sum()),'raw_box_ever_all_boxes_clear':int((region&(raw_all_clear>0)).sum()),'ever_target_box_clear':int((region&(target_clear>0)).sum()),'ever_all_boxes_clear':int((region&(all_clear>0)).sum()),'same_poc_camera_ever_all_boxes_clear':int((region&(same_stream_clear>0)).sum()),'lidar_ground_support_within_15cm':int((region&support).sum()),'both_box_clear_and_lidar_support':int((region&(all_clear>0)&support).sum())}
        row={'scene':name,'actor_id':aid,'reference_frame':ref,'target_track_frames':len(frames),'frame_range':[min(frames),max(frames)],'omitted_missing_target_pose_frames':196-len(frames),'plane':plane,'target_raw_box_bottom_world_z_range_m':[min(bottom_heights),max(bottom_heights)],'core_footprint_inset_m':.2,'core':coverage(core),'surround':coverage(~core),'unknown_policy':'贴地包络视线是保守且条件性的几何筛查，不包含静态薄物体或真实surface visibility；无候选或无LiDAR证据均不得补写成已观测。GT结束不当作目标消失。','human_verdict':None,'elapsed_s':time.monotonic()-started}
        row['core_reference_gate']=core_reference_gate(core,all_clear,support)
        np.savez_compressed(OUT/f'{name}_visibility.npz',query_world=query,query_local=qlocal,grid_shape=xx.shape,core=core,in_fov_views=fov,target_box_clear_views=target_clear,raw_all_box_clear_views=raw_all_clear,all_box_clear_views=all_clear,same_stream_clear_views=same_stream_clear,best_frame_camera=best,rgb=colors,lidar_ground_support=support,lidar_nearest_distance_m=distance,lidar_nearest_frame=gf[index],gt_pose=pose,gt_size=size)
        (OUT/f'{name}_visibility.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');(OUT/f'{name}_view_counts.json').write_text(json.dumps(frame_rows,indent=2)+'\n');summaries.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    (OUT/'summary.json').write_text(json.dumps({'registration':registration,'scenes':summaries,'human_verdict':None},ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()

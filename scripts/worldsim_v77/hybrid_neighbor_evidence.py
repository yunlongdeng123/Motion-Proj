"""r12：保留非目标actor的原RGB/Ω/LiDAR证据；四查询强控制，不生成。"""
import sys, time, datetime
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from repair_common import camera, hull_mask, VIDEO
from geometry import box_mask, unproject
from video_review import scene_frame
from r3_visibility import segment_box_occlusion
from scipy.spatial import cKDTree
from PIL import ImageDraw

cv2.setNumThreads(4)
BASE=ROOT.parent; R2=ROOT; ROOT=BASE/'r12'; NAME='scene_0255'; ACTOR='52'

def prepare_sources(spec, inst, out, depth_transform=None):
    data=Path(spec['data']); sources=[]; stats=[]
    # 包括缺少整车可见参考的原角度和后续较可见角度；范围预登记，不挑生成效果。
    for f in range(65,156,5):
        fr=scene_frame(spec['spec'],f,inst); b=next(b for b in fr['all_boxes'] if b['actor_id']==ACTOR)
        fd=VIDEO/NAME/'frames'/f'{f:03}'
        metrics=read(fd/'metrics.json'); pred=Path(metrics['prediction_path'])
        with np.load(pred) as p: depth=p['depth'][0,...,0]
        depth=depth*read(fd/'alignment.json')['calibrated_global_depth_scale']
        lid=transform(np.fromfile(data/'lidar'/f'{f:03}.bin',np.float32).reshape(-1,4)[:,:3],np.loadtxt(data/'lidar_pose'/f'{f:03}.txt'))
        inside=box_mask(lid,np.array(b['pose']),np.array(b['size_lwh'])+np.array([.1,.1,.1]))
        if inside.sum()<3:
            stats.append(dict(frame=f,skip='fewer_than_3_actor_lidar_points',actor_lidar_points=int(inside.sum()))); continue
        tree=cKDTree(lid[inside])
        for cam in range(6):
            c,k=camera(fr,cam,depth.shape[1:]); d=depth[cam]
            rect=project_bbox(b['pose'],b['size_lwh'],c,k,d.shape)
            if rect is None: continue
            if depth_transform is not None:
                d=depth_transform(d,fr,b,cam,c,k,lid[inside])
                if d is None: continue
            pts=unproject(d,k,c); grad=np.maximum(np.abs(np.gradient(d,axis=0)),np.abs(np.gradient(d,axis=1)))
            valid=box_mask(pts.reshape(-1,3),np.array(b['pose']),np.array(b['size_lwh'])+np.array([.1,.1,.1])).reshape(d.shape)
            valid &= (d>1)&(d<60)&np.isfinite(d)&(grad<.3)
            yy,xx=np.where(valid); pts=pts[valid]
            if not len(pts): continue
            keep=np.ones(len(pts),bool)
            # 本对象可以作为证据；其他GT对象，包括待删actor25，仍是源图遮挡物。
            for other in fr['all_boxes']:
                if other['actor_id']==ACTOR: continue
                ids=np.flatnonzero(keep)
                if len(ids): keep[ids] &= ~segment_box_occlusion(c[:3,3],pts[ids],np.array(other['pose']),np.array(other['size_lwh'])+np.array([.1,.1,.1]))
            yy,xx,pts=yy[keep],xx[keep],pts[keep]
            if not len(pts): continue
            distance,_=tree.query(pts,workers=4); keep=distance<=.20
            yy,xx,pts,distance=yy[keep],xx[keep],pts[keep],distance[keep]
            if not len(pts): continue
            im=np.array(Image.open(data/'images'/f'{f:03}_{cam}.jpg').convert('RGB').resize((d.shape[1],d.shape[0]),Image.Resampling.BILINEAR))
            local=transform(pts,np.linalg.inv(np.array(b['pose'])))
            sources.append(dict(local=local.astype('float32'),rgb=im[yy,xx],uv=np.stack([xx,yy],-1).astype('int16'),distance=distance.astype('float32')))
            stats.append(dict(source_id=len(sources)-1,frame=f,camera=cam,actor=ACTOR,points=len(pts),actor_lidar_points=int(inside.sum()),prediction_path=str(pred),depth_hw=list(d.shape)))
        print('donor',f,'total accepted views',len(sources),flush=True)
    dump(out/'source_index.json',stats)
    np.savez_compressed(out/'sources.npz',**{f'{i}_{k}':v for i,s in enumerate(sources) for k,v in s.items()})
    return sources,[s for s in stats if 'source_id' in s]

def project(src, pose, c, k, roi):
    pts=transform(src['local'],pose); cp=transform(pts,np.linalg.inv(c)); q=cp@k.T
    uv=np.rint(q[:,:2]/np.maximum(q[:,2:],1e-6)).astype(int)
    keep=(cp[:,2]>.5)&(uv[:,0]>=1)&(uv[:,0]<W-1)&(uv[:,1]>=1)&(uv[:,1]<H-1)
    pids=np.flatnonzero(keep); ix=[]; zz=[]; pp=[]
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            x=uv[pids,0]+dx; y=uv[pids,1]+dy; take=roi[y,x]
            ix.append(y[take]*W+x[take]); zz.append(cp[pids[take],2]); pp.append(pids[take])
    ix,zz,pp=np.concatenate(ix),np.concatenate(zz),np.concatenate(pp)
    order=np.argsort(zz); _,first=np.unique(ix[order],return_index=True); sel=order[first]
    return ix[sel],zz[sel],pp[sel]

def main(run_id='r12', source_builder=prepare_sources, registration_extra=None):
    global ROOT
    ROOT=BASE/run_id
    assert not ROOT.exists(); ROOT.mkdir(); out=ROOT/NAME; out.mkdir()
    spec=next(s for s in read(R2/'registration.json')['scenes'] if s['name']==NAME)
    cfg=dict(task_id='WS-V77-HYBRID-BG-20260927',run_id=run_id,registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hypothesis='此前排除全部车辆丢弃了actor-specific DELETE可用的邻车证据；保留actor52检验真实支持量',
        role='CPU evidence diagnostic, not a generated or admitted DELETE result',source_frames=list(range(65,156,5)),query_indices=[0,7,15,29],actor=ACTOR,target_actor='25',
        inputs='original RGB, cached frozen Omega depth, GT camera/boxes/poses, per-frame raw LiDAR',seed=None,
        config=dict(lidar_distance_m=.20,depth_gradient_max=.30,box_size_padding_m=.10,splat=3,source_time_gap_frames=5,depth_agreement_m=.25,rgb_max_error=25,erosion=3),
        resources=dict(gpu_forward=0,cpu_threads=4),human_verdict=None,failure_ledger_refs=['V77-F02'])
    if registration_extra: cfg.update(registration_extra)
    dump(ROOT/'registration.json',cfg); started=time.time()
    inst=read(Path(spec['data'])/'instances/instances_info.json'); sources,index=source_builder(spec,inst,out)
    rows=[]; sheet=Image.new('RGB',(640*3,360*4),(15,20,25))
    for row,i in enumerate(cfg['query_indices']):
        f=spec['source_frames'][i]; fr=scene_frame(spec['spec'],f,inst); c,k=camera(fr,spec['camera'],HW)
        b=next(b for b in fr['all_boxes'] if b['actor_id']==ACTOR); pose=np.array(b['pose'])
        core=mask(OLD/NAME/'sam'/f'core_{i:05}.png'); roi=mask(BASE/'r3'/NAME/'condition'/f'{i:05}.png')
        orig=rgb(R2/NAME/'rgb'/f'{i:05}.png'); zbest=np.full(H*W,np.inf); col=np.zeros((H*W,3),np.uint8); sid=np.full(H*W,-1,np.int16); pid=np.full(H*W,-1,np.int32)
        projected=[]
        for j,src in enumerate(sources):
            ix,z,pp=project(src,pose,c,k,roi)
            pts=transform(src['local'][pp],pose); keep=np.ones(len(ix),bool)
            # QUERY仅删除actor25；其余物体仍可遮挡actor52。
            for other in fr['all_boxes']:
                if other['actor_id'] in [ACTOR,spec['actor']]: continue
                ids=np.flatnonzero(keep)
                if len(ids):keep[ids]&=~segment_box_occlusion(c[:3,3],pts[ids],np.array(other['pose']),np.array(other['size_lwh']))
            ix,z,pp=ix[keep],z[keep],pp[keep]; projected.append((ix,z,pp)); better=z<zbest[ix]; use=ix[better]
            zbest[use]=z[better]; col[use]=src['rgb'][pp[better]]; sid[use]=j; pid[use]=pp[better]
        second=np.full(H*W,-1,np.int16); sourceframes=np.array([s['frame'] for s in index]+[-10000]); firstf=sourceframes[sid]
        for j,(ix,z,pp) in enumerate(projected):
            agree=(np.abs(z-zbest[ix])<=.25)&(np.max(np.abs(sources[j]['rgb'][pp].astype('int16')-col[ix].astype('int16')),axis=-1)<=25)&(np.abs(index[j]['frame']-firstf[ix])>=5)
            second[ix[agree]]=j
        candidate=(sid>=0).reshape(HW); agree=(second>=0).reshape(HW); accepted=cv2.erode(agree.astype('uint8'),np.ones((3,3),np.uint8))>0
        # foreground fence未完整建模，不把支持自动升级为事实准入。
        protect=mask(R2/NAME/'protect'/f'{i:05}.png'); accepted &= ~protect
        proposal=orig.copy(); proposal[roi]=[70,50,80]; proposal[accepted]=col.reshape(H,W,3)[accepted]
        Image.fromarray(proposal).save(out/f'proposal_{i:05}.png');write_mask(out/f'candidate_{i:05}.png',candidate);write_mask(out/f'accepted_{i:05}.png',accepted)
        np.savez_compressed(out/f'provenance_{i:05}.npz',source_id=sid.reshape(HW),source_point=pid.reshape(HW),second_source=second.reshape(HW),z=zbest.reshape(HW),rgb=col.reshape(H,W,3))
        rect=project_bbox(b['pose'],b['size_lwh'],c,k,HW)
        y,x=np.where(core);left=max(0,int(x.min())-70);top=max(0,int(y.min())-50);box=(left,top,min(W,int(x.max())+100),min(H,int(y.max())+65))
        for j,(title,im) in enumerate([('original target25',orig),('r3 candidate',rgb(BASE/'r3'/NAME/'final'/f'{i:05}.png')),('actor52 raw evidence / purple unknown',proposal)]):
            image=Image.fromarray(im); draw=ImageDraw.Draw(image);draw.rectangle(rect,outline='cyan',width=1)
            image=image.crop(box).resize((640,360));ImageDraw.Draw(image).text((5,5),f'{title} f{f}',fill='yellow');sheet.paste(image,(j*640,row*360))
        rec=dict(frame=f,target_core_pixels=int(core.sum()),single_source_core_pixels=int((candidate&core).sum()),two_source_core_pixels=int((agree&core).sum()),filtered_core_pixels=int((accepted&core).sum()),filtered_roi_pixels=int(accepted.sum()),unverified_static_occlusion=True)
        rows.append(rec);print(rec,flush=True)
    sheet.save(out/'neighbor_evidence_probe.jpg',quality=96)
    dump(ROOT/'summary.json',dict(scenes=[NAME],queries=rows,source_views=len(sources),source_points=sum(len(s['local']) for s in sources),elapsed_s=time.time()-started,
        conclusion='Requires visual review; boxed, LiDAR-supported points are still not hidden-background GT',human_verdict=None,background_input_dir=None,failure_ledger_refs=['V77-F02']))
    print('NEIGHBOR EVIDENCE PROBE COMPLETE',flush=True)

if __name__=='__main__':main()

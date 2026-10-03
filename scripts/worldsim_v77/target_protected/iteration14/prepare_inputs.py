"""CPU构建多先验请求；目标RGB始终先擦除，Y不进入参考/BEV/控制图。"""
from common import *
from collections import defaultdict
import numpy as np, cv2
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
from actor_state import Observation, cuboid_front_depth, project_world
from geometry_factory import transform
cv2.setNumThreads(1)
SIZE=(144,256)


def observation(frame, rgb, hole):
    K=np.array(frame['intrinsics_1024'])/4;K[2]=[0,0,1]
    return Observation(cv2.resize(rgb,(256,144)),cv2.resize(hole.astype('float32'),(256,144),interpolation=cv2.INTER_AREA)>0,
        np.array(frame['camera_to_world']),K,frame['timestamp'])


def box_axis(actor, obs):
    R=Quaternion(actor['rotation']).rotation_matrix
    points=np.array(actor['translation'])+np.array([[-actor['size'][1]/2,0,0],[actor['size'][1]/2,0,0]])@R.T
    C=obs.camera_to_world;cam=(points-C[:3,3])@C[:3,:3];pix=cam@obs.K.T
    uv=pix[:,:2]/np.maximum(pix[:,2:],1e-5);v=uv[1]-uv[0]
    if cam[:,2].min()<=.5 or np.linalg.norm(v)<1e-5:return np.array([0.,0.]),uv,False
    return v/np.linalg.norm(v),uv,True


def background_points(case):
    scans=case.get('bev_scans',case['scans']);parts=[]
    for scan in scans:
        if not scan.get('path') or not Path(scan['path']).is_file():continue
        raw=np.fromfile(scan['path'],np.float32).reshape(-1,5)[:,:3]
        M=transform(scan['ego_pose']['translation'],scan['ego_pose']['rotation'])@transform(scan['calibrated_sensor']['translation'],scan['calibrated_sensor']['rotation'])
        world=raw@M[:3,:3].T+M[:3,3]
        # 动态返回按来源时刻的3D框去掉；不能把已经离开的车身点误当墙/路。
        actors=scan.get('actors')
        if actors is None:
            frame=min(case['frames'],key=lambda f:abs(f['timestamp']-scan['timestamp']));actors=frame['actors']
        keep=np.ones(len(world),bool)
        for actor in actors:
            local=(world-np.array(actor['translation']))@Quaternion(actor['rotation']).rotation_matrix
            dims=np.array([actor['size'][1],actor['size'][0],actor['size'][2]])/2+.2
            keep &= ~np.all(abs(local)<dims,axis=1)
        parts.append(world[keep])
    return np.concatenate(parts) if parts else np.empty((0,3))


def bev(case, points, center):
    result=np.zeros((len(case['frames']),10,128,128),np.float32)
    origin=center-40
    def uv(xy):return np.floor((xy-origin)/80*128).astype(int)
    ij=uv(points[:,:2]);good=np.all((ij>=0)&(ij<128),axis=1);ij=ij[good];p=points[good]
    for i,frame in enumerate(case['frames']):
        a=result[i]
        # 三个高度层的真实返回不等于连续表面，也不认证空白网格为空地。
        ground_z=np.array(frame['camera_to_world'])[2,3]-1.5
        height=p[:,2]-ground_z
        for channel,keep in [(2,height<.6),(3,(height>=.6)&(height<2)),(4,height>=2),(5,np.ones(len(p),bool))]:
            a[channel,ij[keep,1],ij[keep,0]]=1
        for actor in frame['actors']:
            if actor['instance_token']==case['target_token']:continue
            R=Quaternion(actor['rotation']).rotation_matrix
            dims=np.array([actor['size'][1],actor['size'][0]])/2
            corners=np.array([[1,1],[-1,1],[-1,-1],[1,-1]])*dims
            xy=(np.c_[corners,np.zeros(4)]@R.T+np.array(actor['translation']))[:,:2]
            mask=np.zeros((128,128),'uint8');cv2.fillPoly(mask,[uv(xy).astype('int32')],1);mask=mask>0
            channel=0 if actor['category'].startswith('vehicle.') else 1
            a[channel,mask]=1;a[9,mask]=.5
            if channel==0:a[7,mask]=R[0,0];a[8,mask]=R[1,0]
        a[6]=1-np.clip(a[0]+a[1]+a[5],0,1)
        a[9,a[5]>0]=np.maximum(a[9,a[5]>0],.8)
    return result


def ray_grid(frames,center):
    yy,xx=np.mgrid[:18,:32];pixels=np.c_[(xx.ravel()+.5)*32,(yy.ravel()+.5)*32,np.ones(xx.size)]
    output=[]
    for fr in frames:
        C=np.array(fr['camera_to_world']);rays=pixels@np.linalg.inv(fr['intrinsics_1024']).T@C[:3,:3].T
        points=C[:3,3]+rays[None]*np.array([4,12,25,40])[:,None,None]
        # align_corners=False下，范围端点映射到[-1,1]；未知射线保留多深度假设。
        output.append(((points[...,:2]-(center-40))/80*2-1).reshape(4,18,32,2))
    return np.array(output,np.float32)


def letterbox(rgb,hole,roi):
    x0,y0,x1,y1=roi;im=rgb[y0:y1,x0:x1];h=hole[y0:y1,x0:x1]
    scale=min(256/im.shape[1],256/im.shape[0]);width=max(1,round(im.shape[1]*scale));height=max(1,round(im.shape[0]*scale))
    x=(256-width)//2;y=(256-height)//2
    out=np.full((256,256,3),127,'uint8');valid=np.zeros((256,256),'uint8')
    out[y:y+height,x:x+width]=cv2.resize(im,(width,height),interpolation=cv2.INTER_AREA)
    hh=cv2.resize(h.astype('float32'),(width,height),interpolation=cv2.INTER_AREA)>0
    valid[y:y+height,x:x+width]=~hh;out[valid==0]=127
    return out,valid,{'scale':scale,'padding_xy':[x,y],'roi_xyxy':roi}


def reference_candidates(c,x,h,protected):
    result=[]
    for index,ref in enumerate(c['references']):
        frame=ref['frame']
        if c['kind']=='synthetic':
            rgb=x[ref['source_frame']].copy();hole=h[ref['source_frame']]
        else:
            if not ref.get('path'):continue
            rgb=np.asarray(Image.open(ref['path']).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS))
            if ref['mask_path']:
                hole=np.asarray(Image.open(ref['mask_path']).convert('L'))>0
            else:
                dummy=Observation(rgb,np.zeros((576,1024),bool),np.array(frame['camera_to_world']),np.array(frame['intrinsics_1024']),frame['timestamp'])
                target=[a for a in frame['actors'] if a['instance_token']==c['target_token']]
                hole=cv2.dilate(np.isfinite(cuboid_front_depth(dummy,target)).astype('uint8'),np.ones((13,13),'uint8'))>0
        rgb=rgb.copy();rgb[hole]=127
        ob=observation(frame,rgb,hole);front=cuboid_front_depth(ob,frame['actors'])
        result.append({'ref_index':index,'rgb':rgb,'hole':hole,'roi':[0,0,1024,576],
                       'token':None,'role':'background_context','quality':0,'ref':ref})
        for a in frame['actors']:
            if a['instance_token'] not in protected:continue
            depth=cuboid_front_depth(ob,[a]);visible=np.isfinite(depth)&(depth<=front+.1)&~ob.hole
            if visible.sum()<8:continue
            yy,xx=np.where(np.isfinite(depth));bb=np.array([xx.min()*4,yy.min()*4,(xx.max()+1)*4,(yy.max()+1)*4])
            cx,cy=(bb[:2]+bb[2:])/2;ww,hh=(bb[2:]-bb[:2])*1.7
            roi=[max(0,int(cx-ww/2)),max(0,int(cy-hh/2)),min(1024,int(cx+ww/2)),min(576,int(cy+hh/2))]
            result.append({'ref_index':index,'rgb':rgb,'hole':hole,'roi':roi,'token':a['instance_token'],
                'actor':a,'role':'protected_actor_appearance','quality':int(visible.sum()),'ref':ref})
    # 只按输入的可见性与时间选择，禁止按生成质量逐例挑参考。
    cars=sorted((v for v in result if v['token']),key=lambda v:(-v['quality'],v['ref_index']))
    chosen=[];used=set();identities=defaultdict(int)
    for v in cars:
        key=(v['token'],v['ref']['camera'],v['ref']['frame']['timestamp'])
        if key in used or identities[v['token']]>=2:continue
        chosen.append(v);used.add(key);identities[v['token']]+=1
        if len(chosen)==4:break
    context=[v for v in result if v['token'] is None]
    for timestamp in [c['frames'][0]['timestamp']-1000000,c['frames'][-1]['timestamp']+1000000]:
        if not context:break
        # 原相机优先，其他相机可用于外观参考；不把它们直接贴回当前图像。
        v=min(context,key=lambda v:(v['ref']['source_kind']!='r21_full_SAM' and c['kind']=='real',abs(v['ref']['frame']['timestamp']-timestamp)))
        chosen.append(v);context.remove(v)
    if not chosen:raise ValueError('没有合法RGB参考')
    while len(chosen)<6:chosen.append(chosen[-1]|{'padding':True})
    return chosen[:6]


def prepare(c):
    cid=c['case_id'];dest=O/'inputs'/cid;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():
        old=read(dest/'result.json')
        if old.get('input_revision')==2:return old
        import shutil
        backup=O/'before_changes/control_mask_coverage'/cid
        backup.mkdir(parents=True,exist_ok=True)
        for name in ['condition.npz','result.json','references.json']:
            if not (backup/name).exists():shutil.copy2(dest/name,backup/name)
    x=images(c,'rgb');h=images(c,'hole')>0;safe=x.copy();safe[h]=127
    obs=[observation(fr,im,hh) for fr,im,hh in zip(c['frames'],safe,h)]
    protected_scores=defaultdict(int)
    for ob,fr in zip(obs,c['frames']):
        for actor in fr['actors']:
            if actor['instance_token']==c['target_token'] or not actor['category'].startswith('vehicle.'):continue
            protected_scores[actor['instance_token']]+=int((np.isfinite(cuboid_front_depth(ob,[actor]))&ob.hole).sum())
    protected=[t for t,n in sorted(protected_scores.items(),key=lambda item:(-item[1],item[0]))[:3] if n>0]
    bg=background_points(c);center=np.array(c['frames'][5]['camera_to_world'])[:2,3]
    geometry=[];rows=[]
    for i,(ob,fr) in enumerate(zip(obs,c['frames'])):
        retained=[a for a in fr['actors'] if a['instance_token']!=c['target_token']]
        front=cuboid_front_depth(ob,retained);vehicle=cuboid_front_depth(ob,[a for a in retained if a['category'].startswith('vehicle.')])
        g=np.zeros((12,*SIZE),np.float32);g[0]=np.isfinite(vehicle);g[4]=np.where(np.isfinite(vehicle),np.clip(vehicle/60,0,1),0)
        for actor in retained:
            if not actor['category'].startswith('vehicle.'):continue
            d=cuboid_front_depth(ob,[actor]);m=np.isfinite(d)&(d<=front+.05);axis,uv,valid=box_axis(actor,ob)
            if valid:g[5,m]=axis[0];g[6,m]=axis[1]
            if actor['instance_token'] in protected:g[7,m]=1
        uv,z,good=project_world(bg,ob);ids=np.flatnonzero(good);u,v=uv[ids].T
        ids=ids[~np.isfinite(front[v,u])]
        order=np.argsort(z[ids],kind='stable');ids=ids[order];flat=uv[ids,1]*256+uv[ids,0]
        _,first=np.unique(flat,return_index=True);ids=ids[first];u,v=uv[ids].T
        g[1,v,u]=1;g[8,v,u]=np.clip(z[ids]/60,0,1);g[9,v,u]=.8
        g[2]=1-np.clip(g[0]+g[1],0,1);g[3]=np.maximum(g[0]*.5,g[1]*.8)
        g[10]=g[0]*(~ob.hole);g[11]=1
        assert not np.any((g[0]>0)&(g[1]>0))
        geometry.append(g);rows.append({'frame':i,'H_pixels':int(h[i].sum()),'control_H_cells':int(ob.hole.sum()),
            'retained_envelope_cells_H':int(((g[0]>0)&ob.hole).sum()),'LiDAR_background_cells_H':int(((g[1]>0)&ob.hole).sum())})
    grid=bev(c,bg,center);chosen=reference_candidates(c,x,h,protected);rr=[];vv=[];poses=[];info=[]
    anchor=np.array(c['frames'][5]['camera_to_world']);ob_anchor=obs[5]
    for i,v in enumerate(chosen):
        rgb,valid,warp=letterbox(v['rgb'],v['hole'],v['roi'])
        if v.get('padding'):valid[:]=0;rgb[:]=127
        C=np.array(v['ref']['frame']['camera_to_world']);delta=(C[:3,3]-anchor[:3,3])@anchor[:3,:3]/40
        forward=C[:3,2]@anchor[:3,:3];time=(v['ref']['frame']['timestamp']-c['frames'][0]['timestamp'])/3e6
        appearance=[0.,0.,0.];axis=[0.,0.]
        if v.get('actor'):
            p=(np.array(v['actor']['translation'])-anchor[:3,3])@anchor[:3,:3];uv=ob_anchor.K@p
            appearance=[float(uv[0]/max(uv[2],.5)/256),float(uv[1]/max(uv[2],.5)/144),float(p[2]/60)]
            axis=box_axis(v['actor'],ob_anchor)[0].tolist()
        poses.append(np.r_[delta,forward,time,appearance,float(v['token'] is not None),axis].astype('float32'))
        rr.append(rgb);vv.append(valid);Image.fromarray(rgb).save(dest/f'reference_{i:02}.png')
        info.append({k:v[k] for k in ['ref_index','token','role','quality']}|{'reference_slot':i,'camera':v['ref']['camera'],
            'timestamp':v['ref']['frame']['timestamp'],'source_kind':v['ref']['source_kind'],'source_path':v['ref'].get('path'),
            'source_mask_path':v['ref'].get('mask_path'),'source_frame':v['ref'].get('source_frame'),
            'source_frame_geometry':v['ref']['frame'],'GPU_source_mask_validation_pending':v['ref'].get('auxiliary_mask_requires_GPU_validation',False),
            'letterbox':warp,'padding':v.get('padding',False)})
    geometry=np.array(geometry);hh=np.array([ob.hole for ob in obs])
    parameters=np.tile(np.array([1,1,1.7,1],np.float32),(10,1)) # DELETE / 非因果离线prior / ROI上下文倍率 / 保留身份
    np.savez_compressed(dest/'condition.npz',geometry=geometry,bev=grid,bev_rays=ray_grid(c['frames'],center),
        references=np.array(rr),reference_valid=np.array(vv),reference_pose=np.array(poses),
        query_time=np.array([(fr['timestamp']-c['frames'][0]['timestamp'])/1e6 for fr in c['frames']],np.float32)[:,None],
        parameters=parameters,hole=hh)
    for f in [0,5,9]:
        Image.fromarray(safe[f]).save(dest/f'target_{f:02}.png')
        Image.fromarray(h[f].astype('uint8')*255).save(dest/f'mask_{f:02}.png')
        color=np.full((*SIZE,3),45,'uint8');color[geometry[f,0]>0]=[50,195,115];color[geometry[f,1]>0]=[65,150,255]
        Image.fromarray(cv2.resize(color,(1024,576),interpolation=cv2.INTER_NEAREST)).save(dest/f'geometry_{f:02}.png')
        display=np.full((128,128,3),45,'uint8');display[grid[f,5]>0]=[65,150,255];display[grid[f,0]>0]=[50,195,115];display[grid[f,1]>0]=[180,140,70]
        Image.fromarray(cv2.resize(display[::-1],(640,640),interpolation=cv2.INTER_NEAREST)).save(dest/f'bev_{f:02}.png')
    prompt={'task':'DELETE指定目标，恢复被挡住的原场景；保持所有protected身份、位置和外观。',
        'target_instance':c['target_token'],'protected_instances':protected,'input_priority':['observed_RGB','projected_geometry','scene_context','generative_prior'],
        'regions':'绿为保留车辆3D包络，不是完整真实轮廓；蓝为实测背景落点；灰未知，不等同无车。',
        'restrictions':['不得将待删目标参考当生成对象','不得把整洞填为车','洞外只用原始像素','未知背景不能宣称已观测'],
        'text_is_audit_description':True,'network_receives_four_compiled_numeric_parameters':True}
    dump(dest/'instruction.json',prompt);dump(dest/'references.json',{'references':info})
    result={'case_id':cid,'input_revision':2,'kind':c['kind'],'scene':c['scene'],'split':c['split'],'frames':rows,'protected_instances':protected,
        'reference_bank':len(c['references']),'selected_references':len(info),'selected_distinct_cameras':sorted({r['camera'] for r in info if not r['padding']}),
        'source_mask_GPU_checks':sum(r['GPU_source_mask_validation_pending'] and not r['padding'] for r in info),
        'background_3D_points':len(bg),'geometry_proxy_not_silhouette':True,'Y_read_by_condition_builder':False,
        'edit_mask_policy':'unchanged r21 full SAM / unchanged original synthetic hole',
        'reference_RGB_erased_before_resize':True,'missing_inputs':0,'GPU_jobs':0,'human_verdict':None}
    dump(dest/'result.json',result);return result


def main():
    plan=read(O/'manifest.json');results=[];pending=[]
    for c in plan['cases']:
        if c['kind']=='real' and (any(not r.get('path') for r in c['references']) or any(not s.get('path') for s in c['bev_scans'])):
            pending.append(c['case_id']);continue
        results.append(prepare(c));dump(O/'prepare_state.json',{'stage':'building','completed':len(results),'pending':pending,'pid':os.getpid()})
        print('INPUT',c['case_id'],flush=True)
    dump(O/'input_summary.json',{'cases':results,'pending':pending,'GPU_jobs':0})
    dump(O/'prepare_state.json',{'stage':'waiting_CPU_extra_inputs' if pending else 'CPU_inputs_ready_GPU_not_run',
        'completed':len(results),'pending':pending,'GPU_jobs':0,'human_verdict':None})


if __name__=='__main__': main()

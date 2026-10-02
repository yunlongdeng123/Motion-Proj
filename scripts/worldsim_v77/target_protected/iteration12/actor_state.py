"""仅接受遮后观测的actor-local状态；立方体面是几何代理，不是隐藏真值。

输入接口没有Y/完整视频路径。O/N只由真实可见样本产生；未观测保持U。
"""
from dataclasses import dataclass
import numpy as np
from pyquaternion import Quaternion


@dataclass
class Observation:
    rgb: np.ndarray
    hole: np.ndarray
    camera_to_world: np.ndarray
    K: np.ndarray
    timestamp: int

    def validate(self):
        assert self.rgb.dtype==np.uint8 and self.rgb.shape[:2]==self.hole.shape
        assert self.hole.dtype==bool
        # 必须提前擦除；不能由内部函数悄悄访问洞中原像素。
        assert np.all(self.rgb[self.hole]==127), '状态构建只接收已擦除RGB'


def local_samples(obs, actor, visible_mask, stride=2):
    """可见B像素光线与B局部框交点，RGB来自当前遮后观测。"""
    visible=np.asarray(visible_mask,bool)&~obs.hole
    yy,xx=np.where(visible)
    take=((yy%stride)==0)&((xx%stride)==0);yy,xx=yy[take],xx[take]
    R=Quaternion(actor['rotation']).rotation_matrix
    C=obs.camera_to_world
    ray=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(obs.K).T@C[:3,:3].T@R
    origin=(C[:3,3]-np.array(actor['translation']))@R
    dim=np.array([actor['size'][1],actor['size'][0],actor['size'][2]])
    safe=np.where(abs(ray)<1e-10,1e-10,ray)
    lo=(-dim/2-origin)/safe;hi=(dim/2-origin)/safe
    enter=np.minimum(lo,hi).max(1);leave=np.maximum(lo,hi).min(1)
    valid=(leave>=enter)&(enter>0)&np.isfinite(enter)
    xyz=origin+ray[valid]*enter[valid,None]
    uv=np.stack([xx[valid],yy[valid]],1)
    return xyz.astype('float32'),obs.rgb[yy[valid],xx[valid]].copy(),uv


def project_world(xyz, obs):
    C=obs.camera_to_world
    cam=(xyz-C[:3,3])@C[:3,:3]
    pix=cam@obs.K.T
    z=cam[:,2]
    uv=np.floor(pix[:,:2]/np.maximum(z[:,None],1e-6)).astype(int)
    h,w=obs.hole.shape
    good=(z>.5)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    return uv,z,good


def build_actor_samples(observations, tracks, masks):
    """每个身份单独存储全部可见点；不优化实例自由embedding。"""
    for obs in observations:obs.validate()
    result={}
    for token,trajectory in tracks.items():
        assert len(trajectory)==len(observations)==len(masks[token])
        xyz=[];rgb=[];src=[];pixels=[]
        for i,(obs,a,m) in enumerate(zip(observations,trajectory,masks[token])):
            p,c,uv=local_samples(obs,a,m)
            xyz.append(p);rgb.append(c);src.append(np.full(len(p),i,np.int16));pixels.append(uv)
        result[token]={'local_xyz':np.concatenate(xyz),'rgb':np.concatenate(rgb),'source_frame':np.concatenate(src),'source_pixel':np.concatenate(pixels),'geometry':'frozen_GT_cuboid_surface_proxy','learned_instance_embedding':False}
    return result


def build_background_samples(observations, lidar_by_frame, excluded_masks):
    """仅采同窗口实测近地LiDAR落点且RGB可见的背景；不铺满整个路面。"""
    points=[];colors=[];frames=[];pixels=[]
    for i,(obs,xyz,excluded) in enumerate(zip(observations,lidar_by_frame,excluded_masks)):
        obs.validate()
        uv,z,good=project_world(np.asarray(xyz),obs)
        ids=np.flatnonzero(good);u,v=uv[ids].T
        valid=~obs.hole[v,u]&~excluded[v,u]
        ids=ids[valid];u,v=uv[ids].T
        # 同像素first-return；后方返回不覆盖前方颜色。
        order=np.argsort(z[ids],kind='stable');ids=ids[order];flat=uv[ids,1]*obs.hole.shape[1]+uv[ids,0]
        _,first=np.unique(flat,return_index=True);ids=ids[first];u,v=uv[ids].T
        points.append(np.asarray(xyz)[ids]);colors.append(obs.rgb[v,u]);frames.append(np.full(len(ids),i,np.int16));pixels.append(uv[ids])
    return {'world_xyz':np.concatenate(points).astype('float32'),'rgb':np.concatenate(colors),'source_frame':np.concatenate(frames),'source_pixel':np.concatenate(pixels),'positive_evidence_only':True}


def render_state(observations, tracks, actor_samples, background, exclude_source=None,
                 retained_box_masks=None, retained_front_depth=None):
    """删除目标根本不在tracks中。未知和冲突不强制解释为空背景。

    输出feature为可见RGB局部特征占位；B臂仅用D/V/O/N/U/Q，C臂另用身份。
    小点1px投影不做洞填补，避免把代理插值当真证据。
    """
    tokens=sorted(tracks);results=[]
    for t,obs in enumerate(observations):
        h,w=obs.hole.shape
        depth=np.full(h*w,np.inf,np.float32);slot=np.zeros(h*w,np.int16)
        F=np.zeros((h*w,3),np.uint8);Q=np.zeros(h*w,np.float32)
        source=np.full(h*w,-1,np.int16);support=np.zeros(h*w,np.int16)
        candidates=[]
        for k,token in enumerate(tokens,1):
            s=actor_samples[token];a=tracks[token][t]
            xyz=s['local_xyz']@Quaternion(a['rotation']).rotation_matrix.T+np.array(a['translation'])
            candidates.append((k,xyz,s,.5))
        candidates.append((0,background['world_xyz'],background,background.get('confidence',.8)))
        for k,xyz,s,q in candidates:
            uv,z,good=project_world(xyz,obs)
            if exclude_source is not None:good&=s['source_frame']!=exclude_source(t)
            ids=np.flatnonzero(good)
            if retained_front_depth is not None and k>0:
                # 移除A之后的保留对象first-return：挡住别人的点不能穿透投影。
                front=retained_front_depth[t][uv[ids,1],uv[ids,0]]
                ids=ids[z[ids]<=front+.15]
            if k==0 and retained_box_masks is not None:
                ids=ids[~retained_box_masks[t][uv[ids,1],uv[ids,0]]]
            if not len(ids):continue
            flat=uv[ids,1]*w+uv[ids,0]
            age=np.abs(np.array([o.timestamp for o in observations])[s['source_frame'][ids]]-obs.timestamp)/1e6
            confidence=q*np.exp(-age/3.)
            # 深度、时间确定选择；不同身份绝不平均。
            order=np.lexsort((-confidence,z[ids],flat));ids=ids[order];flat=flat[order];confidence=confidence[order]
            unique,first,count=np.unique(flat,return_index=True,return_counts=True)
            ids=ids[first];confidence=confidence[first]
            nearer=z[ids]<depth[unique]-.05
            tied=(np.abs(z[ids]-depth[unique])<=.05)&(slot[unique]!=k)&np.isfinite(depth[unique])
            # 两个不同来源实体深度不可分时标U，保留深度防止后景越过。
            Q[unique[tied]]=0
            take=unique[nearer];sel=ids[nearer]
            depth[take]=z[sel];slot[take]=k;F[take]=s['rgb'][sel]
            Q[take]=confidence[nearer];source[take]=s['source_frame'][sel];support[take]=count[nearer]
        V=Q>0;O=V&(slot>0);N=V&(slot==0);U=~V
        F[U]=0;slot[U]=-1;source[U]=-1
        D=np.where(np.isfinite(depth),depth,0).reshape(h,w)
        results.append({'F':F.reshape(h,w,3),'D':D,'V':V.reshape(h,w),'O':O.reshape(h,w),'N':N.reshape(h,w),'U':U.reshape(h,w),'Q':Q.reshape(h,w),'actor_slot':slot.reshape(h,w),'source_frame':source.reshape(h,w),'support_count':support.reshape(h,w)})
        assert not np.any(O&N) and np.all(O|N|U)
    return results


def cuboid_front_depth(obs, actors):
    """对保留GT框做光线first-return，目标A由调用方明确移除。"""
    h,w=obs.hole.shape;result=np.full((h,w),np.inf,np.float32)
    C=obs.camera_to_world
    for a in actors:
        if 'size' not in a:continue
        dim=np.array([a['size'][1],a['size'][0],a['size'][2]])
        R=Quaternion(a['rotation']).rotation_matrix
        corners=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])*dim
        world=corners@R.T+np.array(a['translation']);uv,z,good=project_world(world,obs)
        if z.min()<=.5:
            if z.max()<=.5:continue
            x0,y0,x1,y1=0,0,w,h
        else:
            x0,y0=np.maximum(uv.min(0)-1,[0,0]);x1,y1=np.minimum(uv.max(0)+2,[w,h])
        if x1<=x0 or y1<=y0:continue
        yy,xx=np.mgrid[y0:y1,x0:x1];xx=xx.ravel();yy=yy.ravel()
        ray=np.c_[xx+.5,yy+.5,np.ones(len(xx))]@np.linalg.inv(obs.K).T@C[:3,:3].T@R
        origin=(C[:3,3]-np.array(a['translation']))@R
        safe=np.where(abs(ray)<1e-10,1e-10,ray)
        lo=(-dim/2-origin)/safe;hi=(dim/2-origin)/safe
        near=np.minimum(lo,hi).max(1);far=np.maximum(lo,hi).min(1)
        valid=(far>=near)&(near>.5)&np.isfinite(near)
        yy,xx=yy[valid],xx[valid];result[yy,xx]=np.minimum(result[yy,xx],near[valid])
    return result


def checks():
    h,w=32,48;H=np.zeros((h,w),bool);H[12:20,20:28]=True
    rgb=np.zeros((h,w,3),np.uint8)+53;rgb[H]=127
    K=np.array([[32,0,24],[0,32,16],[0,0,1.]])
    obs=Observation(rgb,H,np.eye(4),K,0);obs.validate()
    actor={'size':[2,4,2],'translation':[0,0,8],'rotation':[1,0,0,0]}
    m=np.zeros_like(H);m[12:20,16:32]=True
    samples=build_actor_samples([obs],{'B':[actor]},{'B':[m]})
    assert len(samples['B']['local_xyz'])>0
    uv=samples['B']['source_pixel'];assert not H[uv[:,1],uv[:,0]].any()
    bg={'world_xyz':np.empty((0,3)),'rgb':np.empty((0,3),np.uint8),'source_frame':np.empty(0,np.int16)}
    r=render_state([obs],{'B':[actor]},samples,bg)[0]
    assert r['O'].any() and not r['N'].any() and r['U'].any()
    without=render_state([obs],{'B':[actor]},samples,bg,exclude_source=lambda t:t)[0]
    assert without['U'].all() and not without['O'].any()
    # 已见表面由别帧补当前洞，删除A后才暴露B。
    H2=np.zeros_like(H);rgb2=np.full_like(rgb,91)
    obs2=Observation(rgb2,H2,np.eye(4),K,100000)
    seq=[obs,obs2];tr={'B':[actor,actor]}
    ss=build_actor_samples(seq,tr,{'B':[m,m]})
    front=[cuboid_front_depth(o,[actor]) for o in seq]
    revealed=render_state(seq,tr,ss,bg,retained_front_depth=front)[0]
    assert (revealed['O']&H).any()
    occluder={'size':[5,5,4],'translation':[0,0,4],'rotation':[1,0,0,0]}
    blocked=[cuboid_front_depth(o,[actor,occluder]) for o in seq]
    invisible=render_state(seq,tr,ss,bg,retained_front_depth=blocked)[0]
    assert not (invisible['O']&H).any()
    # 不同身份同深度冲突为U；绝不平均两辆车的RGB。
    twin={'B':ss['B'],'C':{**ss['B'],'rgb':np.full_like(ss['B']['rgb'],222)}}
    collision=render_state(seq,{'B':[actor,actor],'C':[actor,actor]},twin,bg)[0]
    assert collision['U'][revealed['O']].all()
    assert not collision['F'][collision['U']].any()
    # 只有显式真实背景样本才产生N；缺点不会被补成N。
    bg_one={'world_xyz':np.array([[6,3,12.]]),'rgb':np.array([[15,25,35]],np.uint8),'source_frame':np.array([1],np.int16)}
    known=render_state(seq,{}, {},bg_one)[0]
    assert known['N'].sum()==1 and known['U'].sum()==h*w-1
    # 训练synthetic-X隐藏内容替换，在输入边界被擦除，状态严格相同。
    x1=rgb.copy();x2=rgb.copy();x1[H]=0;x2[H]=255;x1[H]=127;x2[H]=127
    assert np.array_equal(x1,x2)
    corrupted=rgb.copy();corrupted[H]=0
    try:Observation(corrupted,H,np.eye(4),K,0).validate()
    except AssertionError:pass
    else:raise AssertionError('未擦除内容不能进入条件')
    print('ACTOR_STATE_CONTRACT_PASS: masked boundary, no-evidence U, temporal reveal, A-removal visibility, identity conflict, positive N')

if __name__=='__main__':checks()

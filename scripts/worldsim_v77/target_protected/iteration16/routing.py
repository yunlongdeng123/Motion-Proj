"""可见源patch→同一保留实例的框面代理；不读取Y，不铺造背景。"""
import numpy as np
import cv2
from pyquaternion import Quaternion
from actor_state import Observation, cuboid_front_depth, local_samples

SIZE = (144,256)
PURITY = .95  # 沿用参考token有效门槛；混合/冲突保持未知，不逐case调节。


def observation(frame):
    K = np.array(frame['intrinsics_1024'],float)/4;K[2] = [0,0,1]
    return Observation(np.full((*SIZE,3),127,np.uint8), np.zeros(SIZE,bool),
                       np.array(frame['camera_to_world']),K,frame['timestamp'])


def first_owner(obs, actors, identities, target_token=None, remove_target=False):
    """实体代理竞争first return；近同深度的身份冲突标U。"""
    actors = [a for a in actors if not (remove_target and a['instance_token']==target_token)]
    front = cuboid_front_depth(obs,actors)
    owner = np.zeros(SIZE,np.int32);occupied = np.isfinite(front)
    claims = np.zeros(SIZE,np.uint16)
    for a in actors:
        key = a['instance_token']
        if key not in identities or key==target_token:continue
        d = cuboid_front_depth(obs,[a])
        visible = np.isfinite(d)&(d<=front+.05)
        owner[visible] = identities[key];claims[visible] += 1
    owner[claims>1] = 0
    return owner,occupied,front


def continuous_projection(xyz, obs):
    cam = (xyz-obs.camera_to_world[:3,3])@obs.camera_to_world[:3,:3]
    z = cam[:,2];pixel = cam@obs.K.T
    uv = pixel[:,:2]/np.maximum(z[:,None],1e-6)
    good = (z>.5)&(uv[:,0]>=0)&(uv[:,0]<SIZE[1])&(uv[:,1]>=0)&(uv[:,1]<SIZE[0])
    return uv,z,good


def letterbox_coordinates(info):
    """逆向使用实际round后的宽高；不得只用名义scale累积半像素误差。"""
    x0,y0,x1,y1 = info['letterbox']['roi_xyxy']
    px,py = info['letterbox']['padding_xy'];scale = info['letterbox']['scale']
    width = max(1,round((x1-x0)*scale));height = max(1,round((y1-y0)*scale))
    yy,xx = np.mgrid[:256,:256]
    inside = (xx>=px)&(xx<px+width)&(yy>=py)&(yy<py+height)
    # 像素中心坐标，从256参考坐标逆变换到原1024图像，再到控制网格。
    uv = np.stack([x0+(xx-px+.5)*(x1-x0)/width,
                   y0+(yy-py+.5)*(y1-y0)/height],-1)/4
    return uv,inside


def source_background(obs, occupied, points):
    """同批真实LiDAR落点的可见支持。没有点的区域仍为U。"""
    ids = np.zeros(SIZE,np.int32)-1
    if not len(points):return ids
    uv,z,good = continuous_projection(points,obs);i = np.flatnonzero(good)
    pixel = np.floor(uv[i]).astype(int)
    keep = ~occupied[pixel[:,1],pixel[:,0]];i = i[keep];pixel = pixel[keep]
    order = np.argsort(z[i],kind='stable');i = i[order];pixel = pixel[order]
    flat = pixel[:,1]*SIZE[1]+pixel[:,0];_,first = np.unique(flat,return_index=True)
    pixel = pixel[first];ids[pixel[:,1],pixel[:,0]] = i[first]
    return ids


def build_routing(case, priors, reference_manifest, points):
    """函数只接受几何、已擦除参考valid与LiDAR；没有Y/RGB读取入口。"""
    refs = reference_manifest['references'];frames = case['frames']
    identities = {key:i+1 for i,key in enumerate(sorted({a['instance_token']
        for fr in frames+[r['source_frame_geometry'] for r in refs]
        for a in fr['actors'] if a['instance_token']!=case['target_token']
        and a['category'].startswith('vehicle.')}))}
    inverse = {i:key for key,i in identities.items()}
    r = len(refs);n = r*64;t = len(frames)
    owners = np.zeros((r,8,8),np.int32);rq = np.zeros((r,8,8),np.float32)
    local = np.zeros((r,8,8,3),np.float32);world = np.zeros_like(local)
    dimensions = np.zeros_like(local)
    source_uv = np.zeros((r,8,8,2),np.float32);radius = np.zeros((r,8,8),np.float32)
    query_owner = np.zeros((t,1,*SIZE),np.int32);query_q = np.zeros_like(query_owner,np.float32)
    obsq = [observation(fr) for fr in frames];query_front = []
    for f,(fr,ob) in enumerate(zip(frames,obsq)):
        owner,occupied,front = first_owner(ob,fr['actors'],identities,case['target_token'],True)
        query_front.append(front)
        # N必须是原r47实测背景，排除保留实体；不从框外推出N。
        N = (priors['geometry'][f,1]>0)&~occupied
        owner[N] = -1;query_owner[f,0] = owner
        query_q[f,0,owner>0] = .5;query_q[f,0,N] = .8
    source_rows = []
    for slot,info in enumerate(refs):
        if info['padding']:
            source_rows.append({'slot':slot,'padding':True,'known_actor_patches':0,'known_background_patches':0})
            continue
        fr = info['source_frame_geometry'];ob = observation(fr)
        owner,occupied,front = first_owner(ob,fr['actors'],identities,case['target_token'])
        bg_ids = source_background(ob,occupied,points);owner[bg_ids>=0] = -1
        uv,inside = letterbox_coordinates(info)
        pix = np.floor(uv).astype(int);pix[...,0] = np.clip(pix[...,0],0,255);pix[...,1] = np.clip(pix[...,1],0,143)
        label = owner[pix[...,1],pix[...,0]].copy()
        valid = inside&(priors['reference_valid'][slot]>0)
        label[~valid] = 0
        actors = {identities[a['instance_token']]:a for a in fr['actors'] if a['instance_token'] in identities}
        for y in range(8):
            for x in range(8):
                region = np.s_[y*32:(y+1)*32,x*32:(x+1)*32]
                values,count = np.unique(label[region],return_counts=True)
                j = np.argmax(count);key = int(values[j]);purity = count[j]/1024
                if key==0 or purity<PURITY or not valid[region].all():continue
                # 单个patch不能包含两个已知实例；未知边缘只以purity折减可信度。
                if np.any((values!=key)&(values!=0)):continue
                coordinates = uv[region][label[region]==key]
                center = np.mean(coordinates,axis=0)
                chosen = coordinates[np.argmin(np.square(coordinates-center).sum(1))]
                source_uv[slot,y,x] = chosen*4
                if key>0:
                    m = np.zeros(SIZE,bool)
                    p = np.floor(coordinates).astype(int);m[p[:,1],p[:,0]] = True
                    xyz,_,pixels = local_samples(ob,actors[key],m,stride=1)
                    if not len(xyz):continue
                    j = np.argmin(np.square(pixels+.5-chosen).sum(1));local[slot,y,x] = xyz[j]
                    radius[slot,y,x] = np.linalg.norm(xyz-xyz[j],axis=1).max()
                    a = actors[key];dimensions[slot,y,x] = [a['size'][1],a['size'][0],a['size'][2]]
                    rq[slot,y,x] = .5*purity
                else:
                    p = np.floor(chosen).astype(int);world[slot,y,x] = points[bg_ids[p[1],p[0]]]
                    radius[slot,y,x] = 0;rq[slot,y,x] = .8*purity
                owners[slot,y,x] = key
        source_rows.append({'slot':slot,'padding':False,'role':info['role'],'declared_token':info['token'],
            'known_actor_patches':int((owners[slot]>0).sum()),
            'declared_actor_patches':int((owners[slot]==identities.get(info['token'],0)).sum()) if info['token'] else 0,
            'known_background_patches':int((owners[slot]==-1).sum()),
            'unknown_patches':int((owners[slot]==0).sum())})
    projected_uv = np.zeros((t,n,2),np.float32);projected_radius = np.zeros_like(projected_uv)
    projected_q = np.zeros((t,n),np.float32)
    flat_owner = owners.ravel();flat_local = local.reshape(-1,3);flat_world = world.reshape(-1,3)
    for f,(fr,ob) in enumerate(zip(frames,obsq)):
        actor_map = {a['instance_token']:a for a in fr['actors']}
        for key in np.unique(flat_owner):
            if key==0:continue
            ii = np.flatnonzero(flat_owner==key)
            if key>0:
                a = actor_map.get(inverse[int(key)])
                if a is None:continue  # 缺该时刻位姿就只保留实例归属，不伪造空间对应。
                xyz = flat_local[ii]@Quaternion(a['rotation']).rotation_matrix.T+np.array(a['translation'])
            else:xyz = flat_world[ii]
            uv,z,good = continuous_projection(xyz,ob)
            pixel = np.floor(uv[good]).astype(int);jj = ii[good]
            # 删除A后仍被别的实体挡住的代理点不能当空间对应；身份标签仍可用。
            visible = z[good]<=query_front[f][pixel[:,1],pixel[:,0]]+.1
            jj = jj[visible];zz = z[good][visible];pp = uv[good][visible]
            projected_uv[f,jj] = pp/np.array([256,144])
            projected_radius[f,jj] = (radius.ravel()[jj,None]*np.array([ob.K[0,0]/256,ob.K[1,1]/144])/zz[:,None])
            projected_q[f,jj] = rq.ravel()[jj]
    route = dict(routing_query_owner=query_owner,routing_query_q=query_q,
        routing_reference_owner=owners,routing_reference_q=rq,
        routing_projected_uv=projected_uv,routing_projected_radius=projected_radius,
        routing_projected_q=projected_q)
    auxiliary = dict(reference_local_xyz=local,reference_local_dimensions=dimensions,
                     reference_source_uv=source_uv,reference_world_xyz=world)
    return route,auxiliary,{'identities':identities,'source_slots':source_rows,
        'main_protected_token':next((r['token'] for r in refs if not r['padding'] and r['token']),None),
        'source_patch_purity':PURITY,'cuboid_proxy_not_silhouette':True,
        'unknown_bias_zero':True,'Y_read_by_builder':False}


def mismatch_pairing(left, left_meta, right, right_meta):
    """对双方有效主B patch做固定、无重复配对；位置标签仍是recipient的。"""
    lid = left_meta['identities'][left_meta['main_protected_token']]
    rid = right_meta['identities'][right_meta['main_protected_token']]
    li = np.flatnonzero(left['routing_reference_owner'].ravel()==lid)
    ri = np.flatnonzero(right['routing_reference_owner'].ravel()==rid)
    if not len(li) or not len(ri):raise ValueError('无有效主B外观patch')
    lp = left['reference_local_xyz'].reshape(-1,3)[li]*2/left['reference_local_dimensions'].reshape(-1,3)[li]
    rp = right['reference_local_xyz'].reshape(-1,3)[ri]*2/right['reference_local_dimensions'].reshape(-1,3)[ri]
    # 主车尺寸不同，按归一化框面坐标选近邻；不把不同车的RGB变成真值。
    distance = np.square(lp[:,None]-rp[None]).sum(-1)
    order = np.argsort(distance.ravel(),kind='stable');used_l=set();used_r=set();pairs=[]
    for i in order:
        a,b = np.unravel_index(i,distance.shape)
        if a in used_l or b in used_r:continue
        used_l.add(a);used_r.add(b);pairs.append((int(li[a]),int(ri[b])))
        if len(pairs)==min(len(li),len(ri)):break
    return pairs,{'patches_swapped':len(pairs),'recipient_main_B_patches':len(li),
        'donor_main_B_patches':len(ri),'fraction_of_main_B_patches':len(pairs)/len(li),
        'pairing':'固定归一化actor-local近邻、一对一；双方均是有效主B patch',
        'scope':'仅外观错配；不是完整车身真值或严格同车身表面观测'}


def pooled_owner(owner, confidence, size):
    """同一查询格含不同实体/背景则不指派。U不被归为空地。"""
    import torch
    from torch.nn import functional as F
    output = torch.zeros((owner.shape[0],size[0]*size[1]),dtype=torch.long,device=owner.device)
    q = torch.zeros_like(output,dtype=torch.float32)
    # 身份只是分组索引，不训练自由embedding。
    for key in torch.unique(owner).tolist():
        if key==0:continue
        mask = owner==key;fraction = F.adaptive_avg_pool2d(mask.float(),size).flatten(1)
        support = F.adaptive_avg_pool2d((confidence*mask).float(),size).flatten(1)
        other = F.adaptive_max_pool2d(((owner!=key)&(owner!=0)).float(),size).flatten(1)>0
        keep = (fraction>=PURITY)&~other
        output[keep] = key;q[keep] = support[keep]
    return output,q


def attention_bias(condition, size):
    """只有可靠双方证据参与软偏置；空token和未知维持原来的logit。"""
    import torch
    owner,q = pooled_owner(condition['routing_query_owner'],condition['routing_query_q'],size)
    ro = condition['routing_reference_owner'].reshape(1,1,-1)
    rq = condition['routing_reference_q'].reshape(1,1,-1)
    qq = q[:,:,None];known = (owner[:,:,None]!=0)&(ro!=0)
    confidence = torch.minimum(qq,rq)*known
    matching = owner[:,:,None]==ro
    similarity = matching.float()
    yy,xx = torch.meshgrid(torch.arange(size[0],device=q.device),torch.arange(size[1],device=q.device),indexing='ij')
    centers = torch.stack([(xx+.5)/size[1],(yy+.5)/size[0]],-1).reshape(1,-1,1,2)
    sigma = torch.maximum(condition['routing_projected_radius'][:,None],
                          torch.tensor([1/size[1],1/size[0]],device=q.device))
    distance = ((centers-condition['routing_projected_uv'][:,None])/sigma).square().sum(-1)
    spatial = matching&(condition['routing_projected_q'][:,None]>0)&known
    similarity = torch.where(spatial,torch.exp(-.5*distance),similarity)
    # 有身份却无局部对应的同实例token仅做身份绑定；不虚造位置。
    bias = confidence*torch.log(.1+.9*similarity)
    bias = torch.cat([bias,torch.zeros((*bias.shape[:2],1),device=bias.device)],-1)
    return bias,owner,q

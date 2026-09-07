"""同一显式曲面的有序首交点与直接自由空间监督，不读取外观透明度。"""
import torch


@torch.no_grad()
def _select_first_triangle(vertices,faces,origins,directions,
                           ray_chunk=64,face_chunk=512):
    """双面Möller–Trumbore相交；按场景全局深度取首交点并返回归属。"""
    depths=[]
    owners=[]
    for start in range(0,len(origins),ray_chunk):
        origin=origins[start:start+ray_chunk]
        direction=directions[start:start+ray_chunk]
        best=origin.new_full((len(origin),),float('inf'))
        best_owner=torch.full((len(origin),),-1,device=origin.device,dtype=torch.long)
        for fstart in range(0,len(faces),face_chunk):
            triangle=vertices[faces[fstart:fstart+face_chunk]]
            edge1=triangle[:,1]-triangle[:,0]
            edge2=triangle[:,2]-triangle[:,0]
            pvec=torch.cross(direction[:,None,:],edge2[None],dim=-1)
            determinant=(edge1[None]*pvec).sum(-1)
            valid_det=determinant.abs()>1e-8
            inverse=torch.where(valid_det,determinant,torch.ones_like(determinant)).reciprocal()
            tvec=origin[:,None]-triangle[None,:,0]
            u=(tvec*pvec).sum(-1)*inverse
            qvec=torch.cross(tvec,edge1[None],dim=-1)
            v=(direction[:,None]*qvec).sum(-1)*inverse
            distance=(edge2[None]*qvec).sum(-1)*inverse
            inside=valid_det&(u>=0)&(v>=0)&(u+v<=1)&(distance>0)
            distance=torch.where(inside,distance,torch.full_like(distance,float('inf')))
            local,which=distance.min(-1)
            replace=local<best
            best=torch.minimum(best,local)
            local_owner=which+fstart
            best_owner=torch.where(replace,local_owner,best_owner)
        depths.append(best)
        owners.append(best_owner)
    if not depths:
        return origins.new_empty(0),torch.empty(0,device=origins.device,dtype=torch.long)
    return torch.cat(depths),torch.cat(owners)


def first_triangle_intersection(vertices,faces,origins,directions,face_owner=None,
                                ray_chunk=64,face_chunk=512):
    """离散首面选择后只重算所选相交的梯度，避免保存射线×全曲面的计算图。

    支持内位置梯度准确；支持出生、轮廓跨越与排序切换不因此可微，须另有coverage。
    """
    raw,ids=_select_first_triangle(vertices,faces,origins,directions,ray_chunk,face_chunk)
    valid=ids>=0
    if not valid.any():
        # 保留零梯度通路，使全miss的free项可与其他几何项联合反向。
        return raw+vertices.sum()*0,ids
    triangle=vertices[faces[ids[valid]]]
    a,b,c=triangle.unbind(1)
    normal=torch.cross(b-a,c-a,dim=-1)
    distance=((a-origins[valid])*normal).sum(-1)/(directions[valid]*normal).sum(-1)
    depth=raw.clone()
    depth[valid]=distance
    owner=torch.full_like(ids,-1)
    owner[valid]=face_owner[ids[valid]] if face_owner is not None else 0
    return depth,owner


def closest_surface_points(vertices,faces,points,point_chunk=64,face_chunk=512):
    """全三角面最近点；离散选择/最优重心权重不反传，距离按包络定理反传至顶点。

    不将点集凸包化，不要求闭合曲面。计算量仍O(PF)，通过分块限制临时张量。
    """
    outputs=[]
    if len(faces)==0: raise ValueError('最近曲面查询需要非空三角面')
    for part in points.split(point_chunk):
        with torch.no_grad():
            best=part.new_full((len(part),),float('inf'))
            best_ids=torch.zeros(len(part),device=part.device,dtype=torch.long)
            best_bary=part.new_zeros(len(part),3)
            for start in range(0,len(faces),face_chunk):
                tri=vertices[faces[start:start+face_chunk]].detach()
                a,b,c=tri.unbind(1)
                ab,ac=b-a,c-a
                ap=part[:,None]-a
                d00=(ab*ab).sum(-1); d01=(ab*ac).sum(-1); d11=(ac*ac).sum(-1)
                d20=(ap*ab).sum(-1); d21=(ap*ac).sum(-1)
                denom=d00*d11-d01.square()
                v=(d11*d20-d01*d21)/denom.clamp_min(1e-15)
                w=(d00*d21-d01*d20)/denom.clamp_min(1e-15)
                bary=torch.stack([1-v-w,v,w],-1)
                projection=torch.einsum('pfi,fij->pfj',bary,tri)
                distance=(projection-part[:,None]).square().sum(-1)
                inside=(bary>=0).all(-1)&(denom>1e-15)
                distance=distance.masked_fill(~inside,float('inf'))
                for left,right in [(0,1),(1,2),(2,0)]:
                    edge=tri[:,right]-tri[:,left]
                    t=((part[:,None]-tri[:,left])*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-15)
                    t=t.clamp(0,1)
                    edge_bary=torch.zeros_like(bary)
                    edge_bary[...,left]=1-t; edge_bary[...,right]=t
                    edge_point=tri[None,:,left]+t[:,:,None]*edge
                    edge_distance=(edge_point-part[:,None]).square().sum(-1)
                    change=edge_distance<distance
                    distance=torch.minimum(distance,edge_distance)
                    bary=torch.where(change[:,:,None],edge_bary,bary)
                local,which=distance.min(-1)
                chosen=bary[torch.arange(len(part),device=part.device),which]
                change=local<best
                best=torch.minimum(best,local)
                best_ids=torch.where(change,which+start,best_ids)
                best_bary=torch.where(change[:,None],chosen,best_bary)
        outputs.append((vertices[faces[best_ids]]*best_bary[:,:,None]).sum(1))
    return torch.cat(outputs) if outputs else points.clone()


def direct_free_space_loss(first_depth,observed_first_range_m,tolerance_m=.20):
    """仅观测首回波前为空；未相交射线处罚为0，coverage须独立保持支持。"""
    if len(first_depth)==0: return first_depth.sum()*0
    valid=torch.isfinite(first_depth)
    finite_depth=torch.where(valid,first_depth,observed_first_range_m)
    intrusion=(observed_first_range_m-tolerance_m-finite_depth).clamp_min(0)
    return intrusion.mean()


def first_return_metrics(first_depth,observed_first_range_m,tolerance_m=.20):
    if len(first_depth)==0:
        return {'rays':0,'hit_rate':None,'early_rate':None,'miss_rate':None,'returned_mae_m':None}
    hit=torch.isfinite(first_depth)
    residual=first_depth-observed_first_range_m
    return {'rays':len(first_depth),'returned':hit.sum().item(),
        'hit_rate':(hit&(residual.abs()<=tolerance_m)).float().mean().item(),
        'early_rate':(hit&(residual<-tolerance_m)).float().mean().item(),
        'late_rate':(hit&(residual>tolerance_m)).float().mean().item(),
        'miss_rate':(~hit).float().mean().item(),
        'returned_mae_m':residual[hit].abs().mean().item() if hit.any() else None,
        'mean_intrusion_m':direct_free_space_loss(first_depth,observed_first_range_m,tolerance_m).item()}

"""射线条件的最近曲面吸引；未命中时仍移动已有表面，不保证首交点或支持出生。"""
import torch
from .surface_readout import first_triangle_intersection


def ray_conditioned_surface_points(vertices,faces,targets,directions,lateral_ratio=20/3,
                                   ray_chunk=64,face_chunk=512):
    """在沿束/横向各向异性度量下找全三角面最近点，再按最优重心权重反传。

    离散选择与最优权重停止梯度；包络定理适用于非切换处的最优距离，
    不适用于任意依赖该最近点的其他目标。方向及测量坐标为只读监督。
    """
    if lateral_ratio<=0: raise ValueError('横向比例必须为正')
    if not len(faces): raise ValueError('射线最近点需要非空曲面')
    targets=targets.detach(); directions=directions.detach()
    outputs=[]
    for offset in range(0,len(targets),ray_chunk):
        target=targets[offset:offset+ray_chunk]
        direction=directions[offset:offset+ray_chunk,None,None,:]
        with torch.no_grad():
            best=target.new_full((len(target),),float('inf'))
            best_ids=torch.zeros(len(target),device=target.device,dtype=torch.long)
            best_bary=target.new_zeros(len(target),3)
            row=torch.arange(len(target),device=target.device)
            for start in range(0,len(faces),face_chunk):
                relative=vertices.detach()[faces[start:start+face_chunk]][None]-target[:,None,None]
                parallel=(relative*direction).sum(-1,keepdim=True)*direction
                mapped=lateral_ratio*(relative-parallel)+parallel
                a,b,c=mapped.unbind(2); ab,ac=b-a,c-a
                normal=torch.cross(ab,ac,dim=-1)
                denom=normal.square().sum(-1)
                v=(torch.cross(-a,ac,dim=-1)*normal).sum(-1)/denom.clamp_min(1e-15)
                w=(torch.cross(ab,-a,dim=-1)*normal).sum(-1)/denom.clamp_min(1e-15)
                bary=torch.stack([1-v-w,v,w],-1)
                distance=(mapped*bary[...,None]).sum(2).square().sum(-1)
                distance=distance.masked_fill(~((bary>=0).all(-1)&(denom>1e-15)),float('inf'))
                # 边和退化点也是可行支持；不强制投影落在面内部或新增占据面积。
                for left,right in [(0,1),(1,2),(2,0)]:
                    edge=mapped[:,:,right]-mapped[:,:,left]
                    t=(-(mapped[:,:,left]*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-15)).clamp(0,1)
                    edge_bary=torch.zeros_like(bary)
                    edge_bary[...,left]=1-t; edge_bary[...,right]=t
                    edge_distance=(mapped[:,:,left]+t[...,None]*edge).square().sum(-1)
                    change=edge_distance<distance
                    distance=torch.minimum(distance,edge_distance)
                    bary=torch.where(change[...,None],edge_bary,bary)
                local,which=distance.min(-1)
                change=local<best
                best=torch.minimum(best,local)
                best_ids=torch.where(change,which+start,best_ids)
                best_bary=torch.where(change[:,None],bary[row,which],best_bary)
        outputs.append((vertices[faces[best_ids]]*best_bary[:,:,None]).sum(1))
    return torch.cat(outputs) if outputs else targets.clone()


def constructive_ray_support_loss(vertices,faces,origins,directions,ranges,lateral_ratio=20/3,
                                  kind='closest'):
    """仅对归属已知的实际首返回调用；不把缺测/UNKNOWN当free或occupied。

    返回米制各向异性距离。比例是优化取舍，不是已校准的传感器噪声。
    它是方向加权coverage候选，既不保证遮挡顺序，也不创建空网格拓扑。
    """
    if kind not in ('closest','first_surface'):
        raise ValueError('未知射线支持目标')
    if not len(ranges) or not len(faces):
        return vertices.sum()*0,{'count':0,'unavailable':'no_owned_returns' if not len(ranges) else 'no_surface',
                                  'lateral_m':None,'parallel_abs_m':None}
    direction=directions.detach()
    targets=(origins+direction*ranges[:,None]).detach()
    if kind=='first_surface':
        # 首面由几何顺序决定，与目标距离无关；仅miss继续使用最近面吸引。
        depth,_=first_triangle_intersection(vertices,faces,origins.detach(),direction)
        hit=torch.isfinite(depth)
        total=vertices.sum()*0
        if hit.any():
            first_error=(depth[hit]-ranges.detach()[hit]).abs()
            total=total+first_error.sum()
        miss_stats={}
        if (~hit).any():
            fallback,miss_stats=constructive_ray_support_loss(vertices,faces,origins[~hit],
                direction[~hit],ranges[~hit],lateral_ratio,kind='closest')
            total=total+fallback*(~hit).sum()
        return total/len(ranges),{'count':len(ranges),'unavailable':None,'kind':kind,
            'first_surface_count':hit.sum().item(),'miss_attraction_count':(~hit).sum().item(),
            'first_surface_abs_m':first_error.detach().mean().item() if hit.any() else None,
            'miss_attraction_m':fallback.detach().item() if (~hit).any() else None,
            'lateral_ratio':lateral_ratio,'lateral_m':miss_stats.get('lateral_m'),
            'parallel_abs_m':miss_stats.get('parallel_abs_m')}
    nearest=ray_conditioned_surface_points(vertices,faces,targets,direction,lateral_ratio)
    delta=nearest-targets
    parallel=(delta*direction).sum(-1,keepdim=True)*direction
    lateral=delta-parallel
    loss=(lateral_ratio*lateral+parallel).norm(dim=-1).mean()
    return loss,{'count':len(ranges),'unavailable':None,'lateral_ratio':lateral_ratio,
                 'lateral_m':lateral.detach().norm(dim=-1).mean().item(),
                 'parallel_abs_m':parallel.detach().norm(dim=-1).mean().item()}

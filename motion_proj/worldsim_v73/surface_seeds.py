"""build侧表面支持生成查询种子，离散采样保留所选原生深度的位置梯度。"""
import torch


@torch.no_grad()
def farthest_indices(points,count):
    if len(points)==0: raise ValueError('表面种子需要至少一个支持点')
    count_unique=min(count,len(points))
    x=points.detach().float()
    ids=torch.empty(count_unique,device=x.device,dtype=torch.long)
    distance=x.new_full((len(x),),float('inf'))
    current=(x-x.mean(0)).square().sum(-1).argmax()
    for i in range(count_unique):
        ids[i]=current
        distance=torch.minimum(distance,(x-x[current]).square().sum(-1))
        current=distance.argmax()
    if count_unique<count: ids=ids[torch.arange(count,device=x.device)%count_unique]
    return ids


def native_surface_points(depths,metric_scale,camera_from_actor,intrinsics,size_lwh_m,valid_image_rect=None):
    height,width=depths.shape[-2:]
    y,x=torch.meshgrid(torch.arange(height,device=depths.device),torch.arange(width,device=depths.device),indexing='ij')
    pixel=torch.stack([x,y,torch.ones_like(x)],-1).float().reshape(-1,3)
    points=[]
    counts=[]
    for view_index,(depth,pose,calibration) in enumerate(zip(depths,camera_from_actor,intrinsics)):
        camera=(pixel@torch.linalg.inv(calibration).T)*depth.reshape(-1,1)*metric_scale
        actor=(camera-pose[:3,3])@pose[:3,:3]
        # 已知Actor归属框，不按heldout质量选择；计数反映投影/尺度支持不足。
        valid=torch.isfinite(actor).all(-1)&(depth.reshape(-1)>.05)&(actor.abs()<=size_lwh_m/2+.25).all(-1)
        if valid_image_rect is not None:
            rect=torch.as_tensor(valid_image_rect[view_index],device=depths.device)
            valid&=((pixel[:,:2]>=rect[:2])&(pixel[:,:2]<rect[2:])).all(-1)
        points.append(actor[valid]); counts.append(int(valid.sum()))
    candidates=torch.cat(points)
    return candidates,counts


def native_surface_seeds(depths,metric_scale,camera_from_actor,intrinsics,size_lwh_m,build_points,count,valid_image_rect=None):
    candidates,counts=native_surface_points(depths,metric_scale,camera_from_actor,intrinsics,size_lwh_m,valid_image_rect)
    fallback=len(candidates)==0
    if fallback: candidates=build_points
    ids=farthest_indices(candidates,count)
    return candidates[ids],{'per_view_native_support':counts,'native_candidates':sum(counts),
                             'lidar_fallback':fallback,'seed_count':count}

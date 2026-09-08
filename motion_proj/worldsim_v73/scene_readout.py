"""CPU BVH读出只读显式表面，规范Actor与世界背景在真实束上统一排序。"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def pca_patch_surface(points, spacing=.06, count=None):
    """与Actor固定PCA读出同9顶点/8三角片，不补洞或闭合未知背景。"""
    points=np.asarray(points,dtype=np.float32)
    if not len(points):
        return np.empty((0,3),np.float32),np.empty((0,3),np.uint32)
    ids=np.arange(len(points)) if count is None else np.linspace(0,len(points)-1,min(len(points),count)).astype(int)
    centers=points[ids]
    neighbors=cKDTree(points).query(centers,k=min(20,len(points)),workers=2)[1]
    neighbors=np.asarray(neighbors).reshape(len(centers),-1)
    local=points[neighbors]; local=local-local.mean(1,keepdims=True)
    _,vectors=np.linalg.eigh(local.transpose(0,2,1)@local)
    normal=vectors[:,:,0]
    reference=np.zeros_like(normal); reference[:,2]=1
    take=np.abs(normal[:,2])>.9
    reference[take,2]=0; reference[take,1]=1
    tangent=np.cross(normal,reference)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1e-12)
    bitangent=np.cross(normal,tangent)
    grid=np.array([(i,j) for i in range(-1,2) for j in range(-1,2)],np.float32)*spacing
    vertices=(centers[:,None]+tangent[:,None]*grid[None,:,0:1]+bitangent[:,None]*grid[None,:,1:2]).reshape(-1,3)
    base=[]
    for i in range(2):
        for j in range(2):
            a=i*3+j; base.extend([[a,a+1,a+3],[a+1,a+4,a+3]])
    faces=(np.array(base)[None]+np.arange(len(centers))[:,None,None]*9).reshape(-1,3)
    return vertices.astype(np.float32),faces.astype(np.uint32)


class SurfaceBVH:
    def __init__(self,vertices,faces):
        self.scene=None
        if len(faces):
            self.scene=o3d.t.geometry.RaycastingScene(nthreads=2)
            self.scene.add_triangles(o3d.core.Tensor(np.asarray(vertices,np.float32)),
                                     o3d.core.Tensor(np.asarray(faces,np.uint32)))

    def cast(self,origins,directions,world_from_local=None):
        if self.scene is None: return np.full(len(directions),np.inf,np.float32)
        if world_from_local is not None:
            pose=np.asarray(world_from_local)
            origins=(origins-pose[:3,3])@pose[:3,:3]
            directions=directions@pose[:3,:3]
        rays=np.concatenate([np.broadcast_to(origins,directions.shape),directions],axis=-1).astype(np.float32)
        return self.scene.cast_rays(o3d.core.Tensor(rays),nthreads=2)['t_hit'].numpy()

    def cast_per_time(self,origins,directions,timestamps_ns,trajectory,sensor_known=None,chunk=8192):
        """Keep one canonical BVH and transform each beam at its known rigid time."""
        directions=np.asarray(directions); origins=np.broadcast_to(origins,directions.shape)
        times=np.asarray(timestamps_ns,np.int64)
        known=(times>=trajectory.times[0])&(times<=trajectory.times[-1])
        if sensor_known is not None: known&=sensor_known
        depth=np.full(len(times),np.inf,np.float32)
        if self.scene is None: return depth,known
        indices=np.flatnonzero(known)
        for start in range(0,len(indices),chunk):
            take=indices[start:start+chunk]
            poses,_=trajectory.at(times[take]); rotations=poses[:,:3,:3]
            local_origins=np.einsum('nji,nj->ni',rotations,origins[take]-poses[:,:3,3])
            local_directions=np.einsum('nji,nj->ni',rotations,directions[take])
            # Rigid rotation preserves direction norm and the metric ray distance.
            depth[take]=self.cast(local_origins,local_directions)
        return depth,known


def compose_first(background_depth,actor_depths):
    """Actor depths为(owner_id,原始束距离)；未命中=-1，背景=0。"""
    best=background_depth.copy()
    owner=np.where(np.isfinite(best),0,-1).astype(np.int32)
    actor_first=np.full_like(best,np.inf)
    for actor_id,depth in actor_depths:
        closer=depth<best
        owner[closer]=actor_id; best=np.minimum(best,depth)
        actor_first=np.minimum(actor_first,depth)
    return best,owner,actor_first


def ray_statistics(depth,observed,selected):
    depth=depth[selected]; observed=observed[selected]
    returned=np.isfinite(depth); n=len(depth)
    error=np.abs(depth[returned]-observed[returned])
    early=returned&(depth<observed-.2)
    hit=returned&(np.abs(depth-observed)<=.2)
    late=returned&(depth>observed+.2)
    intrusion=np.where(returned,np.maximum(observed-.2-depth,0),0)
    return {'rays':n,'returned':int(returned.sum()),'hit':int(hit.sum()),'early':int(early.sum()),
        'late':int(late.sum()),'miss':int((~returned).sum()),
        'hit_rate':float(hit.mean()) if n else None,'early_rate':float(early.mean()) if n else None,
        'miss_rate':float((~returned).mean()) if n else None,
        'free_intrusion_m':float(intrusion.mean()) if n else None,
        'free_intrusion_sum_m':float(intrusion.sum()),
        'returned_abs_error_sum_m':float(error.sum()),
        'returned_mae_m':float(error.mean()) if len(error) else None}

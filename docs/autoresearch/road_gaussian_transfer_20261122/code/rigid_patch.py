"""鲁棒目标路面框架上的保距高斯块配准，不拉伸微三角几何噪声。"""
import numpy as np
from scipy.spatial import cKDTree
from gaussian_patch_transport import quaternion_wxyz_to_matrix,matrix_to_quaternion_wxyz,covariance_from_sigma_quaternion

class RoadFrames:
    def __init__(self,world,normals,valid,forward):
        self.world=np.asarray(world)[valid];n=np.asarray(normals)[valid]
        n=n/np.linalg.norm(n,axis=1,keepdims=True);n*=np.where(n[:,2:3]<0,-1,1)
        self.normal=n;self.tree=cKDTree(self.world[:,:2]);self.forward=np.asarray(forward,float)
    def at(self,xy,radius):
        ids=np.asarray(self.tree.query_ball_point(xy,radius),int)
        if len(ids)<16:return None
        points=self.world[ids];n=np.median(self.normal[ids],axis=0);n/=np.linalg.norm(n)
        if n[2]<.9:return None
        offsets=points@n;center=np.median(offsets);mad=max(1.4826*np.median(abs(offsets-center)),.005)
        inlier=abs(offsets-center)<=min(3*mad,.05)
        if inlier.sum()<16:return None
        d=np.median(offsets[inlier]);origin=np.array([xy[0],xy[1],(d-n[0]*xy[0]-n[1]*xy[1])/n[2]])
        t=self.forward-n*np.dot(self.forward,n);t/=np.linalg.norm(t)
        frame=np.stack([t,np.cross(n,t),n],axis=1)
        return dict(origin=origin,frame=frame,normal=n,inliers=int(inlier.sum()),samples=len(ids),
            median_absolute_height_residual=float(np.median(abs(offsets-center))),
            inlier_height_residual_p95=float(np.quantile(abs(offsets[inlier]-d),.95)))

def transport_rigid(centers,sigma,quaternion,rgb,opacity,source_ids,*,source_origin,source_frame,target,patch_id,source_asset_id=None):
    centers=np.asarray(centers,np.float64);source_frame=np.asarray(source_frame,np.float64)
    R=np.asarray(target['frame'])@source_frame.T;origin=np.asarray(source_origin)
    rotation=R[None]@quaternion_wxyz_to_matrix(quaternion)
    world=(centers-origin)@R.T+target['origin'];normal=rotation[:,:,2]
    cov0=covariance_from_sigma_quaternion(sigma,quaternion);cov=R[None]@cov0@R.T
    q=matrix_to_quaternion_wxyz(rotation);reconstructed=covariance_from_sigma_quaternion(sigma,q)
    n=len(centers);local=(centers-origin)@source_frame
    assert np.allclose(R.T@R,np.eye(3),atol=1e-10) and np.linalg.det(R)>0
    return dict(world_xyz=world,normal_world=normal,covariance_world=cov,gaussian_sigma_m=np.array(sigma,copy=True),
        gaussian_rotation_world_wxyz=q,gaussian_rotation_world_override=np.ones(n,bool),
        reference_srgb=np.array(rgb,copy=True),road_opacity=np.array(opacity,copy=True),source_row_id=np.array(source_ids,copy=True),
        patch_id=np.full(n,patch_id),source_uv=local[:,:2],target_uv=world[:,:2],source_normal_residual_m=local[:,2],
        jacobian_world=np.broadcast_to(R,(n,3,3)).copy(),valid=np.ones(n,bool),source_input_index=np.arange(n),
        trace=dict(source_origin=origin,source_frame=source_frame,mapping_2d=np.eye(2),translation_uv=target['origin'][:2],
            scale=1.,rotation_rad=0.,normal_residual_carried=True,normal_extension_scale=1.,input_valid=np.ones(n,bool),
            dropped_input_indices=np.empty(0,int),source_row_ids_all=np.array(source_ids,copy=True),source_covariance_world=cov0,
            source_rotation_world_wxyz=np.array(quaternion,copy=True),source_sigma_m=np.array(sigma,copy=True),
            covariance_reconstruction_max_abs=abs(reconstructed-cov).max(axis=(1,2)),source_kind='similar_gaussian_patch_rigid',
            source_asset_id=source_asset_id,rotation_convention='world_wxyz_columns_tangent0_tangent1_normal',world_units='meter',
            target_origin=target['origin'],target_frame=target['frame'],rigid_rotation=R,target_plane=target))

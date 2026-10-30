"""Reuse legal baseline calibration gauge and native spatial LR scale.

No GT geometry, owner labels, RGB or edited references are read here.
"""
import json
from pathlib import Path
import numpy as np


def transform_cameras(poses, transform):
    poses=np.asarray(poses); transform=np.asarray(transform,dtype=np.float64)
    scale=float(np.linalg.norm(transform[:3,0]))
    if not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3)*scale**2,atol=1e-10):
        raise ValueError('Shared gauge must be a similarity')
    result=poses.copy()
    result[:,:3,:3]=np.einsum('ij,njk->nik',transform[:3,:3]/scale,poses[:,:3,:3])
    result[:,:3,3]=poses[:,:3,3]@transform[:3,:3].T+transform[:3,3]
    return result,transform.copy(),scale


def install(gauge_file,source,receipt=None):
    import sys
    sys.path.insert(0,str(source))
    from scene import waymo_loader, waymo_loader2
    gauge=json.loads(Path(gauge_file).read_text())
    transform=np.asarray(gauge['world_to_model'],dtype=np.float64)
    spatial=float(gauge['native_spatial_lr_scale'])
    extent=float(gauge['native_camera_extent'])
    if spatial!=1.: raise ValueError('This native Scene hard-codes model initialization scale1')
    original_norm=waymo_loader2.getNerfppNorm
    def shared_pca(poses,fix_radius=0):
        if fix_radius!=0: raise ValueError('Preserve baseline fix_radius=0 config; shared gauge is explicit')
        return transform_cameras(poses,transform)
    def shared_norm(cameras):
        result=original_norm(cameras)
        # Original loader inverts this radius after getNerfppNorm.
        result['radius']=1/extent
        return result
    for loader in (waymo_loader,waymo_loader2):
        loader.transform_poses_pca=shared_pca
        loader.getNerfppNorm=shared_norm
    import scene as scene_module
    original_scene=scene_module.Scene
    def checked_scene(*args,**kwargs):
        result=original_scene(*args,**kwargs)
        if abs(float(result.cameras_extent)-extent)>1e-10:
            raise RuntimeError('Actual Scene did not preserve baseline camera extent')
        for attr in ('gaussians','dynamic_gaussians'):
            model=getattr(result,attr,None)
            if model is not None and getattr(model,'optimizer',None) is not None and float(model.spatial_lr_scale)!=spatial:
                raise RuntimeError('Actual native spatial LR scale changed')
        return result
    scene_module.Scene=checked_scene
    if receipt:
        Path(receipt).write_text(json.dumps(dict(shared_gauge=str(gauge_file),world_to_model=transform.tolist(),native_spatial_lr_scale=spatial,native_camera_extent=extent,baseline_only_no_GT=True),indent=2))
    return gauge

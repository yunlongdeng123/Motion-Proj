"""CPU-only persistent parent-tangent resampling; no new geometry evidence."""
from pathlib import Path
import hashlib
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    return h.hexdigest()


def array_hash(value):
    a=np.ascontiguousarray(value);h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()


def read_surface(path):
    names=('active','local_xyz','world_xyz','model_xyz','normal_local','normal_world','gaussian_sigma_m',
           'reference_srgb','opacity','opacity_logit','cell_m','coordinate_anchor','world_to_model',
           'meter_to_model','source_parent_cell_ids')
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in names}


def resample_surface(parent,factor):
    if factor not in (1,2,4):raise ValueError('Only explicit fixed factors1/2/4 are implemented')
    count=len(parent['active']);assert parent['active'].all()
    if factor==1:
        result={k:parent[k].copy() for k in ('local_xyz','world_xyz','model_xyz','normal_local','normal_world',
            'gaussian_sigma_m','reference_srgb','opacity','opacity_logit')}
        result.update(parent_indices=np.arange(count),child_indices=np.zeros(count,np.int16),
            source_parent_cell_ids=parent['source_parent_cell_ids'].copy(),delta_local_m=np.zeros((count,3)))
        return result
    axis=(np.arange(factor,dtype=np.float64)+.5)/factor-.5
    xx,yy=np.meshgrid(axis,axis,indexing='ij');offset=np.c_[xx.ravel(),yy.ravel()]*float(parent['cell_m'])
    nchild=factor*factor;ids=np.repeat(np.arange(count),nchild)
    normals=parent['normal_local'][ids].astype(np.float64)
    if np.any(np.abs(normals[:,2])<1e-6):raise ValueError('Tangent graph cannot represent near-vertical parent')
    xy=np.tile(offset,(count,1));dz=-(normals[:,:2]*xy).sum(1)/normals[:,2]
    delta=np.c_[xy,dz]
    world_delta=delta@parent['coordinate_anchor'][:3,:3].T
    model_delta=world_delta@parent['world_to_model'][:3,:3].T
    result=dict(local_xyz=parent['local_xyz'][ids].astype(np.float64)+delta,
        world_xyz=parent['world_xyz'][ids]+world_delta,model_xyz=parent['model_xyz'][ids]+model_delta,
        normal_local=parent['normal_local'][ids].copy(),normal_world=parent['normal_world'][ids].copy(),
        gaussian_sigma_m=(parent['gaussian_sigma_m'][ids]/factor).astype(np.float32),
        reference_srgb=parent['reference_srgb'][ids].copy(),opacity=parent['opacity'][ids].copy(),
        opacity_logit=parent['opacity_logit'][ids].copy(),parent_indices=ids,
        child_indices=np.tile(np.arange(nchild,dtype=np.int16),count),
        source_parent_cell_ids=parent['source_parent_cell_ids'][ids].copy(),delta_local_m=delta)
    assert np.max(np.abs((delta*normals).sum(1)))<1e-10
    assert np.max(np.abs(delta.reshape(count,nchild,3).mean(1)))<1e-12
    return result


def summarize(parent,child,factor):
    parea=np.prod(parent['gaussian_sigma_m'][:,:2].astype(float),1).sum()
    carea=np.prod(child['gaussian_sigma_m'][:,:2].astype(float),1).sum()
    parent_ids=child['parent_indices'];normal=parent['normal_local'][parent_ids].astype(float)
    return dict(factor=factor,parent_rows=len(parent['active']),child_rows=len(parent_ids),
        parent_cell_m=float(parent['cell_m']),child_grid_step_m=float(parent['cell_m'])/factor,
        sigma_policy='all three parent kernel sigmas divided by factor',
        nominal_peak_opacity_policy='copy parent .99; no opacity/n^2 or optical-depth split',
        opacity_logit_copy_exact=np.array_equal(child['opacity_logit'],parent['opacity_logit'][parent_ids]),
        material_copy_exact=np.array_equal(child['reference_srgb'],parent['reference_srgb'][parent_ids]),
        source_identity_copy_exact=np.array_equal(child['source_parent_cell_ids'],parent['source_parent_cell_ids'][parent_ids]),
        tangent_plane_residual_m=float(np.abs((child['delta_local_m']*normal).sum(1)).max()),
        intrinsic_plane_kernel_area_ratio=float(carea/parea),
        projected_alpha_preservation_claim=False,
        warning='Native cov2D filter floor +0.3I may alter projected alpha/density; no AA compensation implemented',
        first_hit_surface_claim=False,geometry_role='same parent tangent surface; not extra LiDAR or height fitting')

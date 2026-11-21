"""仅从既有删除/去影掩码与固定道路求迁移域，不读取扩散RGB。"""
from pathlib import Path
import json
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from fixed_surface import FixedRoadSurface

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=C/'road_gaussian_transfer_20261122_r1'
PARENT=C/'road_surface_residual_20261121_r2/appearance_fit/asset'
OLD=C/'road_multisource_20261122_r1';COMMON=C/'road_common1152_20261121'

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
    out=ROOT/'target';out.mkdir(parents=True,exist_ok=True)
    r=load(PARENT/'road.npz');p=load(PARENT/'provenance.npz')
    base=load(C/'road_asset_20261116_r1/phase2/formal_r1/asset/base_atlas.npz')
    certificate=load(C/'road_surface_adaptive_20261121_r1/proposal/surface_certificate.npz')
    grid=base['grid_world_row'];n=grid.size
    world=base['surface_world_xyz'].copy().reshape(-1,3)
    row_world=np.zeros((n,3));row_world[grid.ravel()]=world
    row_world[certificate['recovered_world_rows']]=certificate['target_world_xyz']
    support=load(COMMON/'formal/r3/G_NUMERIC/asset/provenance.npz')['surface_supported'][:n].copy()
    support[certificate['recovered_world_rows']]=True
    X=p['world_xyz'];requested=np.zeros(len(X),bool);real_world=[];records=[]
    for raw in (86,110):
        camera=load(COMMON/f'maps/{raw:03d}.npz');d=load(C/'road_surface_residual_20261121_r2/proposal/domains'/f'{raw:03d}.npz')
        q=load(OLD/f'sources/targets/{raw:03d}.npz')
        cam=X@camera['world_to_camera'][:3,:3].T+camera['world_to_camera'][:3,3]
        uvh=cam@camera['camera_K'].T;uv=uvh[:,:2]/np.maximum(cam[:,2:],1e-9)-.5
        def sample(mask):return ndi.map_coordinates(mask.astype(np.uint8),[uv[:,1],uv[:,0]],order=0,mode='constant',cval=0)>0
        legal=d['appearance_C']|d['appearance_S'];valid=legal&q['geometry_valid']
        hit=sample(valid)&(cam[:,2]>0)
        requested|=hit
        same=legal&q['same_mask'];real_world.append(q['world_xyz'][same])
        records.append(dict(raw=raw,mask_pixels=int(legal.sum()),certified_pixels=int(valid.sum()),same_pixels=int(same.sum()),requested_parent_centers=int(hit.sum())))
    # 几何源/目标共用同一原三角支持，不作新平面拟合。
    target=requested[:n][grid]&support[grid]
    # 只补网格内一个cell大小的离散裂隙，仍限制在原支持和原请求包围内。
    closed=ndi.binary_closing(target,structure=np.ones((3,3)))
    target|=closed&ndi.binary_dilation(target,iterations=1)&support[grid]
    real=np.concatenate(real_world)
    if len(real):
        near,_=cKDTree(real[:,:2]).query(row_world[grid.ravel(),:2]);target&=near.reshape(grid.shape)>.01
    surface=FixedRoadSurface(row_world[grid],support[grid],base['world_from_uv'],target)
    surface.save(out/'surface.npz')
    inside=surface.domain(X[:,:2]);eligible_geometry=surface.sample(X[:,:2])['valid']
    protected=np.zeros(len(X),bool)
    oldqual=load(OLD/'qualification/allowed_rows.npz');active=np.flatnonzero(r['active'])
    assert len(active)==len(oldqual['K_contributing'])
    protected[active]=oldqual['K_contributing']
    replacement=inside&eligible_geometry&r['active']&~protected
    np.savez_compressed(out/'rows.npz',requested=requested,target_inside=inside,geometric_valid=eligible_geometry,
        protected_K=protected,replace_allowed=replacement,source_same_position_world=real)
    yy,xx=np.nonzero(target);xy=row_world[grid[target],:2]
    receipt=dict(complete=True,parent=str(PARENT),target_vertices=int(target.sum()),target_grid_area_m2=float(target.sum()*abs(np.linalg.det(base['world_from_uv'][:2,:2]))),
        target_xy_min=xy.min(0).tolist(),target_xy_max=xy.max(0).tolist(),grid_bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
        replace_allowed_old_rows=int(replacement.sum()),protected_old_rows_in_target=int((inside&r['active']&protected).sum()),
        keep_same_position_count=len(real),records=records,material_diffusion_RGB_read=False,new_diffusion_calls=0,
        geometry='original fixed canonical triangles plus same348 certified nodes',target_domain='union of original C/S projected onto certified fixed world grid; known same-position witness radius protected')
    write(out/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

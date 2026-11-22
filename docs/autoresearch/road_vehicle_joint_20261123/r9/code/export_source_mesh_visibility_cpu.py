"""Export image-domain depth reconstructed from actual grid triangle XYZ, CPU only."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_fixed_surface_mesh_xyz_cpu import pin,stat


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign-root',type=Path,default=Path('/root/autodl-tmp/drivingeditor_repro/campaign'))
    ap.add_argument('--frames',nargs='+',type=int,default=[86,110]);a=ap.parse_args();C=a.campaign_root
    r7=C/'road_shared_appearance_20261122_r7';r9=C/'old_raw_teacher_joint_20261123_r9'
    gp=r7/'shared_grid_geometry.npz'
    with np.load(gp,allow_pickle=False) as g:
        grid=g['grid_world_row'];world=g['world_xyz'];gv=g['geometry_valid']
        base=np.empty((grid.size,3),np.float64);base[grid.ravel()]=world.reshape(-1,3)
        base_valid=np.empty(grid.size,bool);base_valid[grid.ravel()]=gv.ravel()
    rows=[]
    for raw in a.frames:
        mp=C/'road_multisource_20261122_r1/sources/maps'/f'{raw:03d}.npz';cp=r7/'projection_cache_v1'/f'{raw:03d}'/'view_cache.npz'
        with np.load(mp,allow_pickle=False) as m:
            source_valid=m['geometry_valid'].astype(bool);ids=m['triangle_world_rows'][source_valid];bary=m['triangle_weights'][source_valid]
            reference_xyz=m['world_xyz'][source_valid]
        with np.load(cp,allow_pickle=False) as c:
            E=c['world_to_camera'];K=c['camera_K'];cached_valid=c['projection_valid'].astype(bool);expected_sha=str(c['source_map_sha256'])
        assert pin(mp)['sha256']==expected_sha
        assert ids.min()>=0 and ids.max()<len(base)
        tri=np.einsum('ni,nij->nj',bary,base[ids]);cam=tri@E[:3,:3].T+E[:3,3]
        project=cam@K.T;uv=project[:,:2]/project[:,2:3]-.5
        yy,xx=np.nonzero(source_valid);error=np.linalg.norm(uv-np.stack([xx,yy],-1),axis=-1)
        xyz_error=np.abs(tri-reference_xyz)
        assert xyz_error.max()<1e-8 and error.max()<1e-6
        depth=np.zeros(source_valid.shape,np.float64);depth[source_valid]=cam[:,2]
        corners=np.zeros(source_valid.shape,bool);corners[source_valid]=base_valid[ids].all(1)
        corner_count=np.zeros(source_valid.shape,np.uint8);corner_count[source_valid]=base_valid[ids].sum(1)
        projection_valid=source_valid&cached_valid&np.isfinite(depth)&(depth>0)
        out=r9/'audit/surface_visibility'/f'{raw:03d}';out.mkdir(parents=True,exist_ok=True)
        npz=out/'surface.npz';assert not npz.exists(),npz
        np.savez_compressed(npz,camera_depth_m=depth,triangle_all_corners_geometry_valid=corners,
            projection_valid=projection_valid,source_geometry_valid=source_valid,
            cached_projection_valid=cached_valid,triangle_geometry_valid_corner_count=corner_count,
            camera_K=K,world_to_camera=E,raw=np.asarray(raw),
            geometry_sha256=pin(gp)['sha256'],source_map_sha256=pin(mp)['sha256'],view_cache_sha256=pin(cp)['sha256'])
        row=dict(schema='R9.independent_mesh_visibility_surface.v1',complete=True,raw=raw,
            geometry=pin(gp),source_map=pin(mp),view_cache=pin(cp),output=pin(npz),
            camera_depth_formula='z(sum_i bary_i * actual_R7_base_worldXYZ[triangle_world_rows_i] @ E_camera_R.T + E_camera_t)',
            fixed_grid_to_world_rows='base_worldXYZ[grid_world_row.ravel()]=g.world_xyz.reshape(-1,3)',
            triangle_validity='all three base rows are true in actual g.geometry_valid mapped by grid_world_row',
            projection_validity='source.geometry_valid & cache.projection_valid & finite positive independent camera depth',
            inactive_depth_sentinel=0,depth_sentinel_is_not_material_or_visibility=True,
            source_XYZ_used_for_depth=False,source_XYZ_used_only_for_independent_error_comparison=True,
            XYZ_max_abs_error_m=xyz_error.max(0).tolist(),Z_max_abs_error_m=float(xyz_error[:,2].max()),
            source_pixel_reprojection_error_px=stat(error),
            counts={'source_geometry_valid':int(source_valid.sum()),'cache_projection_valid':int(cached_valid.sum()),
              'projection_valid':int(projection_valid.sum()),'triangle_all_corners_valid':int((projection_valid&corners).sum()),
              'triangle_has_invalid_corner':int((projection_valid&~corners).sum()),
              'corner_valid_count':{str(i):int((source_valid&(corner_count==i)).sum()) for i in range(4)}},
            depth_m=stat(depth[projection_valid]),
            no_old_CS_W_K_used=True,CPU_only=True,GPU_forward=0,diffusion_calls=0,parameter_updates=0)
        (out/'receipt.json').write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8');rows.append(row)
        print(json.dumps({'raw':raw,'output':row['output'],'counts':row['counts'],'Z_max_abs_error_m':row['Z_max_abs_error_m'],'reproject_max_px':error.max()}),flush=True)
    target=r9/'audit/surface_visibility/receipt.json'
    target.write_text(json.dumps({'schema':'R9.independent_mesh_visibility_export.v1','complete':True,'frames':a.frames,'rows':rows,
      'CPU_only':True,'GPU_forward':0,'diffusion_calls':0,'parameter_updates':0},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'receipt':pin(target)}))


if __name__=='__main__':main()

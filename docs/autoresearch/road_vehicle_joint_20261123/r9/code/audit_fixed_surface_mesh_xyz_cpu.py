"""Independent full XYZ reconstruction of source-map triangles from fixed R7 grid."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def pin(p):
    p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def stat(a):
    a=np.asarray(a);return {'count':int(a.size),'mean':float(np.mean(a)),
      'p50':float(np.percentile(a,50)),'p95':float(np.percentile(a,95)),
      'p99':float(np.percentile(a,99)),'max':float(np.max(a))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign-root',type=Path,default=Path('/root/autodl-tmp/drivingeditor_repro/campaign'))
    ap.add_argument('--frames',nargs='+',type=int,default=[86,110,134,179,198]);a=ap.parse_args();C=a.campaign_root
    start=time.time();r7=C/'road_shared_appearance_20261122_r7';r9=C/'old_raw_teacher_joint_20261123_r9'
    gp=r7/'shared_grid_geometry.npz'
    with np.load(gp,allow_pickle=False) as g:
        grid=g['grid_world_row'];world=g['world_xyz'];grid_valid=g['geometry_valid'];row_stored=g['row_world_xyz']
        base=np.empty((grid.size,3),np.float64);base[grid.ravel()]=world.reshape(-1,3)
        row_valid=np.empty(grid.size,bool);row_valid[grid.ravel()]=grid_valid.ravel()
    assert np.array_equal(np.sort(grid.ravel()),np.arange(grid.size))
    grid_row_error=np.abs(base[row_valid]-row_stored[:grid.size][row_valid])
    parent_path=C/'road_surface_residual_20261121_r2/appearance_fit/asset/provenance.npz'
    with np.load(parent_path,allow_pickle=False) as p:
        parent_world=p['world_xyz'];surface=p['surface_supported'][:grid.size]
    compare=row_valid&surface
    target_error=np.abs(base[compare]-parent_world[:grid.size][compare])
    rows=[]
    for raw in a.frames:
        mp=C/'road_multisource_20261122_r1/sources/maps'/f'{raw:03d}.npz'
        cp=r7/'projection_cache_v1'/f'{raw:03d}'/'view_cache.npz'
        with np.load(mp,allow_pickle=False) as m:
            valid=m['geometry_valid'].astype(bool);ids=m['triangle_world_rows'][valid];bary=m['triangle_weights'][valid]
            source_xyz=m['world_xyz'][valid];map_k=m['camera_K'];map_E=m['world_to_camera']
        assert ids.min()>=0 and ids.max()<len(base),(raw,ids.min(),ids.max(),len(base))
        tri=np.einsum('ni,nij->nj',bary,base[ids])
        with np.load(cp,allow_pickle=False) as cache:
            K=cache['camera_K'];E=cache['world_to_camera'];cached_map_sha=str(cache['source_map_sha256'])
        assert np.array_equal(K,map_k) and np.array_equal(E,map_E),(raw,'camera mismatch')
        assert pin(mp)['sha256']==cached_map_sha,(raw,'cache source-map SHA mismatch')
        err=np.abs(tri-source_xyz);cam=tri@E[:3,:3].T+E[:3,3]
        pix=cam@K.T;uv=pix[:,:2]/pix[:,2:3]-.5
        yy,xx=np.nonzero(valid);pixel_error=np.linalg.norm(uv-np.stack([xx,yy],-1),axis=-1)
        assert np.isfinite(tri).all() and np.isfinite(pixel_error).all()
        row=dict(raw=raw,source_map=pin(mp),projection_cache=pin(cp),source_valid_pixel_count=int(valid.sum()),
            triangle_index_min=int(ids.min()),triangle_index_max=int(ids.max()),base_world_row_count=len(base),
            triangle_corners_surface_valid=bool(row_valid[ids].all()),
            barycentric_sum_max_abs_error=float(np.max(abs(bary.sum(1)-1))),
            barycentric_min=float(bary.min()),
            independent_triXYZ_vs_stored_sourceXYZ_max_abs_m=err.max(0).tolist(),
            independent_triXYZ_vs_stored_sourceXYZ_L2_m=stat(np.linalg.norm(err,axis=-1)),
            independent_tri_Z_error_m=stat(err[:,2]),
            independent_triXYZ_to_original_source_pixel_error_px=stat(pixel_error),
            camera_Z_positive=bool((cam[:,2]>0).all()),
            all_XYZ_and_pixel_checks_pass=bool(err.max()<1e-8 and pixel_error.max()<1e-6))
        rows.append(row);print(json.dumps(row),flush=True)
    result=dict(schema='R9.fixed_surface_mesh_full_XYZ_audit.v1',complete=True,
        decision='PASS' if all(x['all_XYZ_and_pixel_checks_pass'] for x in rows) and target_error.max()<1e-8 else 'FAIL',
        method='triXYZ=sum(barycentric*R7 actual grid full XYZ indexed by grid_world_row), then project triXYZ into original source pixels',
        independent_reconstruction_did_not_use_stored_sourceXYZ_as_projection_input=True,
        world_Z_taken_from_actual_grid_not_constant_plane=True,
        geometry=pin(gp),fixed_parent_provenance=pin(parent_path),
        valid_grid_vs_grid_row_stored_max_abs_XYZ_m=grid_row_error.max(0).tolist(),
        valid_grid_vs_fixed_parent_world_XYZ_max_abs_m=target_error.max(0).tolist(),
        matched_valid_base_rows=int(compare.sum()),rows=rows,
        implication='A passing triXYZ reconstruction proves cached mesh/fixed-grid XYZ compatibility, not visibility of every grid center; inverse same-UV/depth gates remain required.',
        self_occlusion_or_footprint_cause_separately_unproven=True,
        CPU_only=True,GPU_forward=0,backward=0,diffusion_calls=0,parameter_updates=0,elapsed_seconds=time.time()-start)
    out=r9/'audit/fixed_surface_mesh_full_XYZ_audit.json';out.parent.mkdir(parents=True,exist_ok=True)
    assert not out.exists(),str(out)
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'decision':result['decision'],'receipt':pin(out)}))


if __name__=='__main__':main()

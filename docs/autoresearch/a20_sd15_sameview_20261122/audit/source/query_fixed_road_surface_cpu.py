"""Query the existing supported road triangulation at wheel XY; no GS proxy."""
from pathlib import Path
import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import numpy as np
from scipy.spatial import cKDTree
import prepare_observation_surface as obs

CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
BASE=CAMPAIGN/'road_asset_20261116_r1/phase2/formal_r1/asset/base_atlas.npz'
CERT=CAMPAIGN/'road_surface_adaptive_20261121_r1/proposal/surface_certificate.npz'
CERT_JSON=CERT.with_suffix('.json')
OLD=CAMPAIGN/'road_common1152_20261121/formal/r3/G_NUMERIC/asset/provenance.npz'
ANCHORS=CAMPAIGN/'road_asset_20261116_r1/asset_r8_guarded/measurement_anchors.npz'
CURRENT=CAMPAIGN/'road_surface_residual_20261121_r2/appearance_fit/asset'


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    base=load(BASE);cert=load(CERT);old=load(OLD);anchors=load(ANCHORS);current=load(CURRENT/'provenance.npz')
    grid=base['grid_world_row'];h,w=grid.shape;count=grid.size
    xyz=np.zeros((count,3));xyz[grid.ravel()]=base['surface_world_xyz'].reshape(-1,3)
    xyz[cert['recovered_world_rows']]=cert['target_world_xyz']
    supported=old['surface_supported'].copy();supported[cert['recovered_world_rows']]=True
    vertex_tree=cKDTree(xyz[:,:2]);where=np.empty((count,2),int)
    yy,xx=np.indices(grid.shape);where[grid.ravel()]=np.column_stack([yy.ravel(),xx.ravel()])
    anchor=anchors['world_xyz'];anchor_tree=cKDTree(anchor[:,:2]);allowed=~np.isin(anchors['source_raw'],[134,179,198])
    allowed_ids=np.flatnonzero(allowed);allowed_tree=cKDTree(anchor[allowed,:2])
    plane=json.loads(CERT_JSON.read_text());n=np.asarray(plane['plane_normal_world']);d=plane['plane_offset_world']
    current_xyz=current['world_xyz'];results=[];geometry_evidence=[]
    for label,receipt_path in [('original_registered',args.original),('physical_pose_once',args.adapted)]:
        receipt=json.loads(receipt_path.read_text())
        for contact in receipt['wheel_results']:
            p=np.asarray(contact['exact_bottom_world_xyz_m'],float)
            _,near=vertex_tree.query(p[:2],k=16)
            cells=set()
            for row,col in where[near]:
                for a in [row-1,row]:
                    for b in [col-1,col]:
                        if 0<=a<h-1 and 0<=b<w-1:cells.add((int(a),int(b)))
            hits=[]
            for row,col in cells:
                a=grid[row,col];b=grid[row,col+1];c=grid[row+1,col];e=grid[row+1,col+1]
                for ids in [np.array([a,b,e]),np.array([a,e,c])]:
                    tri=xyz[ids];matrix=np.column_stack([tri[1,:2]-tri[0,:2],tri[2,:2]-tri[0,:2]])
                    if abs(np.linalg.det(matrix))<1e-12:continue
                    uv=np.linalg.solve(matrix,p[:2]-tri[0,:2]);bary=np.array([1-uv.sum(),uv[0],uv[1]])
                    if bary.min()>=-1e-7:
                        point=bary@tri;normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
                        if normal[2]<0:normal=-normal
                        hits.append(dict(ids=ids,bary=bary,world=point,normal=normal,supported=bool(supported[ids].all())))
            hits.sort(key=lambda q:(not q['supported'],-q['bary'].min()))
            nearest_dist,nearest_id=anchor_tree.query(p[:2]);safe_dist,safe_index=allowed_tree.query(p[:2])
            aid=int(allowed_ids[safe_index])
            q=dict(mesh=label,raw=contact['raw'],wheel=contact['wheel'],world_contact_xyz_m=p.tolist(),
                supported_surface_triangle_found=bool(hits and hits[0]['supported']),surface_triangle_found=bool(hits),
                Gaussian_center_query_was_not_surface_query=True,
                nearest_original_anchor_XY_m=float(nearest_dist),nearest_original_anchor_source_raw=int(anchors['source_raw'][nearest_id]),
                nearest_original_anchor_world_xyz_m=anchor[nearest_id].tolist(),
                nearest_anchor_excluding_134_179_198_XY_m=float(safe_dist),nearest_anchor_excluding_holdout_source_raw=int(anchors['source_raw'][aid]),
                anchors_within_25cm=int(len(anchor_tree.query_ball_point(p[:2],.25))),
                anchors_within_50cm=int(len(anchor_tree.query_ball_point(p[:2],.5))),
                anchors_within_1p5m=int(len(anchor_tree.query_ball_point(p[:2],1.5))),
                signed_clearance_to_existing_global_measured_plane_m=float(n@p-d),
                no_plane_fitting_or_surface_extrapolation=True)
            if hits:
                hit=hits[0];ids=hit['ids'];normal=hit['normal'];point=hit['world']
                q.update(surface_triangle_world_rows=ids.tolist(),barycentric=hit['bary'].tolist(),
                    fixed_surface_world_xyz_m=point.tolist(),fixed_surface_normal_world=normal.tolist(),
                    signed_clearance_to_fixed_triangle_m=float((p-point)@normal),vertical_gap_to_fixed_triangle_m=float(p[2]-point[2]),
                    triangle_vertex_supported=supported[ids].tolist(),
                    triangle_vertex_trusted_geometry=current['trusted_geometry'][ids].tolist(),
                    triangle_vertex_surface_interpolated=current['surface_interpolated'][ids].tolist(),
                    triangle_vertex_active_Gaussians=load(CURRENT/'road.npz')['active'][ids].tolist(),
                    triangle_vertex_material_source_raw=current['source_raw'][ids].tolist(),
                    triangle_vertex_current_geometry_match_max_m=float(np.linalg.norm(current_xyz[ids]-xyz[ids],axis=1).max()))
                geometry_evidence.append((label,contact['raw'],contact['wheel'],ids,xyz[ids],hit['bary']))
            results.append(q)
    result=dict(schema='a20.fixed_road_surface_contact.cpu.v1',
        method='Exact containing fixed-atlas triangle in world XY; existing supported grid plus unchanged348 measured-plane certified nodes. No new interpolation fit, activation, or geometry edit.',
        sources={k:obs.record(v) for k,v in [('base_atlas',BASE),('certificate',CERT),('certificate_json',CERT_JSON),('support',OLD),('anchors',ANCHORS),('current_provenance',CURRENT/'provenance.npz')]},
        original_contact_receipt=obs.record(args.original),adapted_contact_receipt=obs.record(args.adapted),
        fixed_anchors_source_raws=sorted(map(int,np.unique(anchors['source_raw']))),
        inherited_road_uses_full_sequence_measurements=True,new_heldout_raw_files_opened=[],
        newly_measured_road_data_added=False,post_fit_only_diagnostic=True,pose_refit_from_this_query=False,
        rows=results,gpu_used=False,Gaussian_coverage_and_surface_support_separated=True)
    obs.write(args.output/'fixed_surface_contact.json',result)
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['original','adapted','output']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())

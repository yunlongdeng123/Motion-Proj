"""Same BUILD-only CPU geometry diagnostics for the one registered semantic asset."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_completion_prior_20261122_r1')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',type=Path,default=ROOT/'semantic/omni_seed101_image_bbox_r1')
    ap.add_argument('--conditioning',type=Path,default=ROOT/'conditioning')
    ap.add_argument('--audit-core',type=Path,default=ROOT/'shape_cpu_audit_r1/source/shape_audit_cpu.py')
    ap.add_argument('--output',type=Path,default=ROOT/'semantic_cpu_audit_r1/results')
    ap.add_argument('--mesh-name',default='registered_actor_native.ply')
    ap.add_argument('--transform-name',default='registered_transform.json')
    ap.add_argument('--asset-label',default='registered_before_heading_fix')
    args=ap.parse_args(); started=time.monotonic()
    spec=importlib.util.spec_from_file_location('a20_cpu_core',args.audit_core)
    a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
    np=a.np; trimesh=a.trimesh
    mesh_path=args.candidate/args.mesh_name
    transform_path=args.candidate/args.transform_name
    points_path=args.conditioning/'lidar/aggregated_lidar_train_086_110.npz'
    provenance_path=args.conditioning/'lidar/provenance.json'
    pinned={p.name:a.record(p) for p in [mesh_path,transform_path,points_path,provenance_path,args.audit_core]}
    transform=json.loads(transform_path.read_text())
    mesh=trimesh.load(mesh_path,force='mesh',process=False)
    bank=np.load(points_path,allow_pickle=False)
    points=np.asarray(bank['xyz_canonical_m'],float); raw=np.asarray(bank['source_raw'])
    assert set(raw.tolist())=={86,110} and bool(bank['measured_lidar_only'])
    assert not bool(bank['native_gaussian_control_used'])
    args.output.mkdir(parents=True,exist_ok=False)
    distances,closest,face,method=a.exact_point_mesh(points,mesh)
    size=np.asarray(bank['size_lwh_m'],float)
    metrics=[a.camera_audit(mesh,points,distances,args.conditioning/'train'/f'{r:03d}',args.output) for r in [86,110]]
    flags=[]
    for row in metrics:
        if row['visible_core_recall']<.8: flags.append(f'raw{row["raw"]:03d}_visible_core_recall_below_diagnostic_0.8')
        if row['known_background_spill_fraction']>.05: flags.append(f'raw{row["raw"]:03d}_known_background_spill_above_diagnostic_0.05')
    if distances.mean()>.25: flags.append('mean_measurement_distance_above_diagnostic_0.25m')
    if np.quantile(distances,.9)>.5: flags.append('p90_measurement_distance_above_diagnostic_0.50m')
    report=dict(schema='a20.registered_semantic_cpu_audit.v1',asset_label=args.asset_label,human_verdict=None,
        numeric_status='DIAGNOSTIC_FLAGS_REVIEW_REQUIRED' if flags else 'NO_DIAGNOSTIC_FLAGS_VISUAL_REVIEW_REQUIRED',
        diagnostic_flags=flags,threshold_origin='Defined in first CPU audit, not root protocol preregistered; unchanged here.',
        metric_definition_unchanged=True,geometry_modified_by_this_audit=False,gpu_used=False,
        evaluated_raws=[86,110],heldout_used=False,source_point_count=len(points),
        source_raw_counts={str(r):int((raw==r).sum()) for r in [86,110]},
        point_to_surface_m=a.stats(distances),by_source_raw={str(r):a.stats(distances[raw==r]) for r in [86,110]},
        exact_distance_method=method,physical_size_xyz_m=size.tolist(),mesh_bounds_m=mesh.bounds.tolist(),
        mesh_extent_xyz_m=np.ptp(mesh.vertices,axis=0).tolist(),
        extent_to_physical_size=(np.ptp(mesh.vertices,axis=0)/size).tolist(),
        extent_note='Extents are forced by declared deterministic full-bounds registration, not native generator accuracy.',
        registration=transform,registration_source=a.record(transform_path),vertices=len(mesh.vertices),faces=len(mesh.faces),
        watertight=bool(mesh.is_watertight),consistent_winding=bool(mesh.is_winding_consistent),
        projection_metrics=metrics,provenance=pinned,wall_seconds=time.monotonic()-started,
        method_notes=['No LiDAR fit, candidate selection, sign selection, scaling or shape edits in this audit.',
                      'Distances now use genuine BUILD measurements withheld from the image+bbox generator.',
                      'The initial point-only and semantic candidate differ in image and control modality.',
                      'Unknown image pixels excluded; OBB potential not used as amodal silhouette GT.'])
    np.savez_compressed(args.output/'measured_point_surface_residuals.npz',xyz_canonical_m=points,source_raw=raw,
        point_to_surface_m=distances,closest_surface_xyz_m=closest,closest_face=face,
        representative_source_row=bank['representative_source_row'])
    (args.output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    for p in [mesh_path,transform_path,points_path,provenance_path,args.audit_core]:
        assert a.sha(p)==pinned[p.name]['sha256'], 'Read-only audit observed a source mutation'
    print(json.dumps({k:report[k] for k in ['numeric_status','diagnostic_flags','point_to_surface_m',
                       'projection_metrics','wall_seconds']},indent=2),flush=True)


if __name__=='__main__': main()

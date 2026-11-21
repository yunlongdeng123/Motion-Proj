"""Fixed root-authorized affine registration; no LiDAR-driven transform fit."""
from pathlib import Path
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import hashlib
import importlib.util
import json
import time
import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT=Path("/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_completion_prior_20261122_r1")
OUT=ROOT/"semantic/omni_seed101_image_bbox_r1"
TARGET=np.asarray([5.,2.5,2.08])


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    started=time.monotonic()
    if (OUT/"registered_actor_native.ply").exists():raise RuntimeError("Preserve existing registered output")
    originals={name:sha(OUT/name) for name in ["candidate_omni.glb","candidate_actor_native.ply","candidate_actor_native.obj","candidate_actor_native.glb","candidate_latents.pt","output_metric_transform.json"]}
    mesh=trimesh.load(OUT/"candidate_actor_native.ply",force="mesh",process=False)
    R=np.asarray([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    rotated=np.asarray(mesh.vertices)@R.T
    lo,hi=rotated.min(0),rotated.max(0);center=(lo+hi)/2;span=hi-lo
    if not np.isfinite(span).all() or (span<=0).any():raise RuntimeError("Finite positive generated extents required")
    scales=TARGET/span;S=np.diag(scales)
    initial_to_registered=np.eye(4);initial_to_registered[:3,:3]=S@R
    initial_to_registered[:3,3]=-S@center
    vertices=np.asarray(mesh.vertices)@initial_to_registered[:3,:3].T+initial_to_registered[:3,3]
    raw_to_initial=np.asarray(json.loads((OUT/"output_metric_transform.json").read_text())["normalized_omni_to_actor_native_4x4"])
    raw_to_registered=initial_to_registered@raw_to_initial
    registered=trimesh.Trimesh(vertices=vertices,faces=np.asarray(mesh.faces).copy(),process=False)
    registered.export(OUT/"registered_actor_native.ply")
    registered.export(OUT/"registered_actor_native.glb")
    np.savez_compressed(OUT/"registered_actor_native.npz",vertices_actor_native_m=vertices.astype(np.float32),
        vertices=vertices.astype(np.float32),faces=np.asarray(mesh.faces,np.int32),vertex_normals=np.asarray(registered.vertex_normals,np.float32),
        size_lwh_m=TARGET.astype(np.float32),raw_omni_to_registered_4x4=raw_to_registered,
        initial_to_registered_4x4=initial_to_registered,raw_omni_to_initial_4x4=raw_to_initial,
        registration_axis_scales_xyz=scales,generated_shape_prior=np.asarray(True),native_measurement_shape_gt=np.asarray(False))
    determinant_raw=float(np.linalg.det(raw_to_registered[:3,:3]));determinant_post=float(np.linalg.det(initial_to_registered[:3,:3]))
    transform=dict(schema="a20.omni_bbox_fixed_affine_registration.v1",registration_source="root-authorized fixed orientation and known complete OBB dimensions",
        normalized_raw_mesh=str(OUT/"candidate_omni.glb"),initial_metric_mesh=str(OUT/"candidate_actor_native.ply"),
        raw_omni_to_initial_4x4=raw_to_initial.tolist(),initial_metric_to_registered_4x4=initial_to_registered.tolist(),
        raw_omni_to_registered_4x4=raw_to_registered.tolist(),
        fixed_Rz_minus90_newx_oldy_newy_minusoldx=R.tolist(),rotated_generated_bounds_m=[lo.tolist(),hi.tolist()],
        rotated_generated_span_xyz_m=span.tolist(),rotated_generated_bbox_center_m=center.tolist(),
        target_size_lwh_m=TARGET.tolist(),axis_scales_after_rotation=scales.tolist(),
        registered_bounds_m=registered.bounds.tolist(),registered_span_xyz_m=np.ptp(vertices,axis=0).tolist(),
        post_registration_determinant=determinant_post,post_registration_reflection=determinant_post<0,
        raw_to_registered_determinant=determinant_raw,raw_to_registered_reflection=determinant_raw<0,
        winding_policy="Raw->initial axis swap already reversed faces; positive-determinant registration retains all initial faces unchanged",
        size_is_generator_native_prediction=False,dimension_matching_is_explicit_postprocessing=True,
        measured_lidar_used_to_fit_registration=False,heldout_used=False,direction_search_performed=False,
        deleted_vertices=0,deleted_faces=0,manual_mirror_removal=False,source_sha256=sha(__file__),original_files_sha256=originals)
    write(OUT/"registered_transform.json",transform)
    print("REGISTRATION_FROZEN_NO_LIDAR_USED "+json.dumps(dict(scales=scales.tolist(),target=TARGET.tolist(),post_reflection=determinant_post<0)),flush=True)
    # Measurements are opened only after the complete deterministic transform and asset are saved.
    with np.load(ROOT/"conditioning/lidar/aggregated_lidar_train_086_110.npz",allow_pickle=False) as bank:
        measured=np.asarray(bank["xyz_canonical_m"]);source_raw=np.asarray(bank["source_raw"])
    generation=load_module(ROOT/"semantic/source/propose_semantic_bbox_shape.py","_bbox_generation_helpers")
    samples=generation.surface_samples(vertices,np.asarray(mesh.faces),65536,101)
    distances=cKDTree(samples).query(measured,k=1,workers=2)[0]
    np.savez_compressed(OUT/"registered_surface_samples.npz",points_actor_native_m=samples,
        measured_to_surface_approx_m=distances,measurement_source_raw=source_raw)
    renderer=load_module(ROOT/"semantic/source/inspect_semantic_shape_cpu.py","_bbox_preview_helpers")
    previews=[]
    for name,view,label in [("01_front_right",[1,-1,.45],"Front right"),("02_rear_right",[-1,-1,.4],"Rear right"),("03_left",[0,1,.15],"Left side")]:
        previews.append(renderer.shaded_view(registered,TARGET,view,label,OUT/("registered_preview_"+name+".png")))
    contact=np.concatenate([cv2.imread(item["path"]) for item in previews],axis=1)
    cv2.imwrite(str(OUT/"registered_preview_three_views.jpg"),contact,[cv2.IMWRITE_JPEG_QUALITY,92])
    raw=trimesh.load(OUT/"candidate_omni.glb",force="mesh",process=False)
    mapped=np.asarray(raw.vertices)@raw_to_registered[:3,:3].T+raw_to_registered[:3,3]
    roundtrip=float(np.abs(mapped-vertices).max())
    if roundtrip>2e-6:raise RuntimeError("Raw->registered transformation serialization mismatch")
    if any(sha(OUT/name)!=value for name,value in originals.items()):raise RuntimeError("An original candidate file changed")
    report=dict(schema="a20.registered_shape_cpu_diagnostics.v1",status="FIXED_REGISTRATION_COMPLETE_PENDING_INDEPENDENT_REVIEW",
        registered_vertices=len(vertices),registered_faces=len(mesh.faces),full_topology_preserved=True,
        metric_bounds=registered.bounds.tolist(),target_size_lwh_m=TARGET.tolist(),dimension_fit_is_by_construction=True,
        measured_lidar_used_for_transform=False,measurement_raws=sorted(set(source_raw.tolist())),measurement_count=len(measured),heldout_used=False,
        measured_to_registered_surface_approx_m=dict(mean=float(distances.mean()),p50=float(np.quantile(distances,.5)),p90=float(np.quantile(distances,.9)),p99=float(np.quantile(distances,.99))),
        measured_to_registered_per_raw={str(raw):dict(count=int((source_raw==raw).sum()),mean=float(distances[source_raw==raw].mean()),p90=float(np.quantile(distances[source_raw==raw],.9))) for raw in sorted(set(source_raw.tolist()))},
        raw_omni_to_registered_roundtrip_max_abs_m=roundtrip,original_files_sha256_unchanged=True,
        previews=previews,cpu_wall_seconds=time.monotonic()-started,gpu_used=False,new_shape_generation_steps=0,texture_started=False,
        generated_artifacts_retained=["floating mirror fragments","roof ridge/spoiler-like protrusion","wheel-shape artifacts"],
        artifacts={name:dict(bytes=(OUT/name).stat().st_size,sha256=sha(OUT/name)) for name in ["registered_actor_native.ply","registered_actor_native.glb","registered_actor_native.npz","registered_transform.json","registered_surface_samples.npz","registered_preview_three_views.jpg"]})
    write(OUT/"registered_cpu_diagnostics.json",report);print(json.dumps(report),flush=True)


if __name__=="__main__":main()

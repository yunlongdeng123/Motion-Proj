"""Fixed 180-degree heading correction established by rendered nose/tailgate."""
from pathlib import Path
import os
os.environ.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
import json,hashlib,importlib.util,time
import numpy as np
import trimesh
import cv2
from scipy.spatial import cKDTree
ROOT=Path("/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_completion_prior_20261122_r1")
OUT=ROOT/"semantic/omni_seed101_image_bbox_r1"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")

def main():
    start=time.monotonic()
    if (OUT/"heading_fixed_actor_native.ply").exists():raise RuntimeError("Preserve existing heading-fixed asset")
    mesh=trimesh.load(OUT/"registered_actor_native.ply",force="mesh",process=False)
    original_sha=sha(OUT/"registered_actor_native.ply")
    R=np.diag([-1.,-1.,1.]);post=np.eye(4);post[:3,:3]=R
    vertices=np.asarray(mesh.vertices)@R.T
    fixed=trimesh.Trimesh(vertices=vertices,faces=np.asarray(mesh.faces).copy(),process=False)
    chain=json.loads((OUT/"registered_transform.json").read_text())
    raw_to_fixed=post@np.asarray(chain["raw_omni_to_registered_4x4"])
    fixed.export(OUT/"heading_fixed_actor_native.ply");fixed.export(OUT/"heading_fixed_actor_native.glb")
    np.savez_compressed(OUT/"heading_fixed_actor_native.npz",vertices_actor_native_m=vertices.astype(np.float32),vertices=vertices.astype(np.float32),
        faces=np.asarray(mesh.faces,np.int32),vertex_normals=np.asarray(fixed.vertex_normals,np.float32),size_lwh_m=np.asarray([5.,2.5,2.08],np.float32),
        post_actor_transform_4x4=post,raw_omni_to_heading_fixed_4x4=raw_to_fixed,generated_shape_prior=np.asarray(True))
    transform=dict(schema="a20.semantic_heading_fixed_transform.v1",source_registered_mesh=str(OUT/"registered_actor_native.ply"),
        source_registered_sha256=original_sha,post_actor_transform_4x4=post.tolist(),registered_to_heading_fixed_4x4=post.tolist(),
        raw_omni_to_heading_fixed_4x4=raw_to_fixed.tolist(),raw_omni_to_registered_4x4=chain["raw_omni_to_registered_4x4"],
        source_full_registration_transform=str(OUT/"registered_transform.json"),fixed_rotation="Rz(pi): x=-x, y=-y, z unchanged",
        correction_reason="Rendered +X view previously showed tailgate/cargo bed and -X view showed nose/grille; fixed 180 degree yaw puts nose at actor +X",
        direction_choice_source="Root and independent agent semantic inspection of actual fixed three-view previews; no LiDAR optimization or direction search",
        measured_lidar_used_to_select_heading=False,heldout_used=False,determinant=float(np.linalg.det(R)),reflection=False,faces_reversed=False,
        topology_unchanged=True,generated_artifacts_retained=True,texture_uv_transform="UV unchanged; apply this same post_actor_transform after texture inverse mapping",
        metric_bounds_m=fixed.bounds.tolist(),semantic_front_plusX_verified_by_nose_grille_vs_tailgate=False,requires_actual_corrected_preview_confirmation=True,
        source_sha256=sha(__file__))
    write(OUT/"heading_fixed_transform.json",transform)
    with np.load(OUT/"registered_surface_samples.npz",allow_pickle=False) as bank:samples=bank["points_actor_native_m"]@R.T
    with np.load(ROOT/"conditioning/lidar/aggregated_lidar_train_086_110.npz",allow_pickle=False) as bank:
        points=bank["xyz_canonical_m"];raws=bank["source_raw"]
    distances=cKDTree(samples).query(points,k=1,workers=2)[0]
    np.savez_compressed(OUT/"heading_fixed_surface_samples.npz",points_actor_native_m=samples,measured_to_surface_approx_m=distances,measurement_source_raw=raws)
    helper=ROOT/"semantic/source/inspect_semantic_shape_cpu.py";spec=importlib.util.spec_from_file_location("_heading_preview",helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    previews=[]
    for name,view,label in [("01_front_right",[1,-1,.45],"Front right"),("02_rear_right",[-1,-1,.4],"Rear right"),("03_left",[0,1,.15],"Left side")]:
        previews.append(module.shaded_view(fixed,np.asarray([5.,2.5,2.08]),view,label,OUT/("heading_fixed_preview_"+name+".png")))
    cv2.imwrite(str(OUT/"heading_fixed_preview_three_views.jpg"),np.concatenate([cv2.imread(p["path"]) for p in previews],axis=1),[cv2.IMWRITE_JPEG_QUALITY,92])
    if sha(OUT/"registered_actor_native.ply")!=original_sha:raise RuntimeError("Original registration changed")
    report=dict(schema="a20.heading_fixed_cpu_diagnostics.v1",vertices=len(vertices),faces=len(mesh.faces),metric_bounds=fixed.bounds.tolist(),
        measured_to_surface_approx_m=dict(mean=float(distances.mean()),p50=float(np.quantile(distances,.5)),p90=float(np.quantile(distances,.9)),p99=float(np.quantile(distances,.99))),
        per_raw={str(raw):dict(count=int((raws==raw).sum()),mean=float(distances[raws==raw].mean()),p90=float(np.quantile(distances[raws==raw],.9))) for raw in sorted(set(raws.tolist()))},
        measured_lidar_used_for_transform_or_direction=False,heldout_used=False,previews=previews,original_registration_preserved=True,
        source_sha256=sha(__file__),cpu_wall_seconds=time.monotonic()-start,gpu_used=False,new_generation_steps=0,
        artifacts={name:dict(bytes=(OUT/name).stat().st_size,sha256=sha(OUT/name)) for name in ["heading_fixed_actor_native.ply","heading_fixed_actor_native.glb","heading_fixed_actor_native.npz","heading_fixed_transform.json","heading_fixed_preview_three_views.jpg"]})
    write(OUT/"heading_fixed_cpu_diagnostics.json",report);print(json.dumps(report),flush=True)

if __name__=="__main__":main()

"""CPU-only world-patch selection, source rebuild and auditable isolation.

No model/renderer/checkpoint is imported. Selection never loads RGB arrays.
The public inputs are distinct from V44 integration and evaluator-only truth.
Physical all-sensor strictness is deliberately not certified by this script.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import time
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
sys.dont_write_bytecode = True

BASE = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
FULL = BASE / "r15_gn11_fullclip_cpu_20261026"
GEOMETRY = BASE / "shadow_a5_a37_199_20261103/inputs/expanded/evidence.npz"
BUNDLE = BASE / "r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json"
SEED = "scene008-R1-FRONT-geometry-only-20261107-v1"
SCHEMA = "scene008.road_patch_manifest.v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Preserve existing benchmark artifact: " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def stable(text):
    return hashlib.sha256((SEED + "|" + text).encode()).hexdigest()


def inside_xy(xyz, box):
    return (xyz[:, 0] >= box[0]) & (xyz[:, 0] <= box[2]) & (xyz[:, 1] >= box[1]) & (xyz[:, 1] <= box[3])


def in_prisms(xyz, patches):
    mask = np.zeros(len(xyz), bool)
    for patch in patches:
        z = patch["context_z_interval_m"]
        mask |= inside_xy(xyz, patch["guard_xy_bounds_local_m"]) & (xyz[:, 2] >= z[0]) & (xyz[:, 2] <= z[1])
    return mask


def overlaps(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def select(args):
    start = time.time()
    out = args.output
    contract = read(args.contract)
    guard = float(contract["masking"]["guard_m"])
    # Explicit key allowlist: no source_rgb, eligibility, colors or fit errors.
    allowed_keys = ("keys", "local_xyz", "surface_supported", "source_raw")
    with np.load(args.geometry, allow_pickle=False) as source:
        a = {key: source[key].copy() for key in allowed_keys}
    assert np.array_equal(a["source_raw"], np.arange(199))
    xyz = np.asarray(a["local_xyz"], float)
    # The old source_valid also used full-image sharpness. Recompute direct
    # measured FRONT witnesses with no RGB/quality gate or appearance cache.
    witness = np.zeros(len(xyz),np.int16)
    witness_frame_rows=[]
    for raw in range(199):
        path=FULL/"frames"/f"{raw:03d}.npz"
        with np.load(path,allow_pickle=False) as frame:
            K,w=frame["K"],frame["w2l"]
            q=xyz@w[:3,:3].T+w[:3,3]
            uv=(q@K.T)[:,:2]/np.maximum(q[:,2:],1e-9)-.5
            cloud=frame["cloud"].astype(float)
            cq=cloud@w[:3,:3].T+w[:3,3]
            cuv=(cq@K.T)[:,:2]/np.maximum(cq[:,2:],1e-9)-.5
            dist,near=cKDTree(cuv).query(uv)
            ix=np.floor(uv[:,0]+.5).astype(int);iy=np.floor(uv[:,1]+.5).astype(int)
            inside=(q[:,2]>.1)&(ix>=0)&(ix<480)&(iy>=0)&(iy<320)
            guarded=np.ones(len(xyz),bool)
            ids=np.flatnonzero(inside)
            guarded[ids]=frame["guard"][iy[ids],ix[ids]]
            ok=inside&~guarded&(dist<=1.5)&frame["ground"][near]&(np.abs(cq[near,2]-q[:,2])<=.3)
            witness+=ok
            witness_frame_rows.append(dict(raw=raw,qualified_direct_geometry_witnesses=int(ok.sum()),source_geometry_cache_sha256=sha(path)))
        if raw%50==0:
            print(json.dumps(dict(stage="geometry_only_witness",raw=raw,seconds=time.time()-start)),flush=True)
    support = np.asarray(a["surface_supported"], bool)
    with np.load(FULL / "full199_ground_pool.npz", allow_pickle=False) as source:
        pool = source["local_xyz"].astype(float)
    tree = cKDTree(xyz[:, :2])
    pooltree = cKDTree(pool[:, :2])
    anchor = np.asarray(read(BUNDLE)["coordinate_anchor"], float)
    selected, candidate_rows = [], []
    for size in (2.0, 1.0, 0.5):
        candidates = []
        for x in np.arange(4., 30.51, .5):
            for y in np.arange(-3., 6.01, .5):
                box = [float(x-size/2), float(y-size/2), float(x+size/2), float(y+size/2)]
                guardbox = [box[0]-guard, box[1]-guard, box[2]+guard, box[3]+guard]
                ids = tree.query_ball_point([x, y], size*.72)
                ids = np.asarray(ids, int)
                ids = ids[inside_xy(xyz[ids], box)]
                expected = (size / .08)**2
                if len(ids) < .80*expected:
                    continue
                geom_ok = support[ids] & (witness[ids] >= 3)
                if float(geom_ok.mean()) < .85:
                    continue
                ii = pooltree.query_ball_point([x, y], size*.72)
                pts = pool[np.asarray(ii, int)]
                nreal = int(inside_xy(pts, box).sum())
                if nreal < max(8, int(size*size*25)):
                    continue
                token = f"{size:.1f}|{x:.2f}|{y:.2f}"
                split = "validation" if int(stable("split|" + token)[:8], 16) % 2 == 0 else "sealed_test"
                row = dict(size_m=size, split=split, center_local_xy_m=[float(x), float(y)],
                           xy_bounds_local_m=box, guard_xy_bounds_local_m=guardbox,
                           rank=stable("rank|" + token), candidate_geometry_cells=len(ids),
                           geometry_ge3_FRONT_witness_fraction=float(geom_ok.mean()),
                           target_actual_pool_return_count=nreal)
                candidates.append(row)
        candidates.sort(key=lambda r: r["rank"])
        candidate_rows.append(dict(size_m=size, qualified_candidates=len(candidates)))
        for split in ("validation", "sealed_test"):
            legal = [r for r in candidates if r["split"] == split and not any(overlaps(r["guard_xy_bounds_local_m"], p["guard_xy_bounds_local_m"]) for p in selected)]
            if not legal:
                raise ValueError(f"No geometry-only non-overlapping candidate for {size}/{split}")
            row = legal[0]
            row["patch_id"] = f"road_{int(size*100):03d}cm_{split}_{row['rank'][:10]}"
            selected.append(row)
    # Derive masking interval strictly from context outside every selected halo.
    legal_context = np.ones(len(pool), bool)
    for patch in selected:
        legal_context &= ~inside_xy(pool, patch["guard_xy_bounds_local_m"])
    for patch in selected:
        center = np.asarray(patch["center_local_xy_m"])
        context = pool[legal_context & (np.linalg.norm(pool[:, :2]-center, axis=1) <= 8.)]
        if len(context) < 100:
            raise ValueError("Insufficient context to declare geometry-blind height interval")
        interval = [float(context[:, 2].min()-.75), float(context[:, 2].max()+.75)]
        patch["context_z_interval_m"] = interval
        patch["context_height_source_points"] = len(context)
        patch["mask_height_uses_target_geometry"] = False
        box = patch["xy_bounds_local_m"]
        corners = np.array([[box[0], box[1], interval[0]], [box[2], box[1], interval[0]],
                            [box[2], box[3], interval[0]], [box[0], box[3], interval[0]]])
        patch["world_target_corners_m"] = (corners @ anchor[:3, :3].T + anchor[:3, 3]).tolist()
        patch["material_stratum"] = "evaluator_only_unannotated"
        patch["slope_reference"] = "sealed_measured_height_proxy_pending_not_independent_normals"
    manifest = dict(schema=SCHEMA, selection_seed=SEED, selection_reads_RGB=False,
                    selection_reads_fit_errors=False, selection_array_allowlist=list(allowed_keys),
                    coordinate_system="anchor-local metres; coordinate_anchor maps to fixed world metres",
                    coordinate_anchor=anchor.tolist(), declared_cameras=[1], raw_ids=list(range(199)),
                    guard_m=guard, native_pixel_padding=16, patches=selected,
                    strict_all_sensor_blinding_certified=False,
                    other_camera_scope="FRONT only; no surround RGB audit or all-camera claim",
                    selection_hash=stable(json.dumps(selected, sort_keys=True)),
                    witness_rule="Nearest real FRONT return<=1.5px, ground flag, depth residual<=.3m, annotation guard clear; no image sharpness or RGB.",
                    candidate_summary=candidate_rows)
    write(out / "public/patch_manifest.json", manifest)
    write(out / "audit/selection_receipt.json", dict(status="PASS_GEOMETRY_ONLY_PATCH_SELECTION", CPU_only=True,
          source_geometry_sha256=sha(args.geometry), pool_sha256=sha(FULL/"full199_ground_pool.npz"),
          source_array_allowlist=list(allowed_keys), old_source_valid_not_read=True,
          recomputed_no_RGB_witness_frames=witness_frame_rows,target_RGB_read=False,fit_error_read=False,
          patch_manifest_sha256=sha(out/"public/patch_manifest.json"), patches=len(selected),
          split_counts={s: sum(p["split"] == s for p in selected) for s in ("validation", "sealed_test")},
          seconds=time.time()-start, model_updates=0, GPU_used=False))
    print(json.dumps({"stage": "select", "patches": [{k:p[k] for k in ("patch_id", "size_m", "split", "xy_bounds_local_m", "target_actual_pool_return_count")} for p in selected]}), flush=True)


def interval_mul(a, b):
    p = [a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]]
    return [min(p), max(p)]


def interval_scale(a, scale):
    return [min(a[0]*scale, a[1]*scale), max(a[0]*scale, a[1]*scale)]


def interval_add(*items):
    return [sum(i[0] for i in items), sum(i[1] for i in items)]


def interval_square(a):
    return [0. if a[0] <= 0 <= a[1] else min(a[0]**2, a[1]**2), max(a[0]**2, a[1]**2)]


def projected_prism_rectangle(patch, w, intrinsic, width, height, padding=16, motion_margin_m=0.):
    """Conservative Brown interval image rectangle, rather than GT-height silhouette."""
    xy, z = patch["guard_xy_bounds_local_m"], patch["context_z_interval_m"]
    lo = np.array([xy[0], xy[1], z[0]]) - motion_margin_m
    hi = np.array([xy[2], xy[3], z[1]]) + motion_margin_m
    corners = np.array([[x, y, zz] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for zz in (lo[2], hi[2])])
    q = corners @ w[:3, :3].T + w[:3, 3]
    if q[:, 2].max() <= .1:
        return None
    if q[:, 2].min() <= .1:
        return [0, 0, width, height]
    xn, yn = q[:, 0]/q[:, 2], q[:, 1]/q[:, 2]
    x, y = [float(xn.min()), float(xn.max())], [float(yn.min()), float(yn.max())]
    xx, yy = interval_square(x), interval_square(y)
    r2 = interval_add(xx, yy)
    r4, r6 = interval_mul(r2, r2), interval_mul(interval_mul(r2, r2), r2)
    fu, fv, cu, cv, k1, k2, p1, p2, k3 = intrinsic
    radial = interval_add([1., 1.], interval_scale(r2,k1), interval_scale(r4,k2), interval_scale(r6,k3))
    xyterm = interval_mul(x,y)
    dx = interval_add(interval_mul(x,radial), interval_scale(xyterm,2*p1), interval_scale(interval_add(r2,interval_scale(xx,2)),p2))
    dy = interval_add(interval_mul(y,radial), interval_scale(interval_add(r2,interval_scale(yy,2)),p1), interval_scale(xyterm,2*p2))
    u, v = interval_add(interval_scale(dx,fu),[cu-.5,cu-.5]), interval_add(interval_scale(dy,fv),[cv-.5,cv-.5])
    if not np.isfinite(u+v).all():
        return [0,0,width,height]
    box = [max(0,int(np.floor(u[0]))-padding), max(0,int(np.floor(v[0]))-padding),
           min(width,int(np.ceil(u[1]))+padding+1), min(height,int(np.ceil(v[1]))+padding+1)]
    return None if box[2] <= box[0] or box[3] <= box[1] else box


def blind_native_image(rgb, rectangles, output_hw=(320,480)):
    masked = rgb.copy()
    invalid = np.zeros(rgb.shape[:2], bool)
    for box in rectangles:
        if box is not None:
            x0,y0,x1,y1 = box
            invalid[y0:y1,x0:x1] = True
    masked[invalid] = 127
    h,w = output_hw
    resized = np.array(Image.fromarray(masked).resize((w,h),Image.Resampling.LANCZOS),copy=True)
    # The rectangles already have 16 native pixels of filter/DCT padding.
    # Only fully valid native BOX footprints are marked as observed sources.
    valid = np.asarray(Image.fromarray((~invalid).astype(np.uint8)*255).resize((w,h),Image.Resampling.BOX)) == 255
    resized[~valid] = 127
    return resized, valid, invalid


def raw_reader():
    sys.path.insert(0,str(BASE/"nearby_actors_20261015_v1"))
    import waymo_numpy_cpu as reader
    # Fast decoder is the same immutable existing CPU dependency as extraction.
    fast = Path("/root/autodl-tmp")
    sys.path.insert(0,str(fast))
    fast_path=fast/"fast_range_matrix.py"
    reader.reference_matrix=reader.matrix
    if not fast_path.exists() or sha(fast_path)!="6d07d45ed9304ca134ebe0e52f3aca0382193cd115e0cafcacf64df9db1455b1":
        raise ValueError("Expected frozen existing CPU MatrixFloat decoder")
    from fast_range_matrix import matrix
    reader.matrix=matrix
    reader.fast_matrix_dependency=dict(path=str(fast_path),sha256=sha(fast_path))
    return reader


def record_at(stream, row, crc):
    stream.seek(row["record"]["header_offset"])
    header = stream.read(12)
    length, checksum = struct.unpack("<QI",header)
    if length != row["record"]["payload_length"] or crc(header[:8]) != checksum:
        raise ValueError("Indexed record/length CRC mismatch")
    payload, trailer = stream.read(length), stream.read(4)
    if len(payload) != length or crc(payload) != struct.unpack("<I",trailer)[0]:
        raise ValueError("Raw payload CRC mismatch")
    return payload


def rebuild(args):
    start=time.time()
    out=args.output
    manifest=read(out/"public/patch_manifest.json")
    patches=manifest["patches"]
    anchor=np.asarray(manifest["coordinate_anchor"],float)
    invanchor=np.linalg.inv(anchor)
    reader=raw_reader();pb=reader.protobuf();crc=reader.crc_function()
    extraction=read(FULL/"extraction.json")
    raws=list(range(199)) if args.raws == "all" else [int(x) for x in args.raws.split(",")]
    stage=args.stage_tag or ("full199" if raws == list(range(199)) else "smoke_"+"_".join(map(str,raws)))
    target=out/"public/inputs"/stage
    if target.exists():
        raise FileExistsError("Preserve existing rebuilt input stage")
    tracks=("appearance_blind","geometry_RGB_blind")
    for track in tracks:
        for sub in ("rgb","geometry","camera"):
            (target/track/sub).mkdir(parents=True)
        write(target/track/"patch_manifest.json",manifest)
    rows=[]
    with Path(extraction["raw_path"]).open("rb") as stream:
        for raw in raws:
            payload=record_at(stream,extraction["frames"][raw],crc)
            frame=pb.Frame.FromString(payload)
            if frame.context.name != extraction["segment"]:
                raise ValueError("Wrong source scene")
            image=next(x for x in frame.images if x.name == 1)
            cal=next(x for x in frame.context.camera_calibrations if x.name == 1)
            intrinsic=np.asarray(cal.intrinsic,float)
            if len(intrinsic) != 9:
                raise ValueError("Expected complete Brown intrinsics")
            camera_to_kitti=np.eye(4);camera_to_kitti[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]]
            image_pose=np.asarray(image.pose.transform,float).reshape(4,4)
            w=camera_to_kitti@np.linalg.inv(np.asarray(cal.extrinsic.transform,float).reshape(4,4))@np.linalg.inv(image_pose)@anchor
            velocity=np.array([image.velocity.v_x,image.velocity.v_y,image.velocity.v_z])
            angular=np.array([image.velocity.w_x,image.velocity.w_y,image.velocity.w_z])
            dt=max(abs(float(image.camera_trigger_time)-float(image.pose_timestamp)),abs(float(image.camera_readout_done_time)-float(image.pose_timestamp)))
            # Constant-velocity envelope is recorded as a bound assumption.
            max_range=100.
            motion=float(np.linalg.norm(velocity)*dt+max_range*np.linalg.norm(angular)*dt)
            rectangles=[projected_prism_rectangle(p,w,intrinsic,int(cal.width),int(cal.height),manifest["native_pixel_padding"],motion) for p in patches]
            with Image.open(io.BytesIO(image.image)) as decoded:
                rgb=np.asarray(decoded.convert("RGB"))
            if rgb.shape[:2] != (int(cal.height),int(cal.width)):
                raise ValueError("Native image/calibration dimension mismatch")
            blind,valid,native_invalid=blind_native_image(rgb,rectangles)
            poisoned=rgb.copy();poisoned[native_invalid]=255-rgb[native_invalid]
            control,control_valid,_=blind_native_image(poisoned,rectangles)
            if not np.array_equal(blind,control) or not np.array_equal(valid,control_valid):
                raise AssertionError("Target-color intervention changed blinded input")
            for track in tracks:
                Image.fromarray(blind).save(target/track/"rgb"/f"{raw:03d}.png")
                np.savez_compressed(target/track/"rgb"/f"{raw:03d}_valid.npz",valid=valid)
            cloud,details=reader.lidar(frame,pb)
            frame_pose=np.asarray(frame.pose.transform,float).reshape(4,4)
            xyz=cloud[:,:3].astype(float)@(invanchor@frame_pose)[:3,:3].T+(invanchor@frame_pose)[:3,3]
            remove=in_prisms(xyz,patches)
            # XYZ only: even the depth-known track withholds intensity/material channels.
            np.savez_compressed(target/"appearance_blind/geometry"/f"{raw:03d}.npz",local_xyz=xyz.astype(np.float32))
            np.savez_compressed(target/"geometry_RGB_blind/geometry"/f"{raw:03d}.npz",local_xyz=xyz[~remove].astype(np.float32))
            if in_prisms(xyz[~remove],patches).any():
                raise AssertionError("Withheld geometry survived prism filter")
            camera=dict(raw=raw,camera_enum=1,native_hw=[int(cal.height),int(cal.width)],prepared_hw=[320,480],
                        intrinsic_Brown=intrinsic.tolist(),anchor_local_to_camera=w.tolist(),
                        pose_timestamp=float(image.pose_timestamp),trigger=float(image.camera_trigger_time),
                        readout_done=float(image.camera_readout_done_time),rolling_shutter_direction=int(cal.rolling_shutter_direction),
                        motion_bound_constant_velocity_m=motion,prism_rectangles_native=rectangles,
                        pixel_convention="native uv=distort(camera_xy/z)*f+c-.5; cached 480 outputs use uniform 1/4 scaling",
                        physical_timing_bound_independently_validated=False)
            for track in tracks:
                write(target/track/"camera"/f"{raw:03d}.json",camera)
            rows.append(dict(raw=raw,camera_enum=1,record_CRC_passed=True,source_payload_sha256=hashlib.sha256(payload).hexdigest(),
                             source_FRONT_JPEG_sha256=hashlib.sha256(image.image).hexdigest(),input_return_count=len(xyz),
                             geometry_RGB_blind_removed_points=int(remove.sum()),native_masked_pixels=int(native_invalid.sum()),
                             prepared_invalid_pixels=int((~valid).sum()),target_colour_intervention_exact=True,
                             raw_LiDAR_sensor_return_pairs=len(details),motion_bound_constant_velocity_m=motion,
                             rgb_file_sha256=sha(target/"geometry_RGB_blind/rgb"/f"{raw:03d}.png")))
            if raw%10 == 0 or raw == raws[-1]:
                print(json.dumps(dict(stage=stage,raw=raw,seconds=time.time()-start,removed=int(remove.sum()))),flush=True)
    for track in tracks:
        root=target/track
        ownfiles={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob("*")) if p.is_file()}
        write(root/"input_manifest.json",dict(schema="scene008.R1.isolated_input_track.v1",track=track,raws=raws,
              declared_cameras=[1],input_files=ownfiles,only_this_directory_may_be_mounted=True,
              other_track_and_evaluator_access_allowed=False,old_scene_dependency_access_allowed=False,
              physical_all_sensor_strict_blinding_certified=False))
    files={str(p.relative_to(target)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(target.rglob("*")) if p.is_file()}
    write(out/"audit"/f"source_rebuild_{stage}.json",dict(status="FRONT_RAW_SOURCE_REBUILD_PASS_UNDER_DECLARED_SENSOR_MODEL",
          stage=stage,raws=raws,complete_FRONT199=raws==list(range(199)),declared_cameras=[1],all_five_RGB_cameras_audited=False,
          physical_all_sensor_strict_blinding_certified=False,external_estimator_input_isolation_audited=False,
          old_caches_or_checkpoints_used_as_estimator_input=False,raw_XYZ_rebuilt_before_filtering=True,
          CPU_matrix_dependency=reader.fast_matrix_dependency,
          target_color_intervention_exact=True,neural_updates=0,GPU_used=False,rows=rows,
          input_files=files,patch_manifest_sha256=sha(out/"public/patch_manifest.json"),seconds=time.time()-start,
          unresolved=["Rolling-shutter constant-velocity envelope not independently validated.","Other camera RGB excluded, not audited.","Evaluator directory separation is not OS/process isolation.","No candidate completion, external sealed evaluation, or learned-prior result has run."]))


def validate(args):
    out=args.output;m=read(out/"public/patch_manifest.json")
    checks=[]
    def check(name,value):
        checks.append(dict(name=name,passed=bool(value)))
        if not value:
            raise AssertionError(name)
    check("schema",m["schema"]==SCHEMA)
    check("selection_no_RGB_or_fit_errors",m["selection_reads_RGB"] is False and m["selection_reads_fit_errors"] is False)
    check("all199_FRONT_declared",m["raw_ids"]==list(range(199)) and m["declared_cameras"]==[1])
    check("guard_frozen",m["guard_m"]>=.75)
    for i,p in enumerate(m["patches"]):
        check("no_target_height_for_mask:"+p["patch_id"],p["mask_height_uses_target_geometry"] is False)
        for q in m["patches"][:i]:
            check("no_overlap:"+p["patch_id"]+":"+q["patch_id"],not overlaps(p["guard_xy_bounds_local_m"],q["guard_xy_bounds_local_m"]))
    roots=sorted((out/"public/inputs").glob("*")) if (out/"public/inputs").exists() else []
    for root in roots:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            check("allowed_input_extension:"+str(p.relative_to(out)),p.suffix.lower() in (".npz",".png",".json"))
            check("not_symlink:"+str(p.relative_to(out)),not p.is_symlink())
        for p in sorted((root/"geometry_RGB_blind/geometry").glob("*.npz")):
            with np.load(p,allow_pickle=False) as a:
                check("XYZ_only:"+str(p.relative_to(out)),a.files==["local_xyz"])
                check("target_guard_removed:"+str(p.relative_to(out)),not in_prisms(a["local_xyz"],m["patches"]).any())
        for p in sorted((root/"appearance_blind/geometry").glob("*.npz")):
            with np.load(p,allow_pickle=False) as a:
                check("known_geometry_no_material_channel:"+str(p.relative_to(out)),a.files==["local_xyz"])
    name="validation_"+args.validation_tag+".json"
    write(out/"audit"/name,dict(status="PASS_PREPARATION_CHECKS_NOT_PHYSICAL_STRICTNESS_CERTIFICATE",checks=checks,
          check_count=len(checks),all_checks_passed=True,model_or_renderer_imported=any(k.startswith(("torch","tensorflow")) for k in sys.modules),
          strict_all_sensor_blinding_certified=False,old_checkpoint_conditioning_allowed=False,
          required_before_estimator=["Mount/copy only chosen public inputs, audit actual process reads.","Do not expose evaluator_only or old full-scene dependencies.","Freeze candidates/settings before external sealed_test evaluation."]))
    print(json.dumps(dict(stage="validate",checks=len(checks),passed=True)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("select","rebuild","validate"));p.add_argument("--output",type=Path,required=True)
    p.add_argument("--contract",type=Path,default=Path(__file__).with_name("benchmark_contract.json"));p.add_argument("--geometry",type=Path,default=GEOMETRY)
    p.add_argument("--raws",default="0,137,198");p.add_argument("--stage-tag",default=None);p.add_argument("--validation-tag",default="initial")
    args=p.parse_args();globals()[args.stage](args)


if __name__ == "__main__":
    main()

"""Three independent read-only checks of LiDAR convention and local surface physics."""
import argparse, hashlib, importlib.util, json, shutil, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.ndimage import binary_dilation
from diagnose_nonplanar_inference_cpu import (load, read, write, pin, stats, design, normals, medoids, fit)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def angle(a, b):
    a = a/np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-15)
    b = b/np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-15)
    return np.degrees(np.arccos(np.clip(abs((a*b).sum(-1)), 0, 1)))


def signed_and_abs(x):
    return dict(signed=stats(x), absolute=stats(abs(np.asarray(x))))


def ego_cloud(bank, raw):
    obs = read(bank/f"observations/{raw:03d}.json")
    lidar = np.fromfile(bank/obs["lidar_path"], dtype="<f4").reshape(-1, 6)
    pose = np.asarray(obs["ego_world"]).reshape(4, 4)
    return lidar[:, :3].astype(np.float64)@pose[:3, :3].T+pose[:3, 3], lidar, obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    C, out = args.campaign_root, args.output
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    R10 = C/"road_underbody_local_20261123_r10"
    strict = R10/"audit/local_surface_geometry_v1"
    s = load(strict/"local_surface_geometry.npz")
    m = load(strict/"strict_measurements.npz")
    contract = read(strict/"geometry_qualification_contract.json")
    bank = Path("/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs")
    receipt = read(bank/"receipt.json")
    decoder_root = Path("/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101")
    decoder_path = decoder_root/"waymo_numpy_cpu.py"
    matrix_path = decoder_root/"fast_range_matrix.py"
    prep = bank.parent/"source/prepare_cpu.py"
    sources = dict(decoder=pin(decoder_path), fast_matrix=pin(matrix_path), exporter=pin(prep),
                   receipt=pin(bank/"receipt.json"), raw_TFRecord=pin(receipt["raw"]))
    assert sources["decoder"]["sha256"] == receipt["decoder_sha256"]
    assert sources["fast_matrix"]["sha256"] == receipt["matrix_decoder_sha256"]
    assert sources["raw_TFRecord"]["sha256"] == receipt["raw_sha256"]
    sourceout = out/"source_snapshots"; sourceout.mkdir()
    for path in [decoder_path, matrix_path, prep]:
        shutil.copy2(path, sourceout/path.name)
    decoder = module("waymo_numpy_cpu_physics_check", decoder_path)
    decoder.matrix = module("fast_matrix_physics_check", matrix_path).matrix
    pb = decoder.protobuf()
    targets = {86, 110, 137}
    frame_rows = []
    world_by_raw = {}
    for r, payload, crc in decoder.records(receipt["raw"], max(targets)):
        if r not in targets:
            continue
        frame = pb.Frame.FromString(payload)
        regenerated, details = decoder.lidar(frame, pb)
        world, original, obs = ego_cloud(bank, r)
        world_by_raw[r] = world
        assert regenerated.shape == original.shape
        same_bytes = regenerated.astype("<f4").tobytes() == original.tobytes()
        pose = np.asarray(frame.pose.transform).reshape(4, 4)
        front = next(i for i in frame.images if i.name == 1)
        obs_pose = np.asarray(obs["ego_world"]).reshape(4, 4)
        txtpose = np.loadtxt(bank/f"pose/0065{r:03d}.txt")
        timestamp_file = int((bank/f"timestamp/0065{r:03d}.txt").read_text().strip())
        frame_record = next(x for x in receipt["frames"] if x["raw"] == r)
        camera_cal = next(x for x in frame.context.camera_calibrations if x.name == 1)
        camera_times = {}
        for name in ["pose_timestamp", "camera_trigger_time", "camera_readout_done_time"]:
            if hasattr(front, name):
                val = float(getattr(front, name))
                camera_times[name] = val
                camera_times[name+"_minus_frame_seconds"] = val-frame.timestamp_micros*1e-6
        segmentations = []
        for laser in frame.lasers:
            for ret in [1, 2]:
                item = getattr(laser, f"ri_return{ret}")
                value = getattr(item, "segmentation_label_compressed", b"")
                segmentations.append(dict(laser_name=int(laser.name), return_index=ret-1,
                    official_point_segmentation_compressed_bytes=len(value)))
        row = dict(raw=r, timestamp_micros=int(frame.timestamp_micros),
            all_timestamp_bindings_exact=bool(frame.timestamp_micros == obs["timestamp_micros"] == timestamp_file == frame_record["timestamp_micros"]),
            original_bin_redecode_byteexact=same_bytes,
            per_column_max_abs_difference=np.max(abs(regenerated-original), axis=0).tolist(),
            row_count=len(original), sensor_return_details=details, original_TFRecord_CRC=crc,
            frame_pose_vs_obs_max_abs=float(abs(pose-obs_pose).max()),
            frame_pose_vs_posefile_max_abs=float(abs(pose-txtpose).max()),
            camera_pose_vs_obs_max_abs=float(abs(np.asarray(front.pose.transform)-obs["camera_pose_world"]).max()),
            camera_extrinsic_vs_obs_max_abs=float(abs(np.asarray(camera_cal.extrinsic.transform)-obs["camera_extrinsic"]).max()),
            camera_intrinsics_vs_obs_max_abs=float(abs(np.asarray(camera_cal.intrinsic)-obs["camera_intrinsic_Brown"]).max()),
            front_camera_timing=camera_times, official_point_segmentation_presence=segmentations,
            obs_lidar_coordinates=obs.get("lidar_coordinates"), original_lidar=pin(bank/obs["lidar_path"]),
            original_observation=pin(bank/f"observations/{r:03d}.json"))
        assert row["all_timestamp_bindings_exact"]
        assert row["frame_pose_vs_obs_max_abs"] == 0
        frame_rows.append(row)
        print(json.dumps(dict(stage="raw_decoder_crosscheck", **row)), flush=True)
    coordinate = dict(complete=True, decision="PASS_COORDINATE_AND_FRAME_BINDING" if all(x["original_bin_redecode_byteexact"] for x in frame_rows) else "NUMERIC_DIFFERENCE_REQUIRES_REVIEW",
        sources=sources, frame_rows=frame_rows,
        columns=["ego_vehicle_x_m", "ego_vehicle_y_m", "ego_vehicle_z_m", "intensity", "elongation", "TOP_range_image_return_row_column_index_else_minus1"],
        transform="sensor extrinsic already applied; TOP per-pixel pose to world then inverse(frame.pose) to same-frame ego; worldXYZ=exportedXYZ@obs.ego_world.R.T+obs.ego_world.t",
        TOP_roundtrip_arithmetic="Original decoder uses float32 frame transforms before exporting; this audit reproduces that arithmetic. Applying the double-precision obs.ego_world is not a second sensor transform.",
        no_timestamp_reindexing=True, camera_pose_and_frame_pose_are_distinct_and_preserved=True,
        point_semantics_not_replaced_with_column6=True, within_scan_dynamic_motion_compensation_used=False,
        GPU_forward=0)
    write(out/"01_pointcloud_coordinate_and_time.json", coordinate)

    X, raw = m["world_xyz"], m["source_raw"]
    train = m["used_for_fit"]
    centre = s["fit_XY_centre"]; b = s["fit_coefficients"]
    origin = np.asarray(contract["XY_fit_context_bbox"])[0]
    selected = medoids(X, np.flatnonzero(train), origin)
    P = X[selected]; A = design(P[:, :2], centre)
    refit, w = fit(A, P[:, 2]); assert abs(refit-b).max() < 1e-12
    inv = np.linalg.pinv(A.T@(w[:, None]*A))
    train_leverage = np.einsum("ni,ij,nj->n", A, inv, A)
    svd = np.linalg.svd(A*np.sqrt(w[:, None]), compute_uv=False)
    column_norm = np.linalg.norm(A*np.sqrt(w[:, None]), axis=0)
    scaled_svd = np.linalg.svd(A*np.sqrt(w[:, None])/column_norm, compute_uv=False)
    hull = ConvexHull(P[:, :2])
    oldroot = C/"road_native_source_transfer_20261122_r5/native_source_blocks_lidar_v1"
    collector = module("strict_collector_physics_check", R10/"code/qualify_local_surface_geometry_cpu.py")
    plane_rows = []
    point_payload = {}
    for name, r, neighbours in [("P01", 110, [105,110,115]), ("P02", 137, [130,137,145])]:
        geometry_path = oldroot/name/"geometry.json"
        points_path = oldroot/name/"plane_fit_samples.npz"
        geom = read(geometry_path); old = load(points_path)["world_xyz"]
        pn = np.asarray(geom["plane_fit"]["normal_world"])
        po = np.asarray(geom["plane_fit"]["origin_world"])
        normal_distance = (old-po)@pn
        old_inlier = abs(normal_distance) <= geom["plane_fit"]["inlier_threshold_m"]
        actual = world_by_raw[r]
        nearest_distance, nearest_id = cKDTree(actual).query(old)
        exact_reloaded = actual[nearest_id]
        query = np.concatenate([old, np.asarray(geom["physical_quad_world"]), np.asarray(geom["tangent_origin_world"])[None]], axis=0)
        AQ = design(query[:, :2], centre)
        z = AQ@b
        plane_z = po[2]-((query[:, :2]-po[:2])*pn[:2]).sum(-1)/pn[2]
        margin = -(query[:, :2]@hull.equations[:, :2].T+hull.equations[:, 2]).max(-1)
        leverage = np.einsum("ni,ij,nj->n", AQ, inv, AQ)/train_leverage.max()
        dist, _ = cKDTree(P[:, :2]).query(query[:, :2])
        local_X = []; local_raw = []; local_records = []
        quad = np.asarray(geom["physical_quad_world"])
        XY = np.stack([quad[:, :2].min(0)-.4, quad[:, :2].max(0)+.4])
        for rr in neighbours:
            assert rr not in contract["heldout_raw"]
            values, ids, rec, pins, _ = collector.collect(rr, C, XY, po[2], bank)
            local_X.append(values); local_raw.append(np.full(len(values), rr, np.int16)); local_records.append(rec)
        local_X = np.concatenate(local_X); local_raw = np.concatenate(local_raw)
        local_centre = .5*(XY[0]+XY[1])
        ids = medoids(local_X, np.arange(len(local_X)), XY[0])
        local_A = design(local_X[ids, :2], local_centre)
        local_b, local_w = fit(local_A, local_X[ids, 2])
        assert local_b is not None
        local_quad_z = design(quad[:, :2], local_centre)@local_b
        local_quad_normal = normals(quad[:, :2], local_centre, local_b)
        cloud_plane_dz = local_X[:, 2]-(po[2]-((local_X[:, :2]-po[:2])*pn[:2]).sum(-1)/pn[2])
        same = local_raw == r
        row = dict(reference=name, raw=r, old_geometry=pin(geometry_path), old_fit_points=pin(points_path),
            old_plane_normal_world=pn.tolist(), old_plane_origin_world=po.tolist(),
            plane_points_reloaded_nearest_world_distance_m=stats(nearest_distance),
            old_plane_signed_point_residual_m=signed_and_abs(normal_distance),
            old_plane_inlier_count=int(old_inlier.sum()),
            old_plane_inliers_vs_current_profile_signed_dZ_m=signed_and_abs(z[:len(old)][old_inlier]-old[old_inlier,2]),
            current_tight_profile_at_old_reference_is_extrapolation=bool((margin<0).all()),
            query_outside_current_fit_hull_count=int((margin<0).sum()), query_count=len(query),
            current_profile_hull_margin_m=stats(margin),
            current_profile_leverage_over_max_training=stats(leverage), nearest_current_profile_training_medoid_m=stats(dist),
            independent_local_neighbourhood_diagnostic=dict(raws=neighbours,
                source_filter="same original ego/OBB/photo-road0 erode2; .4m XY context around old physical quad; broadZ planeorigin +/-1.5m; no profile residual trim",
                points=len(local_X), medoids=len(ids), frame_records=local_records,
                all_point_signed_height_from_old_plane_m=signed_and_abs(cloud_plane_dz),
                same_frame_signed_height_from_old_plane_m=signed_and_abs(cloud_plane_dz[same]),
                per_frame_signed_height_from_old_plane_m={str(rr):signed_and_abs(cloud_plane_dz[local_raw==rr]) for rr in neighbours},
                local_quadratic_at_old_quad_signed_height_from_old_plane_m=signed_and_abs(local_quad_z-quad[:,2]),
                local_quadratic_normal_vs_old_plane_deg=stats(angle(local_quad_normal, pn)),
                design_condition=float(np.linalg.cond(local_A*np.sqrt(local_w[:,None]))),
                local_fit_not_added_to_strict_geometry_or_activation=True),
            old_reference_is_independent_method_not_independent_raw_coordinate_implementation=True)
        plane_rows.append(row)
        point_payload.update({name+"_old_points":old, name+"_reloaded_original_points":exact_reloaded,
            name+"_profile_dZ_vs_old_plane":z-plane_z, name+"_query_world_XY":query[:,:2],
            name+"_local_diagnostic_worldXYZ":local_X, name+"_local_diagnostic_raw":local_raw,
            name+"_local_quadratic_coefficients":local_b, name+"_local_quadratic_XY_centre":local_centre})
        print(json.dumps(dict(stage="old_physical_reference_crosscheck", **row)), flush=True)
    write(out/"02_P01_P02_independent_physical_references.json", dict(complete=True, rows=plane_rows,
        no_heldout_points_fitted=True, formal_geometry_and_coefficients_unchanged=True))
    np.savez_compressed(out/"physical_reference_points.npz", **point_payload)

    grid_path = C/"road_shared_appearance_20261122_r7/shared_grid_geometry.npz"
    g = load(grid_path)
    u0,v0,u1,v1 = s["bounds_uvxy"].astype(int)
    xyz = g["world_xyz"][v0:v1,u0:u1]
    valid = g["geometry_valid"][v0:v1,u0:u1].copy()
    stored = g["normal_world"][v0:v1,u0:u1]
    boundary = valid & binary_dilation(~valid)
    profile_n = normals(xyz[..., :2].reshape(-1,2), centre, b).reshape(xyz.shape)
    profile_z = (design(xyz[..., :2].reshape(-1,2), centre)@b).reshape(valid.shape)
    du = np.zeros_like(xyz); dv = np.zeros_like(xyz)
    du_ok = np.zeros_like(valid); dv_ok = np.zeros_like(valid)
    # Prefer central differences; one-sided derivatives still require actual valid vertices.
    for axis, deriv, ok in [(1,du,du_ok),(0,dv,dv_ok)]:
        for y,x in zip(*np.nonzero(valid)):
            lo = (y,x-1) if axis==1 else (y-1,x)
            hi = (y,x+1) if axis==1 else (y+1,x)
            haslo = min(lo)>=0 and valid[lo]
            hashi = hi[0]<valid.shape[0] and hi[1]<valid.shape[1] and valid[hi]
            if haslo and hashi:
                deriv[y,x]=xyz[hi]-xyz[lo];ok[y,x]=True
            elif hashi:
                deriv[y,x]=xyz[hi]-xyz[y,x];ok[y,x]=True
            elif haslo:
                deriv[y,x]=xyz[y,x]-xyz[lo];ok[y,x]=True
    fd = np.cross(du,dv); fd_norm=np.linalg.norm(fd,axis=-1)
    fd_ok=valid&du_ok&dv_ok&(fd_norm>1e-12)
    fd=fd/np.maximum(fd_norm[...,None],1e-15);fd=np.where(fd[...,2:]<0,-fd,fd)
    fd_profile=angle(fd,profile_n);fd_stored=angle(fd,stored)
    H,W=valid.shape;ids=np.arange(H*W).reshape(H,W)
    tri=np.concatenate([np.stack([ids[:-1,:-1],ids[:-1,1:],ids[1:,1:]],-1).reshape(-1,3),
                        np.stack([ids[:-1,:-1],ids[1:,1:],ids[1:,:-1]],-1).reshape(-1,3)])
    tri_good=valid.ravel()[tri].all(-1)
    vertices=xyz.reshape(-1,3)[tri]
    tn=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]);area=np.linalg.norm(tn,axis=-1)
    tri_good&=area>1e-12;tn/=np.maximum(area[:,None],1e-15);tn=np.where(tn[:,2:]<0,-tn,tn)
    tri_centres=vertices.mean(1);qn=normals(tri_centres[:,:2],centre,b)
    sn=stored.reshape(-1,3)[tri].mean(1)
    tri_profile=angle(tn,qn);tri_stored=angle(tn,sn)
    tri_boundary=tri_good&boundary.ravel()[tri].any(-1)
    parent = dict(complete=True, parent_geometry=pin(grid_path),
        signed_delta_definition="new_profile_Z minus actual_parent_worldXYZ_Z; normal angles are unsigned orientation-invariant",
        actual_parent_Z_vs_profile={"all_supported":signed_and_abs(profile_z[valid]-xyz[...,2][valid]),
                                  "missing_support_boundary":signed_and_abs(profile_z[boundary]-xyz[...,2][boundary])},
        stored_normal_vs_profile_deg={"all_supported":stats(angle(stored,profile_n)[valid]),
                                     "missing_support_boundary":stats(angle(stored,profile_n)[boundary])},
        actual_worldXYZ_finite_difference_normals=dict(
            construction="central if both valid neighbours, else one-sided; require real support in both U and V; no invalid-Z interpolation",
            valid_vertices=int(fd_ok.sum()), boundary_valid_vertices=int((fd_ok&boundary).sum()),
            normal_vs_profile_deg={"all_supported":stats(fd_profile[fd_ok]), "boundary":stats(fd_profile[fd_ok&boundary])},
            normal_vs_stored_deg={"all_supported":stats(fd_stored[fd_ok]), "boundary":stats(fd_stored[fd_ok&boundary])}),
        actual_worldXYZ_triangle_normals=dict(construction="same diagonal as actual local surface projector; all3 corners geometry_valid; nonzero area",
            valid_triangles=int(tri_good.sum()), boundary_adjacent_valid_triangles=int(tri_boundary.sum()),
            normal_vs_profile_deg={"all_supported":stats(tri_profile[tri_good]),"boundary":stats(tri_profile[tri_boundary])},
            normal_vs_interpolated_stored_deg={"all_supported":stats(tri_stored[tri_good]),"boundary":stats(tri_stored[tri_boundary])}),
        physical_truth_not_inferred_from_stored_gaussian_normals=True,
        profile_weighted_design_condition=float(svd[0]/svd[-1]),
        profile_column_normalized_weighted_design_condition=float(scaled_svd[0]/scaled_svd[-1]),
        profile_singular_values=svd.tolist(),
        supported_point_distribution_by_train_frame={str(rr):dict(points=int(((raw==rr)&train).sum()),
            medoids=int((raw[selected]==rr).sum()), xyz_min=X[(raw==rr)&train].min(0).tolist(),
            xyz_max=X[(raw==rr)&train].max(0).tolist(),
            profile_signed_height_residual_m=signed_and_abs(design(X[(raw==rr)&train,:2],centre)@b-X[(raw==rr)&train,2])) for rr in contract["train_raw"]})
    write(out/"03_parent_actual_geometry_normals.json", parent)
    np.savez_compressed(out/"parent_actual_geometry_normal_maps.npz", uv_global=s["uv_global"],geometry_valid=valid,
        parent_boundary=boundary,profile_minus_parent_Z_m=profile_z-xyz[...,2],
        normal_stored=stored,normal_finite_difference=fd,finite_difference_normal_valid=fd_ok,
        normal_profile=profile_n,triangles=tri,triangle_geometry_valid=tri_good,triangle_boundary=tri_boundary,
        triangle_geometric_normal=tn,triangle_profile_normal=qn,
        finite_difference_vs_profile_deg=fd_profile,finite_difference_vs_stored_deg=fd_stored,
        triangle_vs_profile_deg=tri_profile,triangle_vs_stored_deg=tri_stored)
    result=dict(schema="R10.local_surface_physical_crosscheck.v1",complete=True,
        decision="READ_ONLY_CROSSCHECK_NO_GEOMETRY_ACTIVATION", coordinate_time=pin(out/"01_pointcloud_coordinate_and_time.json"),
        old_physical_reference=pin(out/"02_P01_P02_independent_physical_references.json"),
        actual_parent_geometric_normals=pin(out/"03_parent_actual_geometry_normals.json"),
        original_strict_certificate=pin(strict/"certificate.json"),
        current_profile_coefficients_unchanged=True,parent_geometry_byteexact_unchanged=True,
        new_inferred_renderable_texels=0,new_GPU_forward=0,updates=0,elapsed_seconds=time.time()-start)
    write(out/"crosscheck_receipt.json", result)
    print(json.dumps(dict(complete=True,receipt=pin(out/"crosscheck_receipt.json"))),flush=True)


if __name__=="__main__":
    main()

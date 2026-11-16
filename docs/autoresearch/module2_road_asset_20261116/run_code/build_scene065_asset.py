"""现有065合法输入到持久道路资产；不训练、不读取隐藏答案。"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from PIL import Image

REPO = Path("/root/autodl-tmp/motion_proj_v82_gn11_v42")
CAMPAIGN = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
OLD = CAMPAIGN / "scene065_material_loop_20261114_r1"
LOOP = CAMPAIGN / "scene065_closedloop_20261116_r1"
sys.path.insert(0, str(REPO / "scripts/worldsim_v82"))
from drivingeditor_dynamic_edit.road_asset import (build_observation_priority_atlas,
    repair_road_coverage, interpolate_legal_surface, infer_measured_plane_surface, WorldRoadMaterialAtlas)


def read(path):
    return json.loads(Path(path).read_text())


def load(path, fields=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in (z.files if fields is None else fields)}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def frozen_generation(world, normal, requested, cfg):
    source_dir = Path(cfg["material_dataset"]).parents[2] / "source"
    sys.path.insert(0, str(source_dir))
    from world_feature_field import camera_from_observation, project_world_to_camera, bilinear_sample
    frozen = read(CAMPAIGN / "road_mask_decoupling_20261116_r1/output_guard_probe/control_freeze.json")
    cases = {case["id"]: case for case in frozen["cases"]}
    n = len(world)
    rgb = np.full((n, 3), .5, np.float32)
    best = np.full(n, -1, np.int16)
    best_uv = np.full((n, 2), np.nan, np.float32)
    best_score = np.full(n, -1., np.float64)
    legal = np.zeros(n, bool)
    sources = []
    for raw in [86, 110]:
        case = cases[f"scene065_raw{raw:03d}"]
        observation = Path(cfg["bank"]) / "observations" / f"{raw:03d}.json"
        camera = camera_from_observation(read(observation))
        roles = load(case["native_guard_roles"]["path"], ["W_guard", "K", "H_B"])
        guard = roles["W_guard"] & ~roles["K"]
        uv, depth, inside = project_world_to_camera(world, **{k: camera[k] for k in
            ["camera_K_native", "world_to_camera", "image_hw", "distortion_Brown"]})
        origin = np.linalg.inv(np.asarray(camera["world_to_camera"]))[:3, 3]
        direction = origin - world
        distance = np.linalg.norm(direction, axis=1)
        cosine = np.maximum(np.sum(direction * normal, axis=1) / np.maximum(distance, 1e-6), 0)
        ids = np.flatnonzero(requested & inside & (distance < 60) & (cosine > .025))
        lo = np.floor(uv[ids] - .5).astype(np.int64)
        hi = lo + 1
        x, y = lo.T
        x1, y1 = hi.T
        inside_guard = guard[y, x] & guard[y, x1] & guard[y1, x] & guard[y1, x1]
        ids = ids[inside_guard]
        legal[ids] = True
        score = cosine[ids] / (distance[ids] ** 2 + 4)
        selected = ids[score > best_score[ids]]
        native = np.asarray(Image.open(case["source_raw_B"]["path"]).convert("RGB"), np.float32) / 255
        rgb[selected] = bilinear_sample(native, uv[selected], camera["image_hw"])
        best[selected] = raw
        best_uv[selected] = uv[selected]
        best_score[selected] = cosine[selected] / (distance[selected] ** 2 + 4)
        sources.append(dict(raw=raw, guard_path=case["native_guard_roles"]["path"],
            image_path=case["source_raw_B"]["path"], observation_path=str(observation), legal_rows=int(len(ids))))
    return dict(rgb=rgb, source_raw=best, source_uv_native=best_uv, legal=legal, sources=sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-interpolation", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    cfg = read(OLD / "config.json")
    source_manifest = LOOP / "handoff_coverage_r2/persistent/scene.json"
    manifest = read(source_manifest)
    base = load(manifest["road"])
    geom = load(cfg["geometry_reference"], ["world_xyz", "normal_world", "keys", "real", "active",
                                          "best_source_raw", "reference_srgb", "cell_m"])
    field = load(OLD / "field_inputs/world_field_input.npz")
    world = geom["world_xyz"].copy()
    normal = geom["normal_world"].copy()
    requested = field["requested"]
    original_support = field["surface_supported"]
    support = original_support.copy()
    known = field["known"] | geom["real"]
    measured = field["context_rgb"].copy()
    measured[geom["real"]] = geom["reference_srgb"][geom["real"]]
    protected = base.get("minimum_model_depth", np.zeros(len(world))) > 0
    # 既有活跃几何与合法实测节点均固定，新表面只能使用未活跃占位格。
    trusted = base["active"] | known
    measured_geometry_path = Path(cfg["geometry_reference"]).parents[2] / "geometry_no_colour/geometry.npz"
    measured_geometry = load(measured_geometry_path, ["world_xyz", "plane", "coordinate_anchor"])
    measured_plane_receipt = read(measured_geometry_path.parent / "receipt.json")
    local_normal = np.r_[-measured_geometry["plane"][:2], 1.]
    plane_normal = measured_geometry["coordinate_anchor"][:3, :3] @ local_normal
    plane_normal /= np.linalg.norm(plane_normal)
    normal_unit = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    plane_residual = np.abs(np.sum((world - measured_geometry["world_xyz"]) * plane_normal, axis=1))
    # 全体IDW节点并非等精度锚点。独立测距平面MAE约1.66cm，先筛选与其
    # 位置及法向一致、且经道路观测认证的既有节点，再局部包围插值。
    anchor_mask = support & known & ~protected & (plane_residual <= .05)
    anchor_mask &= np.abs(normal_unit @ plane_normal) >= .98
    from measurement_road_anchors import collect_road_measurements
    geometry_config = read(measured_geometry_path.parents[1] / "geometry_config_no_colour.json")
    measurement_world, measurement_raw, measurement_receipt = collect_road_measurements(geometry_config, measured_geometry)
    np.savez_compressed(args.output / "measurement_anchors.npz", world_xyz=measurement_world, source_raw=measurement_raw,
        normal_world=np.tile(plane_normal, (len(measurement_world), 1)))
    first = frozen_generation(world, normal, requested, cfg)
    candidate = requested & first["legal"] & ~support & ~trusted & ~protected
    if args.no_interpolation:
        candidate[:] = False
    # 直接测距点绕开旧IDW节点的法向/高度噪声。插值锚点来自独立道路测距，
    # 不是由要补的图像、旧车辆残影或生成结果反推的道路形状。
    world, normal, interpolation, plane_support_receipt = infer_measured_plane_surface(world, normal,
        candidate, measurement_world, plane_normal, float(measured_geometry["world_xyz"][0] @ plane_normal))
    added_surface = interpolation["surface_interpolated"]
    diagnostics = {key: dict(quantiles=np.nanquantile(value[candidate], [0, .1, .5, .9, 1]).tolist())
        for key, value in interpolation.items() if key.startswith("interpolation_candidate_")}
    print(json.dumps(dict(interpolation_diagnostics=diagnostics)), flush=True)
    support |= added_surface
    adjusted = {k: v.copy() for k, v in base.items()}
    transform = base["model_from_world"]
    adjusted["model_xyz"][added_surface] = (world @ transform[:3, :3].T + transform[:3, 3])[added_surface]
    adjusted["normal_world"][added_surface] = normal[added_surface]
    generated = frozen_generation(world, normal, requested, cfg)
    # 前景冲突旧行仅保留；不把旧颜色冒充一个新观测/生成来源。
    legal = known | ((support & (base["active"] | requested)) & ~protected)
    observed_raw = field["best_source_raw"].copy()
    use_real = geom["real"] & (observed_raw < 0)
    observed_raw[use_real] = geom["best_source_raw"][use_real]
    atlas, atlas_receipt = build_observation_priority_atlas(world, geom["keys"], measured, known, legal,
        generated_rgb=generated["rgb"], generated_mask=generated["legal"] & requested & support,
        observed_source_raw=observed_raw, generated_source_raw=generated["source_raw"])
    sampled = atlas.query(world)
    if not np.array_equal(sampled["rgb_srgb"][known], measured[known]):
        raise RuntimeError("世界XY查询未逐值锁定实测节点")
    if np.any(sampled["unknown"][known]):
        raise RuntimeError("合法实测节点被误判为unknown")
    footprint_trusted = geom["active"] | known
    target, coverage, coverage_receipt = repair_road_coverage(adjusted, sampled, requested,
        support, footprint_trusted, protected_geometry=protected, cell_m=float(geom["cell_m"]), world_keys=geom["keys"])
    material_update = ~sampled["unknown"] & ~protected & requested
    target["reference_srgb"][material_update] = sampled["rgb_srgb"][material_update]
    # 新表面必须重新落在同一冻结写回许可内；不满足时保留inactive。
    bad_interpolation = added_surface & ~generated["legal"]
    target["active"][bad_interpolation] = False
    if not np.array_equal(target["model_xyz"][trusted], base["model_xyz"][trusted]):
        raise RuntimeError("固定可信位置被改变")
    if not np.array_equal(target["normal_world"][trusted], base["normal_world"][trusted]):
        raise RuntimeError("固定可信法向被改变")
    atlas.save(args.output / "atlas.npz")
    reloaded = WorldRoadMaterialAtlas.load(args.output / "atlas.npz")
    reload_query = reloaded.query(world[::103])
    if not np.array_equal(reload_query["rgb_srgb"], sampled["rgb_srgb"][::103]):
        raise RuntimeError("持久图集重载结果不一致")
    np.savez_compressed(args.output / "road.npz", **target)
    row_source = sampled["source"]
    gen = row_source == 3
    provenance = dict(world_xyz=world, world_keys=geom["keys"], atlas_uv=sampled["uv"],
        row_rgb=sampled["rgb_srgb"], source=row_source, confidence=sampled["confidence"],
        unknown=sampled["unknown"], known=known, observed_reference_srgb=measured,
        source_world_row=sampled["source_world_row"], source_distance_m=sampled["source_distance_m"],
        source_raw=sampled["source_raw"], source_uv_native=np.where(gen[:, None], generated["source_uv_native"], np.nan),
        legal_write=legal, material_updated=material_update, preserved_legacy_rgb=~material_update,
        effective_row_rgb=target["reference_srgb"], effective_active=target["active"],
        effective_model_xyz=target["model_xyz"], effective_sigma_m=target["gaussian_sigma_m"],
        W_world=generated["legal"], requested=requested, surface_supported=support,
        original_surface_supported=original_support, original_active=base["active"],
        original_world_xyz=geom["world_xyz"], **coverage, **interpolation)
    np.savez_compressed(args.output / "provenance.npz", **provenance)
    ids = atlas.grid_world_row
    Image.fromarray(np.uint8(np.clip(atlas.rgb_srgb, 0, 1) * 255 + .5)).save(args.output / "atlas.png")
    palette = np.array([[7, 12, 20], [45, 180, 110], [62, 123, 215], [220, 155, 50]], np.uint8)
    Image.fromarray(palette[atlas.source]).save(args.output / "source.png")
    gcolor = np.zeros((*ids.shape, 3), np.uint8)
    gcolor[original_support[ids]] = [38, 110, 78]
    gcolor[base["active"][ids]] = [60, 165, 110]
    gcolor[added_surface[ids]] = [230, 165, 45]
    gcolor[(requested & ~support)[ids]] = [190, 55, 60]
    Image.fromarray(gcolor).save(args.output / "coverage.png")
    keyframes = []
    for raw in [86, 110, 134, 179, 198]:
        view = load(Path(cfg["road_views"]) / f"{raw:03d}.npz")
        domain = view["target"] & view["inside"] & ~view["retained_reveal_guard"] & ~view["retained_box_guard"]
        ix = view["index"][domain]
        keyframes.append(dict(raw=raw, target_chart_pixels=int(len(ix)),
            old_active_fraction=float(base["active"][ix].mean()), new_active_fraction=float(target["active"][ix].mean()),
            unsupported_fraction=float((~support[ix]).mean()), interpolated_fraction=float(added_surface[ix].mean()),
            material_unknown_fraction=float(sampled["unknown"][ix].mean()),
            bounded_near_fraction=float(protected[ix].mean()), footprint_repaired_fraction=float(coverage["coverage_repaired"][ix].mean())))
    # 独立标线层只发布实测颜色判据，不把检索或生成线条认证成道路结构。
    lum = measured @ np.array([.2126, .7152, .0722], np.float32)
    white = known & (lum > .5) & (np.ptp(measured, axis=1) < .15)
    yellow = known & (measured[:, 0] - measured[:, 2] > .07) & (measured[:, 1] - measured[:, 2] > .05) & (lum > .25)
    np.savez_compressed(args.output / "markings.npz", world_xyz=world,
        observed_white=white, observed_yellow=yellow, confidence=(white | yellow).astype(np.float32),
        phase_known=np.zeros(len(world), bool), inferred_markings=np.zeros(len(world), bool))
    receipt = dict(schema="scene065_persistent_road_asset.v1", completed=True,
        source_manifest=str(source_manifest), source_road=manifest["road"], geometry_reference=cfg["geometry_reference"],
        world_field_input=str(OLD / "field_inputs/world_field_input.npz"), output_road=str(args.output / "road.npz"),
        atlas=atlas_receipt, coverage=coverage_receipt, frozen_generation_sources=generated["sources"],
        interpolation=dict(candidate_nodes=int(candidate.sum()), diagnostics=diagnostics,
            original_active_and_measured_geometry_fixed=True, **plane_support_receipt),
        anchor_selection=dict(nodes=int(anchor_mask.sum()), measured_geometry=str(measured_geometry_path),
            measured_points=measured_plane_receipt["measured_points"], plane_inliers=measured_plane_receipt["plane_inliers"],
            plane_mae_m=measured_plane_receipt["plane_mae_m"], maximum_original_point_plane_residual_m=.05,
            minimum_normal_cosine_to_measured_plane=.98, visible_road_observation_required=True,
            actual_interpolation_anchor_space="measurement_anchors.npz world_xyz rows",
            actual_interpolation_measurements=measurement_receipt),
        known_nodes=int(known.sum()), original_active_nodes=int(base["active"].sum()), active_nodes=int(target["active"].sum()),
        material_updated_nodes=int(material_update.sum()), preserved_legacy_rgb_nodes=int((~material_update).sum()),
        known_atlas_rgb_bitexact=True, trusted_geometry_bitexact=True, no_diffusion_or_training_run=True,
        runtime_material_write_domain="requested & valid sampled material & ~persistent foreground conflict rows; full-domain atlas build retained",
        original_minimum_model_depth_unchanged=bool(np.array_equal(target["minimum_model_depth"], base["minimum_model_depth"])),
        marking_semantics="only observed white/yellow candidates; no forced segment continuation or dash phase",
        raw_pilots=keyframes, reload_verified=True, elapsed_seconds=time.monotonic() - start,
        limitations=["unobserved retrieval/generated appearance is not hidden road truth",
            "unverified road geometry remains unsupported", "joint occlusion and static retirement belong to module4"],
        human_verdict=None)
    write(args.output / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

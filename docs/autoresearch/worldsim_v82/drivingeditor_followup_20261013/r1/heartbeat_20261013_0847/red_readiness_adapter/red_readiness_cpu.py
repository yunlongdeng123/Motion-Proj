#!/usr/bin/env python3
"""红车实际 CPU reader/SGCM 适配。默认 check，无训练入口、无输入覆盖。"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import random
import struct
import sys

sys.dont_write_bytecode = True
BASE = Path("/root/autodl-tmp/drivingeditor_repro")
ROOT = BASE / "campaign/followup_scene_prep_20261013"
MATRIX = BASE / "campaign/hybrid30k_20261012"
SCENE_ID = "scene0029_069_117"
FRAME_IDS = ["0029%03d" % i for i in range(69, 118)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def same(a, b):
    return Path(a).resolve() == Path(b).resolve()


def checked_file(path):
    require(path is not None and Path(path).is_file() and Path(path).stat().st_size > 0,
            "缺少真实输入或回执: " + str(path))
    return Path(path)


def png_wh(path):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    require(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", "PNG header无效: " + str(path))
    return list(struct.unpack(">II", header[16:24]))


def ply_vertices(path):
    with checked_file(path).open("rb") as stream:
        require(stream.readline().strip() == b"ply", "真实PLY header缺失: " + str(path))
        for _ in range(100):
            line = stream.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            require(line != "end_header" and bool(line), "真实PLY vertex count缺失: " + str(path))
    raise ValueError("PLY header超出预期: " + str(path))


def import_module(path, name):
    specification = importlib.util.spec_from_file_location(name, checked_file(path))
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def input_frames(scene, directory, suffix, image=False):
    folder = scene / directory
    require(sorted(p.stem for p in folder.glob("*" + suffix)) == FRAME_IDS,
            "缺少真实49帧或包含未知帧: " + str(folder))
    for name in FRAME_IDS:
        path = checked_file(folder / (name + suffix))
        if image:
            require(png_wh(path) == [1920, 1280], "真实FRONT尺寸不匹配: " + str(path))


def check(args):
    """只用stdlib检查文件与既有回执；不导入Torch/TF、不写文件。"""
    require(__debug__, "继承的assert不能被python -O关闭")
    scene = args.scene.resolve()
    require(same(scene, ROOT / "inputs/scene0029_069_117_red_cpu_v1"), "当前授权仅此实际红车输入")
    prep = read(checked_file(scene / "preparation_manifest.json"))
    rows = prep.get("frames", [])
    require(prep.get("scene_id") == SCENE_ID and [r.get("id") for r in rows] == FRAME_IDS,
            "真实scene与49 source frame映射不匹配")
    require(prep.get("raw_training_range") == [69, 117] and
            [r.get("source_index") for r in rows] == list(range(69, 118)), "raw69..117映射缺失")
    require(all(abs(r["model_time"] - (-.48 + .02*i)) < 1e-12 for i, r in enumerate(rows)),
            "必须继承49 model times[-.48,.48]/.02，不冒称真实秒")
    require(prep.get("fixed_pose_anchor_for_later_training") == "0029079" and
            prep.get("model_frame_interval") == .02 and prep.get("num_points") == 600000 and
            prep.get("seed") == 0 and prep.get("training_ready") is False,
            "真实anchor/原600k混合static seed/.02合同改变")
    require(prep.get("GT_labels_used_to_filter_training_inputs") is False and
            prep.get("m_detector_used") is False and prep.get("shadow_supervision") == "absent",
            "输入角色未明确保持原始约束")
    missing = []
    for label, path in (("root visually_reviewed selection", args.selection),
                        ("real sky/DEVA execution receipt", args.execution_receipt),
                        ("root actor build receipt", scene / "dynamic_multi_actor_preparation.json"),
                        ("root PASS_ACTOR_CPU_ONLY validation", scene / "dynamic_multi_actor_validation.json"),
                        ("original static source indices", args.static_source_indices)):
        if path is None or not Path(path).is_file():
            missing.append({"input": label, "path": str(path) if path is not None else None})
    for directory in ("dynamic_masks/dynamic_mask_0", "dynamic_image/dynamic_image_0"):
        if not (scene / directory).is_dir():
            missing.append({"input": "root real49 " + directory, "path": str(scene / directory)})
    if missing:
        return {"status": "BLOCKED_REAL_ACTOR_INPUTS", "scene_id": SCENE_ID, "missing": missing,
                "GPU_used": False, "Torch_imported": False, "TF_imported": False,
                "training_started": False, "training_ready": False, "output_written": False}
    selection = read(args.selection)
    actors = selection.get("actors", [])
    require(selection.get("status") == "visually_reviewed" and same(selection.get("scene", ""), scene),
            "只能接受root真实视觉审查后的selection，不猜actor ID")
    require(actors and [a["actor_index"] for a in actors] == list(range(len(actors))), "实际actor数/连续索引未确认")
    for actor in actors:
        require(actor.get("description") and actor.get("raw_track_id") and
                [r.get("frame_id") for r in actor.get("frames", [])] == FRAME_IDS,
                "每个真实actor必须有49帧身份与逐帧审查")
        require(all(r.get("reviewed") is True and str(r.get("review_note") or "").strip()
                    and r.get("visibility") in ("visible", "not_visible") for r in actor["frames"]),
                "actor逐帧真实body审查尚未完成")
    execution = read(args.execution_receipt)
    require(execution.get("status") == "REAL_RED49_SKY_AND_DEVA_VALIDATED_ACTOR_SELECTION_PENDING"
            and execution.get("scene_id") == SCENE_ID and execution.get("sky_frames") == 49
            and execution.get("training_ready") is False, "真实49 sky/DEVA terminal回执未通过")
    actor_build = read(scene / "dynamic_multi_actor_preparation.json")
    actor_validation = read(scene / "dynamic_multi_actor_validation.json")
    require(same(actor_build.get("scene", ""), scene) and
            same(actor_build.get("selection_path", ""), args.selection) and
            same(actor_build.get("execution_receipt", ""), args.execution_receipt), "actor构建来源绑定改变")
    require(actor_build.get("gpu_used") is False and actor_build.get("training_ready") is False and
            actor_build.get("static_seed_modified") is False and
            actor_build.get("gt_3d_boxes_used_to_filter_training_points") is False and
            actor_build.get("m_detector_used") is False and actor_build.get("shadow_supervision") == "absent",
            "actor构建必须保持CPU/原static/无GT筛点/无M-detector/无shadow")
    require(actor_validation.get("status") == "PASS_ACTOR_CPU_ONLY" and
            actor_validation.get("frames") == 49 and actor_validation.get("actor_count") == len(actors)
            and actor_validation.get("gpu_used") is False and actor_validation.get("training_ready") is False,
            "真实body/prior CPU validation未通过")
    for key in ("all_ply_vertices_exact_real_sensor_subset", "all_temporal_properties_exact", "binary_union_and_rgb_exact"):
        require(actor_validation.get(key) is True, "actor精确合同未通过: " + key)
    for directory, suffix in (("image/image_0", ".png"), ("sky/sky_0", ".png"),
                              ("mask/mask_0/Annotations", ".png"),
                              ("dynamic_masks/dynamic_mask_0", ".png"),
                              ("dynamic_image/dynamic_image_0", ".png"),
                              ("pose", ".txt"), ("calib", ".txt"), ("velodyne", ".bin")):
        input_frames(scene, directory, suffix, image=suffix == ".png")
    require(not list((scene / "dynamic_masks").glob("shadow*")), "当前真实shadow缺失，不能静默union未知shadow")
    priors = [scene / ("points3d_dynamic_%d.ply" % i) for i in range(len(actors))]
    require(sorted(scene.glob("points3d_dynamic_*.ply")) == priors, "实际actor先验集合不是连续编号")
    counts = [ply_vertices(p) for p in priors]
    require(all(n > 0 for n in counts) and counts == actor_validation.get("real_points"), "真实prior数与验证回执不一致")
    require(ply_vertices(scene / "points3d.ply") == 600000, "不得重采样或删改原600k混合static种子")
    for actor in actors:
        folder = "dynamic_actor_assets/actor_%d" % actor["actor_index"]
        input_frames(scene, folder + "/body_masks", ".png", image=True)
        checked_file(scene / folder / "source_lidar_indices.npz")
    checked_file(args.cpu_template)
    checked_file(args.followup_tools / "contracts.py")
    checked_file(args.followup_tools / "run_scene_gn11.py")
    checked_file(args.followup_tools / "prepare_followup.py")
    reference = read(checked_file(args.reference_config))
    sys.path.insert(0, str(args.followup_tools))
    from contracts import validate_gn11_config
    from prepare_followup import source_contract
    validate_gn11_config(reference)
    source_contract(args.repository)
    require(reference.get("num_pts") == 600000 and reference.get("frame_interval") == .02 and
            reference.get("require_boundingbox") is False and reference.get("initialization") is False,
            "原point数/time/无legacy双actor分支的reader合同改变")
    return {"status": "READY_REAL_RED_CPU_READER_PLAN", "scene_id": SCENE_ID, "frames": 49,
            "actor_count": len(actors), "actor_PLY_vertices": counts, "static_seed_vertices": 600000,
            "fixed_pose_anchor": "0029079", "SGCM_ground_anchor": "0029069",
            "GPU_used": False, "Torch_imported": False, "TF_imported": False,
            "training_started": False, "training_ready": False, "output_written": False}


def verify_real_source_assets(scene, prep, selection, static_index_file):
    """核对root真实body与原sensor source indices；不重做分割/深度/连通域，不使用GT框。"""
    import numpy as np
    from PIL import Image
    from plyfile import PlyData

    def vertices(path):
        return PlyData.read(str(path))["vertex"]

    actors = selection["actors"]
    clouds = [vertices(scene / ("points3d_dynamic_%d.ply" % i)) for i in range(len(actors))]
    indices = [np.load(scene / ("dynamic_actor_assets/actor_%d/source_lidar_indices.npz" % i), allow_pickle=False)
               for i in range(len(actors))]
    with np.load(static_index_file, allow_pickle=False) as seed:
        selected = seed["concatenated_real_point_indices"].copy()
    total = sum(r["lidar_points"] for r in prep["frames"])
    require(selected.dtype == np.int32 and selected.shape == (600000,) and
            np.array_equal(selected, np.random.RandomState(0).choice(total, 600000, replace=True)),
            "原static seed0/600k source-index顺序改变")
    static = vertices(scene / "points3d.ply")
    static_xyz = np.column_stack([static[k] for k in ("x", "y", "z")])
    offsets, raw_offset, counts = [0] * len(actors), 0, []
    try:
        for frame_index, row in enumerate(prep["frames"]):
            name = row["id"]
            rgb = np.asarray(Image.open(scene / "image/image_0" / (name + ".png")).convert("RGB"))
            union = np.zeros((1280, 1920), dtype=bool)
            lidar = np.fromfile(scene / "velodyne" / (name + ".bin"), dtype=np.float32).reshape(-1, 6)
            pose = np.loadtxt(scene / "pose" / (name + ".txt"))
            require(len(lidar) == row["lidar_points"] and np.isfinite(lidar).all() and
                    pose.shape == (4, 4) and np.isfinite(pose).all(), "原sensor/pose缺失或改变: " + name)
            chosen_seed = np.flatnonzero((selected >= raw_offset) & (selected < raw_offset + len(lidar)))
            world_seed = (np.column_stack((lidar[selected[chosen_seed] - raw_offset, :3],
                                          np.ones(len(chosen_seed)))) @ pose.T)[:, :3].astype(np.float32)
            require(np.array_equal(static_xyz[chosen_seed], world_seed), "原混合static world XYZ被改变: " + name)
            for key in ("nx", "ny", "nz", "time"):
                require(np.all(static[key][chosen_seed] == np.float32(row["model_time"])), "原static model time改变")
            raw_offset += len(lidar)
            for actor in actors:
                i = actor["actor_index"]
                body = np.asarray(Image.open(scene / ("dynamic_actor_assets/actor_%d/body_masks" % i) / (name + ".png")))
                require(body.shape == union.shape and set(np.unique(body)).issubset({0, 255}), "真实body必须二值原尺寸")
                foreground = body > 0
                require(bool(foreground.any()) == (actor["frames"][frame_index]["visibility"] == "visible")
                        and not np.any(union & foreground), "body与root审查visibility或actor分离不一致")
                union |= foreground
                source = indices[i][name]
                require(source.dtype == np.int32 and source.ndim == 1 and np.all(source >= 0)
                        and np.all(source < len(lidar)) and len(np.unique(source)) == len(source), "真实actor source indices无效")
                world = (np.column_stack((lidar[source, :3].astype(np.float64), np.ones(len(source)))) @ pose.T)[:, :3].astype(np.float32)
                data = clouds[i][offsets[i]:offsets[i] + len(source)]
                offsets[i] += len(source)
                require(np.array_equal(np.column_stack([data[k] for k in ("x", "y", "z")]), world), "prior必须精确来自原sensor world pose")
                require(all(np.all(data[k] == np.float32(row["model_time"])) for k in ("nx", "ny", "nz", "time")), "prior .02 model times改变")
            canonical = np.asarray(Image.open(scene / "dynamic_masks/dynamic_mask_0" / (name + ".png")))
            dynamic = np.asarray(Image.open(scene / "dynamic_image/dynamic_image_0" / (name + ".png")).convert("RGB"))
            require(set(np.unique(canonical)).issubset({0, 255}) and np.array_equal(canonical > 0, union)
                    and np.array_equal(dynamic, np.where(union[..., None], rgb, 0)), "root完整body union/dynamic RGB不精确")
            counts.append({"frame_id": name, "full_resolution_body_pixels": int(union.sum())})
        require(all(offsets[i] == len(clouds[i]) and offsets[i] > 0 for i in range(len(actors))), "prior source-index行数不匹配")
    finally:
        for archive in indices:
            archive.close()
    return {"static_seed_vertices": 600000, "static_seed0_source_indices_exact": True,
            "static_world_xyz_and_model_times_exact": True, "actor_world_xyz_and_model_times_exact": True,
            "canonical_full_body_union_and_dynamic_rgb_exact": True, "actor_real_points": offsets, "frames": counts}


def build(args):
    plan = check(args)
    require(plan["status"] == "READY_REAL_RED_CPU_READER_PLAN", json.dumps(plan, ensure_ascii=False))
    from contracts import below, build_config, input_inventory, readiness_contract, scene_spec, write_new
    scene = args.scene.resolve()
    output = below(args.output_dir, ROOT / "metadata")
    require(not output.exists(), "拒绝覆盖已有readiness输出: " + str(output))
    prep, selection = read(scene / "preparation_manifest.json"), read(args.selection)
    source_validation = verify_real_source_assets(scene, prep, selection, args.static_source_indices)
    sys.path.insert(0, str(args.cpu_template.parent))
    template = import_module(args.cpu_template, "red_private_cpu_template")
    torch, loader, sgcm, graphics, masks, helper = template.import_cpu_reader(args.repository)
    import numpy as np
    from omegaconf import OmegaConf

    reference = read(args.reference_config)
    cpu_config = dict(reference, source_path=str(scene), start_frame=69, end_frame=117,
                      model_path=str(ROOT / "runs/scene0029_069_117_hybrid_30000_seed0_GN11"),
                      data_device="cpu", sgcm_guard_enabled=False)
    require(not cpu_config["initialization"] and not cpu_config["debug_cuda"], "不得进入初始化写PLY/CUDA调试分支")
    random.seed(0)
    np.random.seed(0)
    geometry_args = OmegaConf.create(cpu_config)
    transform_path = scene / "transform_matrix.txt"
    original_savetxt = np.savetxt
    transform_action = []

    def transform_write_only(path, array, *positional, **keywords):
        require(same(path, transform_path), "CPU reader试图写入非授权输出: " + str(path))
        if transform_path.exists():
            require(np.allclose(np.loadtxt(transform_path), array, rtol=0, atol=1e-12),
                    "既有transform与实际reader不同；拒绝覆盖")
            transform_action.append("existing_actual_transform_checked_without_overwrite")
            return
        with transform_path.open("x", encoding="utf-8") as stream:
            original_savetxt(stream, array, *positional, **keywords)
        transform_action.append("new_actual_loader_transform_written_exclusively")

    np.savetxt = transform_write_only
    try:
        info = loader.readWaymoInfo_add(geometry_args)
    finally:
        np.savetxt = original_savetxt
    require(len(transform_action) == 1 and not torch.cuda.is_initialized(), "实际loader transform或CPU隔离未满足")
    require(len(info.train_cameras) == 49 and not info.test_cameras and
            [c.image_name for c in info.train_cameras] == FRAME_IDS, "实际loader必须49 train/0 test")
    require(np.allclose([c.timestamp for c in info.train_cameras], [r["model_time"] for r in prep["frames"]],
                        rtol=0, atol=1e-12) and np.allclose(info.time_duration, [-.48, .48])
            and np.isclose(info.time_interval, .02), "实际reader model times改变")
    require(len(info.dynamic_point_cloud.points) == sum(plan["actor_PLY_vertices"]), "实际loader丢失actor先验")
    require(len(info.static_point_cloud.points) > 0 and np.isfinite(info.static_point_cloud.points).all()
            and np.isfinite(info.dynamic_point_cloud.points).all(), "实际reader点云无效")
    transform = np.loadtxt(transform_path)
    scale = sgcm.validate_similarity(transform)
    frozen_views, mask_rows = template.cpu_views(info, scene, torch, graphics, masks, helper)
    controller = sgcm.create_sgcm_controller(str(scene), geometry_args)
    require(not torch.cuda.is_initialized() and len(frozen_views) == 49, "CPU SGCM不得初始化CUDA")
    require(same(controller.metadata["anchor_pose"], scene / "pose/0029069.txt") and
            np.allclose(controller.metadata["world_to_model"], transform, atol=1e-12, rtol=0)
            and np.isclose(controller.metadata["dr_model"], .5 * scale), "SGCM ground anchor/metric transform不匹配")
    # precise文件若已有只允许与实际reader同值；factory优先读它，不能接受旧scene变换。
    precise = scene / "sgcm_transform_matrix.txt"
    if precise.exists():
        require(np.allclose(np.loadtxt(precise), transform, atol=1e-12, rtol=0), "既有SGCM precise transform与本scene实际loader不同")
    actor_build = read(scene / "dynamic_multi_actor_preparation.json")
    spec = {"scene_id": SCENE_ID, "source_path": str(scene), "frame_ids": FRAME_IDS, "frame_count": 49,
            "camera": "FRONT", "camera_directory": "image_0", "raw_frame_range": [69, 117],
            "resolution_scale": 4, "original_WH": [1920, 1280], "render_WH": [480, 320],
            "fixed_pose_anchor": "0029079", "fixed_pose_camera_index": 10,
            "model_timestamp_values": [float(c.timestamp) for c in info.train_cameras],
            "actor_selection_path": str(args.selection),
            "human_verdict": selection.get("human_verdict"),
            "actors": [{"actor_index": a["actor_index"], "description": a["description"],
                        "raw_track_id": a["raw_track_id"], "reviewed_selection": str(args.selection),
                        "reviewed_frames": [dict(r) for r in a["frames"]],
                        "instance_ids_by_frame": [{"frame_id": r["frame_id"], "instance_ids": r.get("instance_ids"),
                                                   "visibility": r["visibility"]} for r in a["frames"]]}
                       for a in selection["actors"]],
            "actor_PLY_paths": [str(scene / ("points3d_dynamic_%d.ply" % i)) for i in range(plan["actor_count"])],
            "author_video_verified_raw_range": prep["author_video_verified_raw_range"],
            "author_training_window_publicly_confirmed": False,
            "training_window_provenance": prep["training_window_provenance"]}
    normalized = scene_spec(spec)
    candidate = build_config(reference, normalized, ROOT)
    result = {"status": "PASS_CPU_PREPARATION_TRAINING_GATED",
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "host": "wm-vgpu-1011",
              "gpu_used": False, "cuda_initialized": False, "training_started": False, "training_ready": False,
              "scene_copy": str(scene), "repository_copy": str(args.repository),
              "actual_loader": str(args.repository / "scene/waymo_loader2.py"),
              "CPU_template": str(args.cpu_template), "CPU_template_functions": ["import_cpu_reader", "cpu_views"],
              "train_cameras": 49, "test_cameras": 0, "static_world_prior_vertices": 600000,
              "static_points_after_exact0p1m_exclusion": len(info.static_point_cloud.points),
              "dynamic_points": len(info.dynamic_point_cloud.points), "actor_PLY_vertices": plan["actor_PLY_vertices"],
              "canonical_full_mask_validation": mask_rows, "real_source_assets_validation": source_validation,
              "fixed_pose_anchor": "0029079", "fixed_pose_anchor_timestamp": float(info.train_cameras[10].timestamp),
              "SGCM_ground_anchor_frame": "0029069", "transform_matrix": transform.tolist(),
              "transform_write_action": transform_action[0], "SGCM_geometry": controller.metadata,
              "SGCM_view_inputs": {"count": len(frozen_views), "ids": [Path(v["id"]).stem for v in frozen_views]},
              "training_guard_enabled_in_candidate_config": True, "cpu_geometry_guard_enabled": False,
              "guard_scope": "Actual CPU CameraInfo projections and geometry only; GaussianModel/lineage/CUDA runtime binding pending.",
              "actor_selection": str(args.selection), "actor_build_receipt": str(scene / "dynamic_multi_actor_preparation.json"),
              "actor_validation_receipt": str(scene / "dynamic_multi_actor_validation.json"),
              "sky_DEVA_execution_receipt": str(args.execution_receipt), "actor_build_provenance": actor_build,
              "CPU_adapters": ["Original BasicPointCloud-only scene import shim", "Exact cKDTree query forced to1 CPUworker",
                               "CUDA lazy init/Tensor.cuda/Module.cuda traps", "No-clobber original loader transform writer"],
              "training_gates_pending": ["root scene-specific GN11 gate", "independent source prepare",
                                         "GPU ownership and actual model/SGCM guard binding", "complete checkpoint/runtime validation"],
              "shadow_supervision": "absent", "m_detector_used": False, "gt_labels_used_to_filter_training_inputs": False,
              "static_prior_policy": "Original mixed world600k seed0 retained; only original strict Euclidean<0.1m exclusion.",
              "architecture_components_mermaid": "flowchart LR\nA[Real49 RGB LiDAR pose sky]-->B[Root reviewed body and real prior]\nB-->C[Actual CPU reader and transform]\nC-->D[49 canonical masks and SGCM geometry]\nD-->E[SceneSpec and CPU readiness]\nE-->F[Independent source prepare]\nF-->G[Root gate and training runtime]"}
    readiness_contract(result, normalized)
    inventory = input_inventory(normalized)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "scene0029_spec.json", spec)
    write_new(output / "scene0029_cpu_readiness.json", result)
    write_new(output / "scene0029_inherited_GN11_reference.json", dict(reference, source_path=str(scene),
               start_frame=69, end_frame=117, model_path=candidate["model_path"]))
    write_new(output / "scene0029_cpu_geometry_config.json", cpu_config)
    write_new(output / "scene0029_input_inventory.json", inventory)
    return {"status": result["status"], "scene_spec": str(output / "scene0029_spec.json"),
            "readiness_file": str(output / "scene0029_cpu_readiness.json"),
            "reference_config": str(output / "scene0029_inherited_GN11_reference.json"),
            "static_seed_points": 600000, "static_points_after_exact0p1m_exclusion": len(info.static_point_cloud.points),
            "actor_points": plan["actor_PLY_vertices"], "ground_rmse_m": controller.metadata["ground_fit"]["rmse_m"],
            "training_started": False, "training_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("check", "build"), default="check")
    parser.add_argument("--scene", type=Path, default=ROOT / "inputs/scene0029_069_117_red_cpu_v1")
    parser.add_argument("--selection", type=Path, help="root完成实际49 body视觉审查后的selection；无默认actor ID")
    parser.add_argument("--execution-receipt", type=Path, default=ROOT / "metadata/red_masks_20261013_attempt2/status.json")
    parser.add_argument("--static-source-indices", type=Path,
                        default=ROOT / "metadata/red_preparation/scene0029_069_117/static_seed_real_source_indices.npz")
    parser.add_argument("--repository", type=Path, default=ROOT / "frozen_repository")
    parser.add_argument("--cpu-template", type=Path, default=ROOT / "tools/prepare_scene0008_cpu.py")
    parser.add_argument("--followup-tools", type=Path, default=ROOT / "tools/followup")
    parser.add_argument("--reference-config", type=Path, default=MATRIX / "configs/GN11.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "metadata/red_readiness_adapter")
    args = parser.parse_args()
    try:
        result = build(args) if args.stage == "build" else check(args)
        result["python_version"] = sys.version.split()[0]
        require(not any(n == "tensorflow" or n.startswith("tensorflow.") for n in sys.modules), "禁止导入TensorFlow")
        if args.stage == "check":
            require("torch" not in sys.modules, "默认check必须不导入Torch")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if result["status"].startswith("BLOCKED"):
            raise SystemExit(2)
    except (ValueError, KeyError, OSError, TypeError, AssertionError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_RED_CPU_READINESS", "reason": str(error),
                          "training_started": False, "training_ready": False}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

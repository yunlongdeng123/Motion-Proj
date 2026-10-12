"""隔离复制 scene0008 v2 后运行实际 dataset reader；默认只输出计划。

必须在 wm-vgpu-1011 用已有 drivingeditor 环境执行 --prepare；motionproj 环境缺 open3d。
仅新 followup 目录可写，不改历史输入、不创建训练进程、不执行 Camera/GaussianModel。
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import copy
import datetime
import importlib
import json
from pathlib import Path
import random
import shutil
import sys
import types

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image

from audit_two_scene_inputs import BASE, SCENE, SELECTION, audit_scene, file_inventory, read_json, resource_snapshot

ROOT = BASE / "campaign/followup_scene_prep_20261013"
REPOSITORY = BASE / "campaign/hybrid30k_20261012/frozen_repository"
TEMPLATE_CONFIG = BASE / "campaign/hybrid30k_20261012/configs/GN11.json"


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def ensure_below(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path == parent or parent not in path.parents:
        raise ValueError("Path must be strictly within requested followup directory: " + str(path))
    return path


def plan(root):
    root = Path(root).resolve()
    if root != ROOT.resolve():
        raise ValueError("This helper only writes the requested followup_scene_prep_20261013 directory")
    scene = root / "inputs/scene0008_023_071_hybrid_cpu_v1"
    resources = resource_snapshot()
    inventory = file_inventory(SCENE)
    repo_inventory = file_inventory(REPOSITORY)
    needed = inventory["bytes"] + repo_inventory["bytes"]
    if resources["disk_free_bytes"] < needed + 50 * 1024 ** 3:
        raise RuntimeError("Require30GiB disk floor plus20GiB shared-training/scratch reserve after new copy")
    config = read_json(TEMPLATE_CONFIG)
    assert config["iterations"] == config["position_lr_max_steps"] == 30000
    assert config["start_checkpoint"] is None and config["seed"] == 0
    assert config["resolution_scales"] == [4] and config["including_dynamic"] and not config["eval"]
    assert config["hybrid_lambda_dynamic_opacity"] == 100 and config["hybrid_lambda_static_rgb"] == 1
    assert config["lambda_t_reg"] == .1 and config["lambda_v_reg"] == .01 and config["lidar_decay"] == 1
    assert config["paper_sgcm_enabled"] and config["hybrid_nps_enabled"] and config["sgcm_guard_enabled"]
    for name in ("checkpoint_iterations", "save_iterations", "test_iterations"):
        assert config[name] == [10000, 20000, 30000]
    return {"status": "CPU_PLAN_ONLY", "source": str(SCENE), "target": str(scene), "root": str(root),
            "source_inventory": inventory, "frozen_repository_inventory": repo_inventory,
            "required_copy_bytes": needed, "resources": resources,
            "gpu_used": False, "training_started": False,
            "actual_reader": "scene.waymo_loader2.readWaymoInfo_add",
            "fixed_pose_anchor": "0008033", "raw_training_range": [23, 71],
            "author_video_verified_range": [23, 51], "author_training_window_publicly_confirmed": False,
            "training_gate": "Current silver-scene GN11 gate and separate scene-specific training-runtime validation remain required.",
            "architecture_components_mermaid": "flowchart LR\nA[Existing49 RGB LiDAR pose body sky]-->B[Independent input copy]\nB-->C[Actual CPU dataset reader]\nC-->D[Transform two actor roles full mask]\nD-->E[SGCM ground regions view inputs]\nE-->F[CPU readiness evidence]\nF-->G[GN11 gate then separate training validation]"}


def import_cpu_reader(repository):
    """保留实际 reader，避免 scene 包初始化模型；BasicPointCloud 使用原数据结构。"""
    import torch
    import scipy.spatial

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    def forbidden_cuda(*args, **kwargs):
        raise RuntimeError("CUDA use is forbidden in this CPU preparation helper")

    torch.cuda._lazy_init = forbidden_cuda
    torch.Tensor.cuda = forbidden_cuda
    torch.nn.Module.cuda = forbidden_cuda
    assert not torch.cuda.is_initialized()
    sys.path.insert(0, str(repository))
    package = types.ModuleType("scene")
    package.__path__ = [str(repository / "scene")]
    sys.modules["scene"] = package
    graphics = importlib.import_module("utils.graphics_utils")
    minimal_model = types.ModuleType("scene.gaussian_model")
    minimal_model.BasicPointCloud = graphics.BasicPointCloud
    sys.modules["scene.gaussian_model"] = minimal_model
    original_tree = scipy.spatial.cKDTree

    class SingleThreadTree:
        def __init__(self, *args, **kwargs):
            self.tree = original_tree(*args, **kwargs)

        def query(self, *args, **kwargs):
            kwargs["workers"] = 1
            return self.tree.query(*args, **kwargs)

    scipy.spatial.cKDTree = SingleThreadTree
    loader = importlib.import_module("scene.waymo_loader2")
    sgcm = importlib.import_module("scene.paper_sgcm")
    masks = importlib.import_module("utils.paper_masks")
    helper = importlib.import_module("scene.hybrid_sgcm_helpers")
    return torch, loader, sgcm, graphics, masks, helper


def cpu_views(info, scene, torch, graphics, masks, helper):
    views, mask_rows = [], []
    for camera in info.train_cameras:
        assert camera.width == 1920 and camera.height == 1280
        assert np.isfinite(camera.pointcloud_camera).all()
        world = torch.from_numpy(graphics.getWorld2View2(camera.R, camera.T)).transpose(0, 1)
        projection = graphics.getProjectionMatrixCenterShift(.01, 1000., camera.cx / 4, camera.cy / 4,
                                                             camera.fx / 4, camera.fy / 4, 480, 320).transpose(0, 1)
        view = types.SimpleNamespace(world_view_transform=world, full_proj_transform=world @ projection,
                                     timestamp=camera.timestamp, image_width=480, image_height=320,
                                     image_path=camera.image_path, image_name=camera.image_name)
        actual = masks.load_dynamic_mask(str(scene), view)
        with Image.open(scene / "dynamic_masks/dynamic_mask_0" / (camera.image_name + ".png")) as original:
            nearest = np.array(original.convert("L").resize((480, 320), Image.Resampling.NEAREST)) > 0
        assert actual.device.type == "cpu" and actual.shape == (1, 320, 480)
        assert np.array_equal(actual.numpy()[0] > 0, nearest)
        mask_rows.append({"frame_id": camera.image_name, "scale4_body_pixels": int(nearest.sum()),
                          "actual_canonical_mask_loader_exact": True})
        views.append(view)
    frozen = helper.freeze_training_views(views)
    assert len(frozen) == 49
    return frozen, mask_rows


def prepare(root):
    record = plan(root)
    root = Path(record["root"])
    scene = ensure_below(record["target"], root)
    repository = ensure_below(root / "frozen_repository", root)
    metadata = ensure_below(root / "metadata", root)
    for target in (scene, repository, metadata / "scene0008_cpu_readiness.json"):
        if target.exists():
            raise FileExistsError("Refuse to overwrite previous preparation: " + str(target))
    # 先核对历史源；仅目标复制与输出会写入新目录。
    source_audit = audit_scene(SCENE, SELECTION)
    scene.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SCENE, scene)
    shutil.copytree(REPOSITORY, repository, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    metadata.mkdir(parents=True, exist_ok=True)
    selection_copy = metadata / "scene0008_dual_actor_selection.inherited.json"
    shutil.copy2(SELECTION, selection_copy)
    write_new(metadata / "scene0008_source_readonly_audit.json", source_audit)
    copied_audit = audit_scene(scene, selection_copy)
    write_new(metadata / "scene0008_copied_input_audit.json", copied_audit)
    assert source_audit["inventory"] == copied_audit["inventory"]
    config = read_json(TEMPLATE_CONFIG)
    config.update(source_path=str(scene), start_frame=23, end_frame=71,
                  model_path=str(root / "runs/scene0008_023_071_hybrid_30000_seed0_GN11"))
    # 这份训练配置只记录继承项；当前 helper 不提供启动入口。
    write_new(metadata / "scene0008_inherited_training_config_candidate.json", config)
    cpu_config = copy.deepcopy(config)
    cpu_config["data_device"] = "cpu"
    cpu_config["sgcm_guard_enabled"] = False
    write_new(metadata / "scene0008_cpu_geometry_config.json", cpu_config)
    from omegaconf import OmegaConf

    torch, loader, sgcm, graphics, masks, helper = import_cpu_reader(repository)
    random.seed(0)
    np.random.seed(0)
    args = OmegaConf.create(cpu_config)
    info = loader.readWaymoInfo_add(args)
    assert not torch.cuda.is_initialized()
    assert len(info.train_cameras) == 49 and len(info.test_cameras) == 0
    assert [camera.image_name for camera in info.train_cameras] == copied_audit["frame_ids"]
    prep = read_json(scene / "preparation_manifest.json")
    assert np.allclose([camera.timestamp for camera in info.train_cameras], [row["model_time"] for row in prep["frames"]], rtol=0, atol=1e-12)
    assert np.allclose(info.time_duration, [-.48, .48]) and np.isclose(info.time_interval, .02)
    assert len(info.dynamic_point_cloud.points) == sum(copied_audit["actor_PLY_vertices"])
    transform = np.loadtxt(scene / "transform_matrix.txt")
    scale = sgcm.validate_similarity(transform)
    assert np.isclose(scale, .1)
    frozen_views, mask_rows = cpu_views(info, scene, torch, graphics, masks, helper)
    controller = sgcm.create_sgcm_controller(str(scene), args)
    assert not torch.cuda.is_initialized()
    # 仅做前車保留的诊断；不以 GT 框或标签裁剪任何训练点。
    copied_points = info.static_point_cloud.points
    assert len(copied_points) > 0 and np.isfinite(copied_points).all()
    spec = {"scene_id": "scene0008_023_071", "source_path": str(scene), "frame_ids": copied_audit["frame_ids"],
            "frame_count": 49, "camera": "FRONT", "raw_frame_range": [23, 71], "resolution_scale": 4,
            "original_WH": [1920, 1280], "render_WH": [480, 320], "fixed_pose_anchor": "0008033",
            "fixed_pose_camera_index": 10, "model_timestamp_values": [float(camera.timestamp) for camera in info.train_cameras],
            "actor_instance_ids": [77, 82], "actor_PLY_paths": [str(scene / ("points3d_dynamic_%d.ply" % actor)) for actor in range(2)],
            "preserve_track_id": "vBOasuXpjI7h7d-X_VEQdA", "preserve_DEVA_id": 153,
            "author_video_verified_raw_range": [23, 51], "author_training_window_publicly_confirmed": False}
    write_new(metadata / "scene0008_spec.json", spec)
    result = {"status": "PASS_CPU_PREPARATION_TRAINING_GATED", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "host": "wm-vgpu-1011", "gpu_used": False, "cuda_initialized": False, "training_started": False,
              "training_ready": False, "source_unmodified": str(SCENE), "scene_copy": str(scene), "repository_copy": str(repository),
              "actual_loader": str(repository / "scene/waymo_loader2.py"), "train_cameras": 49, "test_cameras": 0,
              "static_world_prior_vertices": copied_audit["static_PLY_vertices"], "static_points_after_exact0p1m_exclusion": len(copied_points),
              "dynamic_points": len(info.dynamic_point_cloud.points), "actor_PLY_vertices": copied_audit["actor_PLY_vertices"],
              "canonical_full_mask_validation": mask_rows, "fixed_pose_anchor": "0008033", "transform_matrix": transform.tolist(),
              "SGCM_geometry": controller.metadata, "SGCM_view_inputs": {"count": len(frozen_views), "ids": [Path(view["id"]).stem for view in frozen_views]},
              "training_guard_enabled_in_candidate_config": True, "cpu_geometry_guard_enabled": False,
              "guard_scope": "Geometry-only controller; actual49 CameraInfo projections separately validated on CPU. No GaussianModel/SGCM lineage binding or CUDA densification occurs.",
              "CPU_adapters": ["scene package/model import shim exposes original BasicPointCloud only", "cKDTree query workers=1; exact distance/eps/threshold unchanged", "CUDA lazy init/Tensor.cuda/Module.cuda trap"],
              "training_gates_pending": ["Current silver-scene GN11 user gate", "New scene-specific run_matrix/runtime config and fixed-pose adaptation", "Actual training GaussianModel/view/SGCM guard and lineage binding", "GPU ownership and complete checkpoint runtime validation"],
              "shadow_supervision": "absent", "m_detector_used": False, "gt_labels_used_to_filter_training_inputs": False,
              "preprocessing_limits": copied_audit["preprocessing_caveat"],
              "source_window_boundary": "Raw23..51 matches author visible video; raw23..71 is the local49-view training choice, not a publicly confirmed author training window.",
              "resources_after": resource_snapshot(), "architecture_components_mermaid": record["architecture_components_mermaid"]}
    write_new(metadata / "scene0008_cpu_readiness.json", result)
    write_new(scene / "full_loader_validation.followup_cpu.json", {"status": "PASS_CPU_ACTUAL_READER", "scope": result["guard_scope"], "train_cameras": 49,
                                                                  "static_points": len(copied_points), "dynamic_points": len(info.dynamic_point_cloud.points)})
    write_new(scene / "sgcm_input_validation.followup_cpu.json", {"status": "PASS_GEOMETRY_AND_VIEW_INPUTS_TRAINING_GUARD_PENDING", "metadata": controller.metadata})
    print(json.dumps({"status": result["status"], "readiness": str(metadata / "scene0008_cpu_readiness.json"), "training_ready": False}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root)
    else:
        print(json.dumps(plan(args.root), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

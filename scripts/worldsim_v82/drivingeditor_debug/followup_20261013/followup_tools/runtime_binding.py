"""仅真实--run中检查实际Scene/SGCM绑定；不把CPU准备写成GPU formal-ready。"""
import functools
from pathlib import Path
from contracts import completion_contract, read_json, require, require_disk_space, disk_forecast, write_new


def bind_training_contract(training, config, readiness):
    spec = config["followup_scene_spec"]
    original_scene = training.Scene

    @functools.wraps(original_scene)
    def checked_scene(*args, **kwargs):
        scene = original_scene(*args, **kwargs)
        from utils.hybrid30k_runtime import camera_id
        cameras = sorted(scene.getTrainCameras(), key=lambda camera: float(camera.timestamp))
        require([camera_id(c) for c in cameras] == spec["frame_ids_with_camera"], "actual Scene身份与sceneSpec不匹配")
        require(all(abs(float(c.timestamp) - t) < 1e-7 for c, t in zip(cameras, spec["model_timestamps"])),
                "actual Scene timestamps不匹配")
        require(all(c.image_width == 480 and c.image_height == 320 for c in cameras),
                "actual Scene render dimensions不是W480H320")
        require(not scene.getTestCameras(scale=4), "跟随协议不能悄然创建heldout")
        scene.followup_scene_spec = spec
        write_new(Path(config["model_path"]) / "actual_scene_binding.json",
                  {"status": "PASS_ACTUAL_GPU_SCENE_BINDING", "scene_id": spec["scene_id"],
                   "frame_ids": spec["frame_ids_with_camera"], "timestamps": spec["model_timestamps"],
                   "render_wh": [480, 320], "anchor_pose_frame": spec["anchor_frame"],
                   "actual_gpu_initialized": True, "SGCM_binding_pending": True,
                   "restoration_verified": False})
        return scene

    training.Scene = checked_scene
    # training()在函数内local-import该factory；应在真实train前改来源module。
    from scene import paper_sgcm
    original_factory = paper_sgcm.create_sgcm_controller

    @functools.wraps(original_factory)
    def checked_factory(source_path, args, training_views=None, gaussians=None):
        controller = original_factory(source_path, args, training_views=training_views, gaussians=gaussians)
        require(controller.config.guard_enabled is True and gaussians is not None, "actual corrected SGCM guard未绑定")
        require(len(controller.training_views) == 49, "actual SGCM缺49 view/stamp")
        actual_ids = [Path(view["id"]).parent.name + "/" + Path(view["id"]).stem
                      for view in controller.training_views]
        require(set(actual_ids) == set(spec["frame_ids_with_camera"]), "actual SGCM view身份不匹配")
        require(gaussians._sgcm_lineage_ids is not None and gaussians._sgcm_clone_blocked is not None
                and gaussians._sgcm_lineage_ids.shape == gaussians._sgcm_clone_blocked.shape
                and gaussians._sgcm_lineage_ids.numel() == len(gaussians.get_xyz),
                "actual SGCM lineage与静态拓扑不匹配")
        import numpy as np
        require(np.allclose(controller.metadata["world_to_model"], readiness["transform_matrix"], atol=1e-5, rtol=1e-4),
                "actual loader transform与CPU准备不一致")
        write_new(Path(config["model_path"]) / "actual_sgcm_binding.json",
                  {"status": "PASS_ACTUAL_GPU_SGCM_LINEAGE_BINDING", "scene_id": spec["scene_id"],
                   "training_views": 49, "static_points": len(gaussians.get_xyz),
                   "guard_enabled": True, "metadata": controller.metadata,
                   "restoration_verified": False})
        return controller

    paper_sgcm.create_sgcm_controller = checked_factory
    # save前再次预测剩余完整节点；不清理任何资产。
    from utils import paper_checkpoint
    original_save = paper_checkpoint.save_components

    @functools.wraps(original_save)
    def checked_save(model_path, iteration, models):
        static = len(models["static"].get_xyz)
        dynamic = len(models["dynamic"].get_xyz)
        remaining = sum(step >= iteration for step in (10000, 20000, 30000))
        forecast = disk_forecast(config, actual_counts=(static, dynamic), remaining_nodes=remaining)
        result = require_disk_space(Path(model_path), forecast)
        write_new(Path(model_path) / ("disk_before_checkpoint_%d.json" % iteration), result)
        return original_save(model_path, iteration, models)

    paper_checkpoint.save_components = checked_save
    return training


def validate_completion(output, config):
    return completion_contract(output, config)

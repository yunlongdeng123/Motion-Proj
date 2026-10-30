"""用当前真实 GN11 train.training 重建独立仿真；只适配观测合同和工程入口。

不冻结 actor 或背景、不重写训练循环、不读取 privileged 参考资产。
完整 alpha/静态 RGB/静态寿命和速度/LiDAR 衰减/SGCM/NPS 均继承原代码。
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True

SOURCE = Path("/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137")
BASE = Path(__file__).resolve().with_name("gn11_base.json")


class SavedBoundary(Exception):
    pass


class NativePhaseTiming:
    """训练循环墙钟代理；明确扣除测得的评价/保存时间，不冒称纯GPU计算秒。"""
    LOOP_EXCLUDED = {"evaluation", "component_save", "continuation_capture", "checkpoint_publish"}

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.invocation_started = clock()
        self.first_record_at = None
        self.phase_seconds = {}
        self.loop_phase_seconds = {}
        self.stop_trigger = None

    def measured(self, phase, function):
        def call(*args, **kwargs):
            started = self.clock()
            in_loop = self.first_record_at is not None
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = self.clock() - started
                self.phase_seconds[phase] = self.phase_seconds.get(phase, 0.) + elapsed
                if in_loop:
                    self.loop_phase_seconds[phase] = self.loop_phase_seconds.get(phase, 0.) + elapsed
        return call

    def snapshot(self, now=None):
        now = self.clock() if now is None else now
        loop_wall = None if self.first_record_at is None else now - self.first_record_at
        excluded = sum(value for key, value in self.loop_phase_seconds.items() if key in self.LOOP_EXCLUDED)
        return dict(native_invocation_wall_seconds=now - self.invocation_started,
            initialization_and_restore_wall_seconds_before_first_record=(now if self.first_record_at is None else self.first_record_at) - self.invocation_started,
            wall_seconds_since_first_record=loop_wall,
            training_loop_wall_seconds_excluding_measured_eval_checkpoint=None if loop_wall is None else max(0., loop_wall - excluded),
            measured_loop_eval_checkpoint_seconds=excluded,
            phase_wall_seconds=dict(self.phase_seconds), loop_phase_wall_seconds=dict(self.loop_phase_seconds),
            stop_trigger=self.stop_trigger,
            scope="host wall time starts at first loss record; excludes measured native evaluation/component-save/continuation-capture/checkpoint-publication; includes forward/backward/optimizer/densification/logging; not GPU kernel time; first forward precedes timer",
            restore_phase_scope="component_restore and continuation_restore_and_verification occur inside pre-first-record initialization and are diagnostic subsets, not extra additive wall time")


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def assert_method(config):
    required = dict(hybrid_method=True, paper_method=True, including_dynamic=True,
        hybrid_nps_enabled=True, paper_sgcm_enabled=True, sgcm_guard_enabled=True,
        hybrid_lambda_dynamic_rgb=5.0, hybrid_lambda_sky_rgb=5.0, hybrid_lambda_static_rgb=1.0,
        hybrid_lambda_dynamic_opacity=100.0, lambda_t_reg=.1, lambda_v_reg=.01,
        paper_lambda_nps=.01, paper_lambda_sky=.05, paper_lambda_depth=.1,
        lambda_inv_depth=.1, lambda_v_smooth=.1, lambda_opacity_entropy=.05,
        lambda_dssim=.2, lidar_decay=1.0, position_lr_max_steps=30000, iterations=30000,
        random_init_point=0, dynamic_random_init_point=0,
        hybrid_opacity_reset_policy="densification_only")
    for key, value in required.items():
        if type(config.get(key)) is not type(value) or config[key] != value:
            raise ValueError("当前 GN11 默认机制/权重/学习率合同发生变化: " + key)
    if config.get("eval") is not False or config.get("require_boundingbox") is not False:
        raise ValueError("本轮保留 all-observed 重建和无 GT bbox 点筛选")


def configure(args, prepared):
    base = json.loads(args.base_config.read_text(encoding="utf-8-sig"))
    assert_method(base)
    if args.resume:
        import torch
        step = int(args.resume.stem.replace("chkpnt", ""))
        state = torch.load(args.resume.parent / ("training_state%d.pth" % step), map_location="cpu", weights_only=False)
        config = dict(state["config"], model_path=str(args.output.resolve()), start_checkpoint=str(args.resume.resolve()))
        if config["source_path"] != str(args.scene.resolve()):
            raise ValueError("精确恢复要求同一个原生数据集")
        changes = dict(resumed_complete_saved_config=True, model_path=config["model_path"], start_checkpoint=config["start_checkpoint"])
    else:
        spec = prepared["followup_scene_spec"]
        nodes = sorted(set(range(args.checkpoint_every, args.steps + 1, args.checkpoint_every)) | {args.steps})
        if args.steps > base["iterations"]:
            raise ValueError("M0 预算不能超出原生30K合同")
        changes = dict(source_path=str(args.scene.resolve()), model_path=str(args.output.resolve()), start_checkpoint=None,
            start_frame=0, end_frame=len(prepared["frames"]) - 1, resolution_scales=[prepared["render_scale"]],
            frame_interval=prepared["model_frame_interval"], time_duration=[spec["model_timestamps"][0], spec["model_timestamps"][-1]],
            v43_update_budget=args.steps, test_iterations=nodes, save_iterations=nodes, checkpoint_iterations=nodes,
            followup_scene_spec=spec, followup_static_anchor_frame=spec["anchor_frame"])
        config = dict(base, **changes)
    assert_method(config)
    if prepared["render_wh"] != [480, 320]:
        raise ValueError("当前原生评价与 static-time 检查要求 render_wh=480x320；不静默移除检查")
    return config, changes


def bind_scene_contract(train, config):
    original = train.Scene
    spec = config["followup_scene_spec"]

    def checked_scene(*args, **kwargs):
        scene = original(*args, **kwargs)
        cameras = sorted(scene.getTrainCameras(), key=lambda camera: float(camera.timestamp))
        ids = [Path(camera.image_path).parent.name + "/" + camera.image_name for camera in cameras]
        if ids != spec["frame_ids_with_camera"] or len(cameras) != len(spec["frame_ids"]):
            raise ValueError("真实 Scene 观测身份与仿真 manifest 不符")
        if any(abs(float(camera.timestamp) - timestamp) > 1e-7 for camera, timestamp in zip(cameras, spec["model_timestamps"])):
            raise ValueError("真实 Scene 模型时间与 manifest 不符")
        if any([camera.image_width, camera.image_height] != spec["render_wh"] for camera in cameras):
            raise ValueError("真实 Scene 渲染尺寸与 manifest 不符")
        if scene.getTestCameras(scale=config["resolution_scales"][0]):
            raise ValueError("本轮是 all-observed 重建，不额外制造 heldout")
        scene.followup_scene_spec = spec
        write(Path(config["model_path"]) / "actual_scene_binding.json", dict(status="PASS_ACTUAL_SCENE_BINDING",
            frame_ids=ids, timestamps=spec["model_timestamps"], render_wh=spec["render_wh"],
            source_observation_count=len(cameras), source_49_view_check_generalized=True,
            anchor_pose_frame=spec["anchor_frame"]))
        return scene

    train.Scene = checked_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--base-config", type=Path, default=BASE)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--smoke-seconds", type=float, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--entry-steps", type=int, default=0, help="从完整恢复状态再执行指定步数并保存，配置与 LR 合同不变")
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    args.scene, args.output, args.source, args.base_config = [path.resolve() for path in (args.scene, args.output, args.source, args.base_config)]
    if args.resume:
        args.resume = args.resume.resolve()
    if args.steps < 1 or args.checkpoint_every < 1 or args.smoke_seconds < 0 or args.entry_steps < 0:
        parser.error("步数和保存间隔须为正，smoke/entry 预算不能为负")
    prepared = json.loads((args.scene / "preparation_manifest.json").read_text(encoding="utf-8-sig"))
    if not prepared.get("training_ready") or prepared.get("independent_reference_used_for_training") is not False:
        raise ValueError("需要已验收的受限观测适配数据，禁止参考真值流入训练")
    if args.output.exists():
        raise FileExistsError("不覆盖旧训练；新建 attempt，精确恢复只能显式 --resume")
    config, changes = configure(args, prepared)
    args.output.mkdir(parents=True)
    write(args.output / "effective_config.json", config)
    write(args.output / "config_delta.json", dict(base=str(args.base_config), changes=changes,
        initialization="600K sampled sparse LiDAR static prior; dynamic projection/depth/component prior",
        training_algorithm_changed=False, losses_sampling_distribution_lr_densification_inherited=True,
        adapter_delta=["49-view cardinality guard generalized to declared unique observed view count",
                       "scene binding metadata reads simulator manifest", "input resolution scale reflects actual480x320 pixels"],
        smoke_extra_not_formal=bool(args.smoke_seconds), entry_extra_not_formal=bool(args.entry_steps),
        observations=dict(frames=len(prepared["frames"]), original_wh=prepared["original_wh"], render_wh=prepared["render_wh"],
                          masks=prepared["mask_input"], gt_mesh_used_for_initialization=False, gt_owner_used_for_initialization=False),
        scientific_contamination_claim=False, real_scene_generalization_claim=False))
    if args.config_only:
        print(json.dumps(dict(status="CONFIG_ONLY", output=str(args.output), engine_schedule_iterations=config["iterations"],
                              update_budget=config.get("v43_update_budget", config["iterations"]))))
        return
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import random
    import torch
    from omegaconf import OmegaConf
    import train
    import utils.paper_checkpoint as checkpoint
    import utils.hybrid30k_runtime as runtime
    from utils.general_utils import seed_everything
    if Path(train.__file__).resolve() != args.source.resolve() / "train.py":
        raise RuntimeError("未导入已核对的当前 DrivingEditor 真实训练入口")
    if not torch.cuda.is_available():
        raise RuntimeError("真实训练必须使用 CUDA；不降级为 CPU 或 toy trainer")
    original_sampler = runtime.CameraSampler

    class ObservedCountSampler(original_sampler):
        """仅替换历史49视图 guard，select/capture/restore 均继承原类。"""
        def __init__(self, seed, cameras):
            self.ids = sorted(runtime.camera_id(camera) for camera in cameras)
            if len(self.ids) != len(prepared["frames"]) or len(set(self.ids)) != len(self.ids):
                raise ValueError("仿真观测数量或唯一身份不符")
            self.python = random.Random(seed)
            self.numpy = np.random.RandomState(seed)
            self.stack, self.steps = [], 0

    runtime.CameraSampler = ObservedCountSampler
    bind_scene_contract(train, config)
    os.chdir(args.source)
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    seed_everything(config["seed"])
    started, active_started = time.monotonic(), [None]
    timing = NativePhaseTiming()
    native_ended = [None]
    train.complete_eval = timing.measured("evaluation", train.complete_eval)
    checkpoint.save_components = timing.measured("component_save", checkpoint.save_components)
    checkpoint.capture_training_state = timing.measured("continuation_capture", checkpoint.capture_training_state)
    for model_type in (train.GaussianModel, train.Dynamic_GaussianModel, train.EnvLight):
        model_type.restore = timing.measured("component_restore", model_type.restore)
    stop, last, first, step_calls, gradient_checks, boundary_stop = [False], [0], [None], [0], [0], [False]
    stop_kind = [None]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    original_record = runtime.record_step

    def record(*call_args, **kwargs):
        iteration = int(call_args[1])
        values = call_args[-1]
        if not all(math.isfinite(float(value)) for value in values.values() if isinstance(value, (int, float))):
            raise FloatingPointError(values)
        if active_started[0] is None:
            active_started[0], first[0] = time.monotonic(), iteration
            timing.first_record_at = active_started[0]
        step_calls[0] += 1
        last[0] = iteration
        current_timing = timing.snapshot()
        elapsed = current_timing["training_loop_wall_seconds_excluding_measured_eval_checkpoint"]
        if stop[0]:
            stop_kind[0] = "signal"
        elif args.entry_steps and step_calls[0] >= args.entry_steps:
            stop_kind[0] = "entry"
        elif args.smoke_seconds and elapsed >= args.smoke_seconds:
            stop_kind[0] = "smoke"
        elif not args.entry_steps and iteration >= config.get("v43_update_budget", config["iterations"]):
            stop_kind[0] = "budget"
        boundary_stop[0] = stop_kind[0] is not None
        if boundary_stop[0]:
            if timing.stop_trigger is None:
                timing.stop_trigger = dict(reason=stop_kind[0], iteration=iteration,
                    requested_smoke_loop_wall_seconds=args.smoke_seconds,
                    training_loop_wall_seconds_excluding_measured_eval_checkpoint=elapsed,
                    wall_seconds_since_first_record=current_timing["wall_seconds_since_first_record"],
                    measured_loop_eval_checkpoint_seconds=current_timing["measured_loop_eval_checkpoint_seconds"])
                write(args.output / "stop_trigger_timing.json", timing.stop_trigger)
            for key in ("checkpoint_iterations", "save_iterations", "test_iterations"):
                if iteration not in getattr(call_args[0], key):
                    getattr(call_args[0], key).append(iteration)
        original_record(*call_args, **kwargs)
        if iteration == first[0] or iteration % 100 == 0 or boundary_stop[0]:
            write(args.output / "progress.json", dict(phase="training", iteration=iteration,
                finite_forward_steps=step_calls[0],
                training_loop_wall_seconds_excluding_measured_eval_checkpoint=elapsed,
                wall_seconds_since_first_record=current_timing["wall_seconds_since_first_record"],
                allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved()))

    runtime.record_step = record
    original_optimize = train.densification_and_optimization

    def optimize(model, *call_args, **kwargs):
        if last[0] == first[0] or last[0] % 100 == 0 or boundary_stop[0]:
            for group in model.optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError("真实 backward 产生非有限梯度: " + group.get("name", "unknown"))
            gradient_checks[0] += 1
        return original_optimize(model, *call_args, **kwargs)

    train.densification_and_optimization = optimize

    def equal(x, y):
        if torch.is_tensor(x):
            return torch.is_tensor(y) and x.shape == y.shape and torch.equal(x.detach().cpu(), y.detach().cpu())
        if isinstance(x, np.ndarray):
            return isinstance(y, np.ndarray) and np.array_equal(x, y)
        if isinstance(x, dict):
            return isinstance(y, dict) and set(x) == set(y) and all(equal(x[key], y[key]) for key in x)
        if isinstance(x, (tuple, list)):
            return type(x) is type(y) and len(x) == len(y) and all(equal(i, j) for i, j in zip(x, y))
        return x == y

    original_restore = checkpoint.restore_training_state

    def restore(path, iteration, scene, models, current_args, **kwargs):
        result = original_restore(path, iteration, scene, models, current_args, **kwargs)
        state = torch.load(path, map_location="cpu", weights_only=False)
        root = Path(path).parent
        checks = {}
        for name, prefix in (("static", "chkpnt"), ("dynamic", "dynamic_chkpnt"), ("environment", "env_light_chkpnt")):
            saved, saved_step = torch.load(root / ("%s%d.pth" % (prefix, iteration)), map_location="cpu", weights_only=False)
            checks[name + "_parameters_and_optimizer"] = saved_step == iteration and equal(models[name].capture(), saved)
        checks.update(python_rng=equal(random.getstate(), state["python_rng"]), numpy_rng=equal(np.random.get_state(), state["numpy_rng"]),
            torch_rng=equal(torch.get_rng_state(), state["torch_rng"]), cuda_rng=equal(torch.cuda.get_rng_state_all(), state["cuda_rng"]),
            sampler=equal(scene.hybrid_sampler.capture(), state["hybrid_sampler"]), scale=scene.scale_index == state["scale_index"])
        captured = checkpoint.capture_training_state(iteration, scene, result[0], models, current_args,
                                                     sgcm=kwargs.get("sgcm"), ema=result[1])
        for key in ("camera_orders", "viewpoint_stack", "model_runtime", "ema", "sgcm"):
            checks[key] = equal(captured[key], state[key])
        write(args.output / "restore_verification.json", dict(iteration=iteration, checks=checks,
            all_passed=all(checks.values()), cuda_trajectory_bitwise_claimed=False))
        if not all(checks.values()):
            raise RuntimeError("完整状态恢复验收失败: " + repr(checks))
        return result

    checkpoint.restore_training_state = timing.measured("continuation_restore_and_verification", restore)
    original_publish = checkpoint.publish_checkpoint_event
    measured_publish = timing.measured("checkpoint_publish", original_publish)

    def publish(*call_args, **kwargs):
        event = measured_publish(*call_args, **kwargs)
        iteration = int(call_args[1])
        event_path = args.output / "checkpoint_events" / ("iteration_%06d.json" % iteration)
        write(args.output / "latest_complete_node.json", dict(iteration=iteration, event=str(event_path.resolve()),
            timing=timing.snapshot()))
        write(args.output / "next_stage_entry.json", dict(status="NATIVE_RECONSTRUCTION_BUNDLE_COMPLETE", iteration=iteration,
            event=str(event_path.resolve()), source_scene=str(args.scene.resolve()),
            next_stage="independent delete/shift rendering and reference audit; no automatic scientific verdict",
            checkpoints=event["checkpoint_paths"], training_state=event["checkpoints"]["training_state"]["path"],
            independent_reference_used_for_training=False, scientific_contamination_claim=False))
        if boundary_stop[0]:
            raise SavedBoundary()
        return event

    checkpoint.publish_checkpoint_event = publish
    reason = "native_returned"
    try:
        train.training(OmegaConf.create(config))
    except SavedBoundary:
        reason = str(stop_kind[0]) + "_saved"
    finally:
        native_ended[0] = time.monotonic()
        write(args.output / "cuda_memory_peak.json", dict(allocated_bytes=torch.cuda.max_memory_allocated(),
            reserved_bytes=torch.cuda.max_memory_reserved(), scope="native initialization, training, evaluation and checkpoint"))
    write(args.output / "returned.json", dict(reason=reason, iteration=last[0], first_iteration=first[0],
        observed_finite_step_calls=step_calls[0], checked_backward_components=gradient_checks[0],
        wall_seconds=time.monotonic() - started,
        timing=timing.snapshot(now=native_ended[0]),
        native_source=str(args.source.resolve()), formal_config_budget=config.get("v43_update_budget", config["iterations"]),
        inherited_engine_and_learning_rate_horizon=config["iterations"],
        extra_smoke=bool(args.smoke_seconds), extra_entry=bool(args.entry_steps),
        quality_accepted=False, human_verdict=None, scientific_contamination_claim=False,
        real_scene_generalization_claim=False))
    if reason in ("native_returned", "budget_saved") and last[0] != config.get("v43_update_budget", config["iterations"]):
        raise RuntimeError("真实训练未达到登记的预算")


if __name__ == "__main__":
    main()

"""GN11 v3 单臂 GPU gate/train 入口。

导入不会启动任务。每个臂必须先完成零更新 CUDA gate；失败后保留现场，
不自动重试、不续训、不删除旧结果。
"""

from pathlib import Path
import argparse
import datetime
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback


METHOD_VERSION = "GN11_STAGE1_V3_20261020"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def validate(arguments):
    plan = read(arguments.plan)
    config = read(arguments.baseline_config)
    bundle = read(arguments.bundle)
    if plan.get("method_version") != METHOD_VERSION:
        raise RuntimeError("wrong v3 experiment plan")
    if arguments.arm not in plan.get("arms", {}):
        raise RuntimeError("arm is not registered in the fixed v3 plan")
    if arguments.output.exists():
        raise FileExistsError("new output required; never overwrite/replay a run")
    if bundle.get("status") != "CPU_INPUT_BUNDLE_VERIFIED":
        raise RuntimeError("CPU real-input gate incomplete")
    if bundle.get("full49_observed", 0) + bundle.get("prior", 0) <= 0:
        raise RuntimeError("road module has no fixed real target")
    if config.get("seed") != 0 or config.get("iterations") != 30000:
        raise RuntimeError("wrong baseline configuration")
    if not all(config.get(key) for key in (
        "including_dynamic", "paper_method", "hybrid_method"
    )):
        raise RuntimeError("actual native hybrid A configuration required")
    if "full5_pipeline/inputs/scene0008_089_137" not in str(config["source_path"]):
        raise RuntimeError("use actual A49 effective config")
    if float(config.get("hybrid_lambda_dynamic_opacity", -1)) != 100.0:
        raise RuntimeError("complete raw dynamic-mask opacity MSE100 must remain enabled")
    if bool(config.get("white_background", False)):
        raise RuntimeError("v3 raw dynamic color contract requires black raster background")
    evidence = read(plan["background_temporal_evidence"])
    if evidence.get("status") != "CPU_GEOMETRY_EVIDENCE_READY":
        raise RuntimeError("background temporal geometry evidence has not been prepared")
    if evidence.get("policy") != plan["background_temporal_policy"]:
        raise RuntimeError("plan and actual background temporal evidence differ")
    if arguments.mode == "train":
        if arguments.gpu_gate is None:
            raise RuntimeError("zero-update CUDA receipt required before training")
        gate = read(arguments.gpu_gate)
        if (
            gate.get("status") != "PASSED_ACTUAL_CUDA_V3_FORWARD_BACKWARD_ZERO_UPDATES"
            or gate.get("optimizer_updates") != 0
            or gate.get("method_version") != METHOD_VERSION
            or gate.get("arm") != arguments.arm
            or gate.get("background_temporal_version") != evidence.get("version")
        ):
            raise RuntimeError("CUDA gate belongs to another method/arm or performed updates")
    return plan, config, bundle


def main():
    parser = argparse.ArgumentParser()
    for key in (
        "engine", "output", "bundle", "baseline-config", "readiness",
        "binding-tools", "gpu-lock", "plan",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--arm", choices=("full", "real_only", "prior_only", "ownership_only"), required=True)
    parser.add_argument("--mode", choices=("gate", "train"), required=True)
    parser.add_argument("--gpu-gate", type=Path)
    arguments = parser.parse_args()
    plan, config, bundle = validate(arguments)

    lock = arguments.gpu_lock.open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoints = list(plan["checkpoint_steps"])
    static_limit = int(config["densify_until_num_points"])
    dynamic_limit = int(config["dynamic_densify_until_num_points"])
    reserve = len(checkpoints) * (
        static_limit * 640 + dynamic_limit * 800 + 230e6
    ) * 1.1 + 8e9
    free = shutil.disk_usage(arguments.output.parent).free
    if free < reserve:
        raise RuntimeError(
            f"insufficient space for six preserved checkpoints: {free} < {reserve}"
        )
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
    ).strip()
    if applications:
        raise RuntimeError("GPU has an existing compute process: " + applications)

    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.path.insert(0, str(arguments.binding_tools))
    sys.path.insert(0, str(arguments.engine))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if total_memory < int(plan["minimum_gpu_memory_bytes"]):
        raise RuntimeError("actual GPU memory is below the reviewed v3 admission floor")

    from omegaconf import OmegaConf
    import train
    from utils import v3_runtime_context
    from utils.v3_temporal import TEMPORAL_POLICY, TEMPORAL_VERSION

    if Path(train.__file__).resolve() != (arguments.engine / "train.py").resolve():
        raise RuntimeError("wrong train import")
    if train.V3_ENGINE_CONTRACT != METHOD_VERSION:
        raise RuntimeError("train.py is not the reviewed v3 engine")
    if v3_runtime_context.V3_METHOD_VERSION != METHOD_VERSION:
        raise RuntimeError("runtime context is not the reviewed v3 engine")
    if plan.get("common_v3_policy") != v3_runtime_context.V3_POLICY:
        raise RuntimeError("plan policy differs from the reviewed v3 runtime policy")
    if plan["background_temporal_policy"] != TEMPORAL_POLICY:
        raise RuntimeError("plan differs from the selective background temporal runtime")
    os.chdir(arguments.engine)

    config.update(
        model_path=str(arguments.output),
        start_checkpoint=None,
        seed=0,
        iterations=30000,
        checkpoint_iterations=checkpoints,
        save_iterations=checkpoints,
        test_iterations=checkpoints,
        v3_bundle=str(arguments.bundle),
        v3_zero_update_gate=arguments.mode == "gate",
        v3_arm=arguments.arm,
        v3_policy=dict(v3_runtime_context.V3_POLICY),
        v3_temporal_policy=dict(TEMPORAL_POLICY),
        v3_temporal_evidence=plan["background_temporal_evidence"],
        lambda_t_reg=0.0,
        lambda_v_reg=0.0,
        hybrid_nps_enabled=False,
    )
    arguments.output.mkdir()
    write(arguments.output / "effective_config.json", config)
    write(
        arguments.output / "v3_run_contract.json",
        {
            "method_version": METHOD_VERSION,
            "arm": arguments.arm,
            "mode": arguments.mode,
            "logical_budget": 30000,
            "second_stage_updates": 0,
            "same_seed": 0,
            "source_path": config["source_path"],
            "checkpoint_steps": checkpoints,
            "road_weights": plan["arms"][arguments.arm]["road_weights"],
            "full_dynamic_opacity_mse_weight": 100.0,
            "background_temporal_version": TEMPORAL_VERSION,
            "background_temporal_policy": TEMPORAL_POLICY,
            "background_temporal_evidence": plan["background_temporal_evidence"],
            "extra_optimizer_updates": 0,
            "free_bytes": free,
            "reserved_bytes": reserve,
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )

    import runtime_binding
    runtime_binding.bind_training_contract(train, config, read(arguments.readiness))
    from utils import paper_checkpoint

    original_save = paper_checkpoint.save_components

    def save_with_admission(path, iteration, models):
        remaining = sum(step >= iteration for step in checkpoints)
        estimate = (
            len(models["static"].get_xyz) * 640
            + len(models["dynamic"].get_xyz) * 800
            + 230e6
        ) * 1.1
        required = remaining * estimate + 4e9
        waiting = False
        while shutil.disk_usage(arguments.output).free < required:
            if not waiting:
                write(
                    arguments.output / "storage_admission.json",
                    {
                        "status": "WAITING_STORAGE_MEMORY_PRESERVED",
                        "completed_optimizer_steps": iteration,
                        "required_bytes": required,
                        "free_bytes": shutil.disk_usage(arguments.output).free,
                        "automatic_replay": False,
                    },
                )
                waiting = True
            time.sleep(30)
        if waiting:
            write(
                arguments.output / "storage_admission.json",
                {
                    "status": "STORAGE_ADMISSION_RECOVERED",
                    "completed_optimizer_steps": iteration,
                    "required_bytes": required,
                    "free_bytes": shutil.disk_usage(arguments.output).free,
                },
            )
        return original_save(path, iteration, models)

    paper_checkpoint.save_components = save_with_admission
    from utils.general_utils import seed_everything

    ZeroUpdateGateComplete = v3_runtime_context.ZeroUpdateGateComplete
    started = time.monotonic()
    status = {
        "status": "initializing",
        "method_version": METHOD_VERSION,
        "arm": arguments.arm,
        "mode": arguments.mode,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "iteration": 0,
        "logical_budget": 30000,
        "second_stage_updates": 0,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "GPU": torch.cuda.get_device_name(0),
        "quality_accepted": False,
    }
    write(arguments.output / "status.json", status)
    original_eval = train.complete_eval

    def observed_eval(*positional, **keyword):
        iteration = int(positional[1])
        schedule = positional[2]
        losses = positional[6]
        numbers = {
            key: float(value.detach()) if torch.is_tensor(value) else float(value)
            for key, value in losses.items()
        }
        if not all(math.isfinite(value) for value in numbers.values()):
            raise FloatingPointError("non-finite actual training scalar")
        context = v3_runtime_context.ACTIVE_CONTEXT
        if iteration == 1 or iteration % 100 == 0 or iteration in schedule:
            counts = dict(context.optimizer_steps)
            if any(counts.get(name) != iteration for name in ("static", "dynamic", "environment")):
                raise RuntimeError("optimizer update count disagrees with logical iteration")
            status.update(
                status="evaluating" if iteration in schedule else "training",
                iteration=iteration,
                wall_seconds=time.monotonic() - started,
                loss=numbers.get("loss"),
                psnr=numbers.get("psnr"),
                static_points=len(context.models[0].get_xyz),
                dynamic_points=len(context.models[1].get_xyz),
                optimizer_step_counts=counts,
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            write(arguments.output / "status.json", status)
        return original_eval(*positional, **keyword)

    train.complete_eval = observed_eval
    seed_everything(0)
    try:
        train.training(OmegaConf.create(config))
    except ZeroUpdateGateComplete:
        if arguments.mode != "gate":
            raise
        write(
            arguments.output / "gpu_gate_receipt.json",
            {
                "status": "PASSED_ACTUAL_CUDA_V3_FORWARD_BACKWARD_ZERO_UPDATES",
                "method_version": METHOD_VERSION,
                "arm": arguments.arm,
                "background_temporal_version": TEMPORAL_VERSION,
                "optimizer_updates": 0,
                "GPU": torch.cuda.get_device_name(),
                "actual_checks": read(arguments.output / "gpu_gate_details.json"),
                "same_fresh_seed_required_for_training": True,
                "quality_verified": False,
            },
        )
        status.update(
            status="zero_update_gate_passed",
            optimizer_step_counts=dict(v3_runtime_context.ACTIVE_CONTEXT.optimizer_steps),
        )
        write(arguments.output / "status.json", status)
        return
    except BaseException as error:
        context = v3_runtime_context.ACTIVE_CONTEXT
        status.update(
            status="failed_preserved_no_restart",
            error=repr(error),
            traceback=traceback.format_exc(),
            optimizer_step_counts=(dict(context.optimizer_steps) if context is not None else {}),
            wall_seconds=time.monotonic() - started,
            automatic_restart=False,
        )
        write(arguments.output / "status.json", status)
        raise

    if arguments.mode == "gate":
        raise RuntimeError("zero-update gate did not stop before Adam")
    for step in checkpoints:
        event = read(arguments.output / "checkpoint_events" / f"iteration_{step:06d}.json")
        if event.get("status") != "complete" or len(event.get("checkpoints", [])) != 4 or len(event.get("frames", [])) != 49:
            raise RuntimeError("six-node checkpoint/evaluation contract incomplete")
    counts = dict(v3_runtime_context.ACTIVE_CONTEXT.optimizer_steps)
    if any(counts.get(name) != 30000 for name in ("static", "dynamic", "environment")):
        raise RuntimeError("training returned without exactly 30K optimizer updates")
    status.update(
        status="complete",
        iteration=30000,
        wall_seconds=time.monotonic() - started,
        optimizer_step_counts=counts,
    )
    write(arguments.output / "status.json", status)
    write(
        arguments.output / "v3_training_returned.json",
        {
            "status": "TRAINING_RETURNED_FINAL_ASSET_AUDIT_PENDING",
            "method_version": METHOD_VERSION,
            "arm": arguments.arm,
            "quality_accepted": False,
            "full49_deletion_evaluation_still_required": True,
        },
    )


if __name__ == "__main__":
    main()

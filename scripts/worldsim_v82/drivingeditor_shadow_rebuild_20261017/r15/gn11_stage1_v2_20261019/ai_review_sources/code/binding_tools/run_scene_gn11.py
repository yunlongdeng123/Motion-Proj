#!/usr/bin/env python3
"""独立单scene GN11：默认CPU validate；--prepare可在gate pending时做小型隔离源，--run必须过双门。"""
import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time

from contracts import (REMOTE_BASE, build_config, completion_contract, disk_forecast, gate_contract,
                       input_inventory, read_json, readiness_contract, require, require_disk_space,
                       scene_spec, validate_gn11_config, write_new)
from prepare_followup import prepare, source_contract, verify_prepared

_ATTEMPTS = itertools.count()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    actions = result.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run", action="store_true")
    result.add_argument("--metadata-only", action="store_true",
                        help="仅本地审查scp小JSON；不认证实际文件/源码，不可prepare/run")
    result.add_argument("--scene-spec", required=True, type=Path)
    result.add_argument("--readiness-file", required=True, type=Path)
    result.add_argument("--reference-config", required=True, type=Path,
                        help="继承GN11候选/原effective配置，不是已加入followup字段的prepared cfg")
    result.add_argument("--source-repository", required=True, type=Path)
    result.add_argument("--matrix-tools-dir", required=True, type=Path,
                        help="当前run_matrix.py和memory_monitor.py目录；不调用其main")
    result.add_argument("--campaign-root", required=True, type=Path)
    result.add_argument("--gate-file", type=Path)
    result.add_argument("--verify-paired-steps", action="store_true")
    return result


def paths_from_args(args):
    return {key: str(getattr(args, key).resolve()) for key in
            ("scene_spec", "readiness_file", "reference_config", "source_repository",
             "matrix_tools_dir", "campaign_root")}


def validate(args):
    require(not args.metadata_only or not (args.prepare or args.run),
            "--metadata-only不能生成prepare或真实启动")
    paths = paths_from_args(args)
    spec = scene_spec(read_json(args.scene_spec))
    ready = read_json(args.readiness_file)
    ready_status = readiness_contract(ready, spec)
    reference = read_json(args.reference_config)
    validate_gn11_config(reference)
    config = build_config(reference, spec, args.campaign_root)
    gate = None
    gate_error = "gate file尚未提供"
    if args.gate_file:
        try:
            gate = gate_contract(args.gate_file, spec, reference, args.verify_paired_steps)
            gate_error = None
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            gate_error = str(error)
    if args.run:
        require(gate is not None, "GPU启动blocked: " + str(gate_error))
    inventory = None
    if not args.metadata_only:
        inventory = input_inventory(spec)
        source_contract(args.source_repository)
        require((args.matrix_tools_dir / "run_matrix.py").is_file()
                and (args.matrix_tools_dir / "memory_monitor.py").is_file(), "当前worker/内存监控源缺失")
    result = {"status": ("FOLLOWUP_METADATA_DRY_RUN_ONLY" if args.metadata_only
                         else "FOLLOWUP_CPU_VALIDATED_GPU_GATE_PENDING" if gate is None
                         else "FOLLOWUP_CPU_VALIDATED_GATE_RECHECKED"),
              "scene_id": spec["scene_id"], "arm": "GN11", "iterations": 30000,
              "checkpoint_nodes": [10000, 20000, 30000], "anchor_pose_frame": spec["anchor_frame"],
              "scene_cpu_status": ready_status, "input_inventory": inventory,
              "actual_files_checked": not args.metadata_only, "gate_passed": gate is not None,
              "gpu_started": False, "gpu_start_blocked_reason": gate_error,
              "actual_model_sgcm_binding_verified": False, "restoration_verified": False,
              "disk_forecast": disk_forecast(config, gate["gn11_event"] if gate else None)}
    return paths, spec, ready, reference, config, gate, result


def validate_for_run(args):
    paths, spec, ready, reference, config, gate, result = validate(args)
    require(sys.platform.startswith("linux"), "真实worker只在用户wm-vgpu-1011 Linux host运行")
    campaign = Path(paths["campaign_root"])
    require(campaign.parent == REMOTE_BASE / "campaign"
            and campaign != REMOTE_BASE / "campaign/hybrid30k_20261012",
            "真实followup必须使用共享BASE下独立campaign，不能改当前matrix")
    record = verify_prepared(paths, config, spec)
    require(not Path(config["model_path"]).exists(), "run输出已存在；拒绝覆盖或自动恢复")
    # 本函数在dispatch及worker再次调用；每次start按元数据预测完整保留成本。
    space = require_disk_space(campaign, disk_forecast(config, gate["gn11_event"]))
    return paths, spec, ready, reference, config, gate, record, space


def run(args):
    paths, spec, ready, reference, config, gate, record, space = validate_for_run(args)
    support = Path(record["training_support"])
    sys.path.insert(0, str(support))
    module_spec = importlib.util.spec_from_file_location("followup_private_matrix_worker", support / "matrix_worker.py")
    worker = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(worker)
    worker.BASE = REMOTE_BASE
    worker.ROOT = Path(paths["campaign_root"])
    worker.REPOSITORY = Path(record["training_source"])
    # 原worker持有共享gpu_slot.lock并检查GPU compute apps；这里只绑定单GN11。
    def config_callback(arm):
        require(arm == "GN11", "新runner禁止C00/G10/N01/四臂队列")
        return validate_for_run(args)[4]

    def prepared_callback():
        validate_for_run(args)

    from runtime_binding import bind_training_contract
    worker.check_config = config_callback
    worker.check_prepared = prepared_callback
    worker.bind_training_contract = lambda training, cfg: bind_training_contract(training, cfg, ready)
    worker.validate_completion = lambda output, cfg: completion_contract(output, cfg)
    record_start_attempt(worker.ROOT, spec, space)
    for key, value in {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                       "PYTHONDONTWRITEBYTECODE": "1",
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}.items():
        os.environ.setdefault(key, value)
    worker.run_arm("GN11")


def record_start_attempt(campaign, spec, space):
    # 共享slot忙而output未创建时，后续attempt可重试；旧audit receipt保留。
    name = "%s_disk_before_start_%d_%d_%d.json" % (spec["scene_id"], os.getpid(), time.time_ns(), next(_ATTEMPTS))
    path = Path(campaign) / "training_metadata" / name
    write_new(path, dict(space, scene_id=spec["scene_id"], attempt_pid=os.getpid(), gpu_started=False))
    return path


def main():
    args = parser().parse_args()
    try:
        if args.run:
            run(args)
            return
        paths, spec, ready, reference, config, gate, result = validate(args)
        if args.prepare:
            receipt = prepare(paths, config, spec, result["scene_cpu_status"])
            result.update(status=receipt["status"], preparation=receipt,
                          gpu_started=False,
                          gpu_start_blocked_reason="真实--run仍需GN11 gate passed、GPU ownership及actual model/guard绑定")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FOLLOWUP_REJECTED", "error": str(error), "gpu_started": False},
                         ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

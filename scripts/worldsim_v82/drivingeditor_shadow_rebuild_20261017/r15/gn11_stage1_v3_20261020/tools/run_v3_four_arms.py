"""在用户明确 GPU 授权后，按固定顺序运行 GN11 v3 四臂。

本控制器默认拒绝执行。它要求命令行显式开关、独立审批文件、完整磁盘
准入以及空输出目录；任一 gate/训练失败即保存现场并停止，不自动重试。
"""

from pathlib import Path
import argparse
import datetime
import fcntl
import json
import os
import shutil
import subprocess
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


def command(campaign, plan, arm, mode, output, gate=None):
    result = [
        str(plan["python"]),
        str(campaign / "run_v3_arm.py"),
        "--engine", str(campaign / "engine"),
        "--output", str(output),
        "--bundle", str(plan["input_bundle"]),
        "--baseline-config", str(plan["baseline_config"]),
        "--readiness", str(plan["readiness"]),
        "--binding-tools", str(plan["binding_tools"]),
        "--gpu-lock", str(plan["gpu_lock"]),
        "--plan", str(campaign / "plan.json"),
        "--arm", arm,
        "--mode", mode,
    ]
    if gate is not None:
        result.extend(["--gpu-gate", str(gate)])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--execute-after-user-gpu-approval",
        action="store_true",
        help="required acknowledgement; still does not replace GPU_APPROVAL.json",
    )
    arguments = parser.parse_args()
    campaign = arguments.campaign.resolve()
    if not arguments.execute_after_user_gpu_approval:
        raise RuntimeError("explicit post-review GPU execution flag is required")

    plan = read(campaign / "plan.json")
    if plan.get("method_version") != METHOD_VERSION:
        raise RuntimeError("campaign plan is not the reviewed v3 method")
    approval_path = campaign / "GPU_APPROVAL.json"
    if not approval_path.is_file():
        raise RuntimeError("GPU_APPROVAL.json is absent; user has not authorized execution")
    approval = read(approval_path)
    if (
        approval.get("status") != "USER_AUTHORIZED_GPU_V3"
        or approval.get("method_version") != METHOD_VERSION
        or approval.get("scope") != "zero_update_gates_and_four_fresh_30k"
    ):
        raise RuntimeError("GPU approval does not match this method and four-arm scope")

    expected = list(plan["arm_order"])
    if expected != ["full", "real_only", "prior_only", "ownership_only"]:
        raise RuntimeError("reviewed arm order changed")
    if int(plan["max_concurrent_training"]) != 1:
        raise RuntimeError("v3 requires sequential single-worker execution")
    free = shutil.disk_usage(campaign).free
    required = int(plan["minimum_campaign_free_bytes"])
    if free < required:
        write(
            campaign / "status.json",
            {
                "status": "WAITING_STORAGE_BEFORE_GPU_EXECUTION",
                "method_version": METHOD_VERSION,
                "free_bytes": free,
                "required_free_bytes": required,
                "training_updates": 0,
                "GPU_used": False,
                "quality_accepted": False,
            },
        )
        raise RuntimeError(f"campaign needs {required} free bytes; found {free}")

    lock_handle = (campaign / "controller.lock").open("a+")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registry_path = campaign / "four_arm_registry.json"
    if registry_path.exists():
        raise FileExistsError("controller registry already exists; audit before any continuation")
    for arm in expected:
        arm_root = campaign / "arms" / arm
        if (arm_root / "gate_001").exists() or (arm_root / "run30k").exists():
            raise FileExistsError(f"non-fresh output exists for {arm}")

    registry = {
        "status": "RUNNING_SEQUENTIALLY",
        "method_version": METHOD_VERSION,
        "arm_order": expected,
        "max_concurrent_training": 1,
        "current_arm": None,
        "completed_arms": [],
        "training_updates_planned": 120000,
        "automatic_restart": False,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    write(registry_path, registry)
    write(campaign / "status.json", registry)

    try:
        for arm in expected:
            registry.update(current_arm=arm, phase="zero_update_cuda_gate")
            write(registry_path, registry)
            write(campaign / "status.json", registry)
            arm_root = campaign / "arms" / arm
            gate_output = arm_root / "gate_001"
            gate_log = arm_root / "gate_001.log"
            with gate_log.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    command(campaign, plan, arm, "gate", gate_output),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                    cwd=campaign,
                )
            gate_receipt = gate_output / "gpu_gate_receipt.json"
            gate = read(gate_receipt)
            if gate.get("optimizer_updates") != 0:
                raise RuntimeError(f"{arm} CUDA gate changed model state")

            registry.update(phase="fresh_stage1_30k")
            write(registry_path, registry)
            write(campaign / "status.json", registry)
            run_output = arm_root / "run30k"
            train_log = arm_root / "run30k.log"
            with train_log.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    command(
                        campaign,
                        plan,
                        arm,
                        "train",
                        run_output,
                        gate=gate_receipt,
                    ),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                    cwd=campaign,
                )
            terminal = read(run_output / "status.json")
            if terminal.get("status") != "complete" or terminal.get("iteration") != 30000:
                raise RuntimeError(f"{arm} did not finish the exact 30K contract")
            registry["completed_arms"].append(arm)
            registry.update(
                current_arm=None,
                phase="between_arms",
                completed_optimizer_updates=30000 * len(registry["completed_arms"]),
            )
            write(registry_path, registry)
            write(campaign / "status.json", registry)
    except BaseException as error:
        registry.update(
            status="FAILED_PRESERVED_NO_AUTOMATIC_RESTART",
            error=repr(error),
            traceback=traceback.format_exc(),
            failed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            automatic_restart=False,
        )
        write(registry_path, registry)
        write(campaign / "status.json", registry)
        raise

    registry.update(
        status="FOUR_ARMS_TRAINING_COMPLETE_EVALUATION_PENDING",
        current_arm=None,
        phase="full49_evaluation_pending",
        completed_optimizer_updates=120000,
        quality_accepted=False,
        human_verdict=None,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    write(registry_path, registry)
    write(campaign / "status.json", registry)


if __name__ == "__main__":
    main()

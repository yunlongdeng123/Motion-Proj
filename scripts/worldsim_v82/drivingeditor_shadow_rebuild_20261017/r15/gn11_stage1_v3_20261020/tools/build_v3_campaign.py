"""从已冻结 v2 engine 构建隔离的 GN11 v3 campaign；不启动 GPU。"""

from pathlib import Path
import argparse
import datetime
import json
import shutil
import subprocess
import sys


METHOD_VERSION = "GN11_STAGE1_V3_20261020"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def copy_overlay(overlay, engine):
    copied = []
    for source in sorted(overlay.rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        relative = source.relative_to(overlay)
        destination = engine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(relative))
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source-engine", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--approval-template", type=Path, required=True)
    parser.add_argument('--road-builder', type=Path, required=True)
    arguments = parser.parse_args()
    plan = read(arguments.plan)
    if plan.get("method_version") != METHOD_VERSION:
        raise RuntimeError("wrong v3 plan")
    if str(arguments.campaign.resolve()) != plan["campaign"]:
        raise RuntimeError("campaign path must match the reviewed plan")
    if arguments.campaign.exists():
        raise FileExistsError("campaign already exists; never overwrite prepared or trained work")
    for required in (
        arguments.source_engine / "train.py",
        arguments.source_engine / "gaussian_renderer" / "__init__.py",
        arguments.overlay / "train.py",
        arguments.overlay / "utils" / "v3_runtime_context.py",
        arguments.overlay / "utils" / "v3_regularizer_contract.py",
        arguments.overlay / "utils" / "v3_multiview_road.py",
        arguments.overlay / "scene" / "gaussian_model.py",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    arguments.campaign.mkdir(parents=True)
    engine = arguments.campaign / "engine"
    shutil.copytree(
        arguments.source_engine,
        engine,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )

    copied = copy_overlay(arguments.overlay, engine)
    shutil.copy2(arguments.plan, arguments.campaign / "plan.json")
    shutil.copy2(arguments.runner, arguments.campaign / "run_v3_arm.py")
    shutil.copy2(arguments.controller, arguments.campaign / "run_v3_four_arms.py")
    shutil.copy2(arguments.road_builder, arguments.campaign / 'prepare_multiview_road.py')
    shutil.copy2(arguments.readme, arguments.campaign / "README.md")
    shutil.copy2(arguments.architecture, arguments.campaign / "architecture.svg")
    shutil.copy2(arguments.readme.with_name('road_architecture.svg'), arguments.campaign/'road_architecture.svg')
    shutil.copy2(arguments.readme.with_name('supervision_architecture.svg'), arguments.campaign/'supervision_architecture.svg')
    shutil.copy2(arguments.readme.with_name('supervision_short_controls.json'), arguments.campaign/'supervision_short_controls.json')
    shutil.copy2(arguments.runner.with_name('report_supervision_audit.py'), arguments.campaign/'report_supervision_audit.py')
    shutil.copy2(arguments.runner.with_name('report_render_audit.py'), arguments.campaign/'report_render_audit.py')
    shutil.copy2(arguments.runner.with_name('validate_render_audit_cpu.py'), arguments.campaign/'validate_render_audit_cpu.py')
    shutil.copy2(arguments.readme.with_name('render_audit_architecture.svg'), arguments.campaign/'render_audit_architecture.svg')
    shutil.copy2(arguments.readme.with_name('road_review_architecture.svg'), arguments.campaign/'road_review_architecture.svg')
    shutil.copy2(arguments.readme.with_name('pair_schedule_architecture.svg'), arguments.campaign/'pair_schedule_architecture.svg')
    for name in ['road_source_audit.py', 'road_source_counterfactual.py', 'v3_gpu_authorization.py', 'upgrade_road_input.py',
                 'v3_pair_runtime.py', 'run_v3_pairs.py', 'check_v3_progress.py']:
        shutil.copy2(arguments.runner.with_name(name), arguments.campaign/name)
    shutil.copy2(
        arguments.evaluation_contract,
        arguments.campaign / "evaluation_contract.json",
    )
    shutil.copy2(
        arguments.approval_template,
        arguments.campaign / "GPU_APPROVAL_REQUIRED.json",
    )
    shutil.copytree(
        arguments.tests,
        arguments.campaign / "tests",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )

    subprocess.run([sys.executable, str(arguments.campaign/'prepare_multiview_road.py'),
                    '--engine', str(engine), '--engineering', plan['road_engineering_source'],
                    '--output', str(Path(plan['input_bundle']).parent)], check=True)
    arms_root = arguments.campaign / "arms"
    arms_root.mkdir()
    for arm in plan["arm_order"]:
        arm_root = arms_root / arm
        arm_root.mkdir()
        write(
            arm_root / "READY.json",
            {
                "status": "READY_FOR_REVIEW_GPU_NOT_AUTHORIZED",
                "method_version": METHOD_VERSION,
                "background_temporal_version": plan["background_temporal_version"],
                "arm": arm,
                "road_weights": plan["arms"][arm]["road_weights"],
                "fresh_updates_planned": 30000,
                "updates_executed": 0,
                "gate_executed": False,
                "training_started": False,
                "quality_accepted": False,
                "human_verdict": None,
            },
        )

    free_bytes = shutil.disk_usage(arguments.campaign).free
    required_free_bytes = int(plan["minimum_campaign_free_bytes"])
    storage_ready = free_bytes >= required_free_bytes
    receipt = {
        "status": "DEPLOYED_CPU_ONLY_GPU_NOT_AUTHORIZED",
        "method_version": METHOD_VERSION,
        "background_temporal_version": plan["background_temporal_version"],
        "background_temporal_policy": plan["background_temporal_policy"],
        "campaign": str(arguments.campaign),
        "source_engine": str(arguments.source_engine),
        "overlay_source": str(arguments.overlay),
        "overlay_files": copied,
        "arms": list(plan["arm_order"]),
        "training_updates": 0,
        "GPU_used": False,
        "controller_started": False,
        "free_bytes_at_deployment": free_bytes,
        "required_free_bytes_before_four_arm_controller": required_free_bytes,
        "storage_ready_for_four_arms": storage_ready,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    write(arguments.campaign / "deployment_receipt.json", receipt)
    write(
        arguments.campaign / "status.json",
        {
            "status": (
                "WAITING_USER_REVIEW_AND_GPU_APPROVAL"
                if storage_ready
                else "WAITING_USER_REVIEW_GPU_APPROVAL_AND_STORAGE"
            ),
            "method_version": METHOD_VERSION,
            "background_temporal_version": plan["background_temporal_version"],
            "training_updates": 0,
            "GPU_used": False,
            "active_workers": [],
            "free_bytes_at_deployment": free_bytes,
            "required_free_bytes_before_four_arm_controller": required_free_bytes,
            "storage_ready_for_four_arms": storage_ready,
            "quality_accepted": False,
            "human_verdict": None,
        },
    )
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

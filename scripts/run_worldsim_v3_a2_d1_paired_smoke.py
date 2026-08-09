#!/usr/bin/env python
"""顺序运行 scene-0230 A2-D1 的 D0/D1 配对工程 smoke。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import OmegaConf

from motion_proj.worldsim_v3.actor_quota import validate_a2_d1_contract
from scripts.prepare_worldsim_v3_a2_d1_drivestudio import (
    sha256_file,
    verify_patched_tree,
)
from scripts.run_worldsim_v3_a0_scene import run_stage
from scripts.run_worldsim_v3_a0_smoke import (
    atomic_json,
    command_output,
    now,
)


PROJECT = Path("/root/autodl-tmp/motion_proj")
DRIVESTUDIO = Path(
    "/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-d1-r5"
)
DRIVESTUDIO_PYTHON = Path("/root/autodl-tmp/envs/drivestudio/bin/python")
SOURCE_CONFIG = Path(
    "/root/autodl-tmp/runs/worldsim_v3/WS-V3-A0-NATIVE-BASELINE-01/"
    "20260805T171624Z__scene0230-reuse-eval-s0-r1/assets/"
    "native_checkpoint/config.yaml"
)
VARIANTS = ("d0-native", "d1-actor-quota")
_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value.update(
        {
            "PYTHONPATH": f"{PROJECT}:{DRIVESTUDIO}",
            "WANDB_MODE": "disabled",
            "HF_HOME": "/root/autodl-tmp/hf_cache",
            "HF_HUB_CACHE": "/root/autodl-tmp/hf_cache/hub",
            "HF_ENDPOINT": "https://hf-mirror.com",
            "TORCH_HOME": "/root/autodl-tmp/cache/torch",
            "XDG_CACHE_HOME": "/root/autodl-tmp/cache/xdg",
        }
    )
    return value


def materialize(
    *, contract: Path, variant: str, num_iters: int, output: Path
) -> list[str]:
    command = [
        str(DRIVESTUDIO_PYTHON),
        str(PROJECT / "scripts/materialize_worldsim_v3_a2_d1_config.py"),
        "--source-config",
        str(SOURCE_CONFIG),
        "--contract",
        str(contract),
        "--variant",
        variant,
        "--num-iters",
        str(num_iters),
        "--output",
        str(output),
    ]
    subprocess.run(
        command,
        cwd=PROJECT,
        env=environment(),
        check=True,
    )
    return command


def normalized_paired_config(config: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    payload["model"]["RigidNodes"]["ctrl"]["a2_actor_quota"][
        "enabled"
    ] = "paired-variant"
    payload["worldsim_v3"]["variant"] = "paired-variant"
    return payload


def build_train_command(
    run_dir: Path, config: Path, variant: str, num_iters: int
) -> tuple[list[str], Path]:
    run_name = (
        f"scene0230_a2_{variant.replace('-', '_')}_s0_i{num_iters}"
    )
    checkpoint = (
        run_dir
        / "work_dirs"
        / "worldsim_v3_a2"
        / run_name
        / "checkpoint_final.pth"
    )
    command = [
        str(DRIVESTUDIO_PYTHON),
        str(DRIVESTUDIO / "tools/train.py"),
        "--config_file",
        str(config),
        "--output_root",
        str(run_dir / "work_dirs"),
        "--project",
        "worldsim_v3_a2",
        "--run_name",
        run_name,
    ]
    return command, checkpoint


def native_tensors_are_finite(models: dict[str, Any]) -> bool:
    def visit(value: Any) -> bool:
        if torch.is_tensor(value):
            return not value.is_floating_point() or bool(
                torch.isfinite(value).all().item()
            )
        if isinstance(value, dict):
            return all(
                visit(child)
                for key, child in value.items()
                if not str(key).startswith("worldsim_")
            )
        if isinstance(value, (list, tuple)):
            return all(visit(child) for child in value)
        return True

    return visit(models)


def checkpoint_audit(
    checkpoint: Path,
    *,
    variant: str,
    num_iters: int,
    frozen_initial_counts: list[int],
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu")
    models = payload["models"]
    rigid = models["RigidNodes"]
    background = models["Background"]
    actor_ids = rigid["points_ids"][..., 0].to(torch.long)
    actor_counts = torch.bincount(
        actor_ids, minlength=len(frozen_initial_counts)
    )
    quota_state = rigid.get("worldsim_a2_actor_quota")
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "step": int(payload["step"]),
        "variant": variant,
        "rigid_actor_counts": actor_counts.tolist(),
        "rigid_total": int(actor_counts.sum().item()),
        "background_total": int(background["_means"].shape[0]),
        "rigid_ancestry_present": "worldsim_a2_ancestry" in rigid,
        "background_ancestry_present": (
            "worldsim_a2_ancestry" in background
        ),
        "actor_quota_present": quota_state is not None,
        "native_tensors_finite": native_tensors_are_finite(models),
    }
    if quota_state is not None:
        initial = quota_state["initial_counts"].to(torch.long)
        minimum = quota_state["minimum_counts"].to(torch.long)
        maximum = quota_state["maximum_counts"].to(torch.long)
        result.update(
            {
                "quota_initial_counts": initial.tolist(),
                "quota_minimum_counts": minimum.tolist(),
                "quota_maximum_counts": maximum.tolist(),
                "quota_initial_total": int(initial.sum().item()),
                "quota_minimum_total": int(minimum.sum().item()),
                "quota_maximum_total": int(maximum.sum().item()),
                "quota_counters": quota_state["counters"],
                "quota_maximum_respected": bool(
                    torch.all(actor_counts <= maximum).item()
                ),
                "quota_initial_matches_frozen_scene": (
                    initial.tolist() == frozen_initial_counts
                ),
            }
        )
    expected_quota = variant == "d1-actor-quota"
    valid = (
        result["step"] == num_iters
        and result["rigid_ancestry_present"]
        and result["background_ancestry_present"]
        and result["actor_quota_present"] == expected_quota
        and result["native_tensors_finite"]
    )
    if expected_quota:
        valid = (
            valid
            and result["quota_maximum_respected"]
            and result["quota_initial_matches_frozen_scene"]
            and result["quota_initial_total"] == 75002
            and result["quota_minimum_total"] == 37504
            and result["quota_maximum_total"] == 180013
        )
    result["valid"] = bool(valid)
    del payload, models, rigid, background, actor_ids, actor_counts
    gc.collect()
    if not result["valid"]:
        raise RuntimeError(f"checkpoint audit failed: {result}")
    return result


def provenance_fingerprint(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "background_lidar_sample": payload["background_lidar_sample"],
        "instance_lidar_samples": payload["instance_lidar_samples"],
        "initialized_gaussians": payload["initialized_gaussians"],
        "rng_reset": payload["rng_reset"],
        "truth_tier": payload["truth_tier"],
    }


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=PROJECT / "configs/worldsim_v3/a2_d1_v1.yaml",
    )
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--train-timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)

    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    validate_a2_d1_contract(contract)
    if args.num_iters != int(contract["paired_smoke"]["num_iters"]):
        raise ValueError("paired smoke iteration budget drift")
    if not SOURCE_CONFIG.is_file():
        raise FileNotFoundError(SOURCE_CONFIG)
    free_gib = shutil.disk_usage("/root/autodl-tmp").free / (1024**3)
    minimum_free = float(
        contract["resource_contract"]["minimum_free_disk_gib"]
    )
    if free_gib < minimum_free:
        raise RuntimeError(
            f"free disk {free_gib:.2f} GiB is below {minimum_free:.2f} GiB"
        )

    project_commit = command_output("git", "rev-parse", "HEAD", cwd=PROJECT)
    project_status = command_output(
        "git", "status", "--short", cwd=PROJECT
    ).splitlines()
    compatibility_patch = PROJECT / contract["drivestudio"][
        "compatibility_patch"
    ]
    instrumentation_patch = PROJECT / contract["drivestudio"][
        "instrumentation_patch"
    ]
    quota_patch = PROJECT / contract["drivestudio"]["quota_patch"]
    worktree = verify_patched_tree(
        DRIVESTUDIO,
        compatibility_patch,
        instrumentation_patch,
        quota_patch,
    )

    _ACTIVE_RUN_DIR = args.run_dir
    for name in (
        "artifacts",
        "environment",
        "logs",
        "source_snapshot",
        "stages",
    ):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )
    shutil.copy2(args.contract, args.run_dir / "resolved.yaml")
    atomic_json(args.run_dir / "artifacts/worktree.json", worktree)

    sources = (
        PROJECT / "scripts/run_worldsim_v3_a2_d1_paired_smoke.py",
        PROJECT / "scripts/materialize_worldsim_v3_a2_d1_config.py",
        PROJECT / "scripts/prepare_worldsim_v3_a2_d1_drivestudio.py",
        PROJECT / "scripts/smoke_worldsim_v3_a2_d1_quota.py",
        PROJECT / "scripts/smoke_worldsim_v3_a2_ancestry.py",
        PROJECT / "motion_proj/worldsim_v3/actor_quota.py",
        PROJECT / "motion_proj/worldsim_v3/gaussian_ancestry.py",
        args.contract,
        compatibility_patch,
        instrumentation_patch,
        quota_patch,
    )
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(PROJECT)
        destination = args.run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = sha256_file(source)

    configs = {}
    materialize_commands = {}
    for variant in VARIANTS:
        output = args.run_dir / "artifacts" / f"{variant}.yaml"
        materialize_commands[variant] = materialize(
            contract=args.contract,
            variant=variant,
            num_iters=args.num_iters,
            output=output,
        )
        configs[variant] = output
    matched_configs = normalized_paired_config(
        configs[VARIANTS[0]]
    ) == normalized_paired_config(configs[VARIANTS[1]])
    if not matched_configs:
        raise RuntimeError("D0/D1 materialized configs are not budget-matched")

    synthetic_outputs = {
        "module_off": args.run_dir
        / "artifacts/module_off_equivalence.json",
        "quota": args.run_dir / "artifacts/quota_integration.json",
    }
    synthetic_commands = {
        "module_off": [
            str(DRIVESTUDIO_PYTHON),
            str(PROJECT / "scripts/smoke_worldsim_v3_a2_ancestry.py"),
            "--config",
            str(PROJECT / "configs/worldsim_v3/a2_instrumentation_v1.yaml"),
            "--output",
            str(synthetic_outputs["module_off"]),
        ],
        "quota": [
            str(DRIVESTUDIO_PYTHON),
            str(PROJECT / "scripts/smoke_worldsim_v3_a2_d1_quota.py"),
            "--config",
            str(args.contract),
            "--output",
            str(synthetic_outputs["quota"]),
        ],
    }
    for name, command in synthetic_commands.items():
        run_stage(
            run_dir=args.run_dir,
            stage=f"synthetic_{name}",
            command=command,
            cwd=DRIVESTUDIO,
            environment=environment(),
            validate=lambda output=synthetic_outputs[name]: (
                output.is_file()
                and json.loads(output.read_text(encoding="utf-8"))[
                    "status"
                ]
                == "done",
                {"output": str(output)},
            ),
            timeout_seconds=300,
        )

    manifest = {
        "schema_version": 1,
        "task_id": contract["task_id"],
        "component": "A2-D1 paired engineering smoke",
        "scene": "scene-0230",
        "variants": list(VARIANTS),
        "seed": 0,
        "num_iters": args.num_iters,
        "formal": False,
        "matched_configs": matched_configs,
        "project_commit": project_commit,
        "project_status": project_status,
        "source_hashes": source_hashes,
        "materialize_commands": materialize_commands,
        "started_at": now(),
    }
    atomic_json(args.run_dir / "manifest.json", manifest)

    audits = {}
    provenance = {}
    train_resources = {}
    frozen_counts = contract["frozen_scene_0230_reference"][
        "initial_actor_counts"
    ]
    for variant in VARIANTS:
        command, checkpoint = build_train_command(
            args.run_dir, configs[variant], variant, args.num_iters
        )
        provenance_path = (
            args.run_dir / "artifacts" / f"{variant}-init-provenance.json"
        )
        stage_environment = environment()
        stage_environment["WORLDSIM_V3_INIT_PROVENANCE"] = str(
            provenance_path
        )
        stage_environment["WORLDSIM_V3_INIT_SEED"] = "0"
        stage = run_stage(
            run_dir=args.run_dir,
            stage=f"train_{variant.replace('-', '_')}_{args.num_iters}",
            command=command,
            cwd=DRIVESTUDIO,
            environment=stage_environment,
            validate=lambda checkpoint=checkpoint, provenance_path=provenance_path: (
                checkpoint.is_file()
                and checkpoint.stat().st_size > 0
                and provenance_path.is_file()
                and provenance_path.stat().st_size > 0,
                {
                    "checkpoint": str(checkpoint),
                    "provenance": str(provenance_path),
                },
            ),
            timeout_seconds=args.train_timeout_seconds,
        )
        audits[variant] = checkpoint_audit(
            checkpoint,
            variant=variant,
            num_iters=args.num_iters,
            frozen_initial_counts=frozen_counts,
        )
        provenance[variant] = provenance_fingerprint(provenance_path)
        train_resources[variant] = {
            key: stage[key]
            for key in (
                "duration_seconds",
                "peak_gpu_memory_mib_sampled",
                "peak_gpu_memory_mib_torch_log",
                "peak_cgroup_memory_bytes",
            )
        }
        atomic_json(
            args.run_dir / "artifacts" / f"{variant}-checkpoint-audit.json",
            audits[variant],
        )

    provenance_matched = provenance[VARIANTS[0]] == provenance[VARIANTS[1]]
    if not provenance_matched:
        raise RuntimeError("D0/D1 initialization provenance mismatch")
    summary = {
        "status": "done",
        "task_id": contract["task_id"],
        "component": "A2-D1 paired engineering smoke",
        "scene": "scene-0230",
        "seed": 0,
        "num_iters": args.num_iters,
        "formal": False,
        "project_commit": project_commit,
        "resolved_config_sha256": sha256_file(
            args.run_dir / "resolved.yaml"
        ),
        "quota_patch_sha256": worktree["quota_patch_sha256"],
        "matched_configs": matched_configs,
        "initialization_provenance_matched": provenance_matched,
        "synthetic_module_off": json.loads(
            synthetic_outputs["module_off"].read_text(encoding="utf-8")
        ),
        "synthetic_quota": json.loads(
            synthetic_outputs["quota"].read_text(encoding="utf-8")
        ),
        "checkpoint_audits": audits,
        "train_resources": train_resources,
        "completed_at": now(),
    }
    atomic_json(args.run_dir / "summary.json", summary)
    manifest["status"] = "done"
    manifest["completed_at"] = summary["completed_at"]
    atomic_json(args.run_dir / "manifest.json", manifest)
    atomic_json(
        args.run_dir / "terminal.json",
        {
            "status": "done",
            "updated_at": summary["completed_at"],
            "failure": None,
        },
    )
    _TERMINAL_FINAL = True
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if (
            _ACTIVE_RUN_DIR is not None
            and _ACTIVE_RUN_DIR.is_dir()
            and not _TERMINAL_FINAL
        ):
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "updated_at": now(),
                    "failure": {
                        "code": "A2_D1_PAIRED_SMOKE_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
            )
        raise

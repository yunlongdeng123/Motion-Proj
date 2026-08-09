#!/usr/bin/env python
"""Run the frozen scene-0230 A2-D1 D0/D1 formal pair sequentially."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

PROJECT = Path("/root/autodl-tmp/motion_proj")
sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v3.a2_formal import (
    VARIANTS,
    compare_view,
    select_matched_checkpoint,
    validate_a2_d1_formal_contract,
)
from motion_proj.worldsim_v3.actor_quota import validate_a2_d1_contract
from scripts.prepare_worldsim_v3_a2_d1_drivestudio import (
    sha256_file,
    verify_patched_tree,
)
from scripts.run_worldsim_v3_a0_scene import compact_actor, run_stage
from scripts.run_worldsim_v3_a0_smoke import (
    atomic_json,
    command_output,
    now,
    resource_sample,
)
from scripts.run_worldsim_v3_a2_d1_paired_smoke import checkpoint_audit


DRIVESTUDIO = Path(
    "/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-d1-r5"
)
DRIVESTUDIO_PYTHON = Path("/root/autodl-tmp/envs/drivestudio/bin/python")
SCENE_NAME = "scene-0230"
SCENE_INDEX = 179
HIGH_TOKEN = "af663976db5e412e83db033d309c5c29"
BOUNDARY_TOKEN = "18c7f0c5fa6b49449f71c9dbae5c31d4"
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


def _clean_relevant_sources(relative_paths: tuple[str, ...]) -> None:
    process = subprocess.run(
        ["git", "status", "--porcelain", "--", *relative_paths],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    )
    if process.stdout.strip():
        raise RuntimeError(
            "formal runtime/config/test sources are not committed:\n"
            + process.stdout
        )


def normalized_paired_config(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    payload["model"]["RigidNodes"]["ctrl"]["a2_actor_quota"][
        "enabled"
    ] = "paired-variant"
    payload["worldsim_v3"]["variant"] = "paired-variant"
    return payload


def materialize(
    *,
    source_config: Path,
    base_contract: Path,
    variant: str,
    num_iters: int,
    checkpoint_interval: int,
    output: Path,
) -> list[str]:
    command = [
        str(DRIVESTUDIO_PYTHON),
        str(PROJECT / "scripts/materialize_worldsim_v3_a2_d1_config.py"),
        "--source-config",
        str(source_config),
        "--contract",
        str(base_contract),
        "--variant",
        variant,
        "--num-iters",
        str(num_iters),
        "--stage",
        "formal",
        "--checkpoint-interval",
        str(checkpoint_interval),
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=PROJECT, env=environment(), check=True)
    return command


def build_train_command(
    run_dir: Path, config: Path, variant: str, num_iters: int
) -> tuple[list[str], Path]:
    run_name = f"scene0230_a2_{variant.replace('-', '_')}_formal_s0_i{num_iters}"
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


def checkpoint_grid(
    final_checkpoint: Path,
    *,
    steps: list[int],
    variant: str,
    frozen_initial_counts: list[int],
    arm_started_epoch: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    final_step = max(steps)
    for step in steps:
        checkpoint = (
            final_checkpoint
            if step == final_step
            else final_checkpoint.parent / f"checkpoint_{step:05d}.pth"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        audit = checkpoint_audit(
            checkpoint,
            variant=variant,
            num_iters=step,
            frozen_initial_counts=frozen_initial_counts,
        )
        result.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": audit["checkpoint_sha256"],
                "rigid_gaussians": audit["rigid_total"],
                "background_gaussians": audit["background_total"],
                "total_gaussians": audit["rigid_total"]
                + audit["background_total"],
                "elapsed_to_checkpoint_seconds": max(
                    0.0, checkpoint.stat().st_mtime - arm_started_epoch
                ),
                "audit": audit,
            }
        )
    return result


def _checkpoint_source_contract(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": audit["checkpoint"],
        "exists": True,
        "bytes": audit["checkpoint_bytes"],
        "sha256": audit["checkpoint_sha256"],
        "step": audit["step"],
        "background_gaussians": audit["background_total"],
        "rigid_gaussians": audit["rigid_total"],
    }


def evaluate_checkpoint(
    *,
    run_dir: Path,
    view_name: str,
    checkpoint_audit_payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    safe_name = view_name.replace("-", "_")
    output = run_dir / "artifacts" / "evaluations" / view_name
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(checkpoint_audit_payload["checkpoint"])
    checkpoint_sha = checkpoint_audit_payload["checkpoint_sha256"]
    registry = output / "actor_registry.json"
    registry_command = [
        str(DRIVESTUDIO_PYTHON),
        str(PROJECT / "scripts/build_dr_v2_drivestudio_registry.py"),
        "--checkpoint",
        str(checkpoint),
        "--drivestudio-root",
        str(DRIVESTUDIO),
        "--scene-name",
        SCENE_NAME,
        "--selected-token",
        HIGH_TOKEN,
        "--output",
        str(registry),
    ]
    registry_stage = run_stage(
        run_dir=run_dir,
        stage=f"registry_{safe_name}",
        command=registry_command,
        cwd=PROJECT,
        environment=environment(),
        validate=lambda: (
            registry.is_file() and registry.stat().st_size > 0,
            {"registry": str(registry)},
        ),
        timeout_seconds=900,
    )
    registry_payload = json.loads(registry.read_text(encoding="utf-8"))
    if registry_payload.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("actor registry checkpoint SHA mismatch")
    by_token = {
        row["instance_token"]: row for row in registry_payload.get("actors", [])
    }
    selected = {
        "high-support": compact_actor(by_token.get(HIGH_TOKEN)),
        "boundary-support": compact_actor(by_token.get(BOUNDARY_TOKEN)),
    }
    if any(
        row is None or row.get("availability") != "available"
        for row in selected.values()
    ):
        raise RuntimeError(f"formal actor is unavailable: {selected}")

    eval_dir = checkpoint.parent / "metrics_eval"
    video_dir = checkpoint.parent / "videos_eval"
    metrics_before = set(eval_dir.glob("images_test_*.json"))
    videos_before = set(video_dir.glob("test_set_*.mp4"))
    eval_command = [
        str(DRIVESTUDIO_PYTHON),
        str(DRIVESTUDIO / "tools/eval.py"),
        "--resume_from",
        str(checkpoint),
        "--render_video_postfix",
        f"a2_{safe_name}",
        "render.render_test=true",
        "render.render_full=false",
        "render.render_novel=null",
    ]

    def validate_eval() -> tuple[bool, dict[str, Any]]:
        new_metrics = sorted(set(eval_dir.glob("images_test_*.json")) - metrics_before)
        new_videos = sorted(set(video_dir.glob("test_set_*.mp4")) - videos_before)
        return len(new_metrics) == 1 and bool(new_videos), {
            "new_metric_files": [str(path) for path in new_metrics],
            "new_video_files": [str(path) for path in new_videos],
        }

    eval_stage = run_stage(
        run_dir=run_dir,
        stage=f"eval_{safe_name}",
        command=eval_command,
        cwd=DRIVESTUDIO,
        environment=environment(),
        validate=validate_eval,
        timeout_seconds=timeout_seconds,
    )
    metric_files = sorted(set(eval_dir.glob("images_test_*.json")) - metrics_before)
    heldout_metrics = json.loads(metric_files[0].read_text(encoding="utf-8"))

    evaluator_source = {
        "status": "done",
        "task_id": "WS-V3-A2-ACTOR-DENSIFY-01",
        "scene_name": SCENE_NAME,
        "scene_index": SCENE_INDEX,
        "variant": view_name,
        "checkpoint": _checkpoint_source_contract(checkpoint_audit_payload),
        "registry": str(registry),
        "registry_sha256": sha256_file(registry),
        "selected_actors": selected,
        "heldout_metrics": heldout_metrics,
    }
    source_summary = output / "evaluator_source_summary.json"
    atomic_json(source_summary, evaluator_source)
    actor_output = output / "actor_metrics"
    actor_command = [
        str(DRIVESTUDIO_PYTHON),
        str(PROJECT / "scripts/eval_worldsim_v3_a0_actor_metrics.py"),
        "--source-summary",
        str(source_summary),
        "--output-dir",
        str(actor_output),
        "--drivestudio-root",
        str(DRIVESTUDIO),
        "--non-target-union",
    ]

    def validate_actor() -> tuple[bool, dict[str, Any]]:
        summary_path = actor_output / "summary.json"
        if not summary_path.is_file():
            return False, {"summary": str(summary_path), "exists": False}
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        roles = payload.get("roles") or {}
        non_target = payload.get("non_target") or {}
        unchanged = (
            payload.get("checkpoint_sha256_before")
            == payload.get("checkpoint_sha256_after")
            == checkpoint_sha
        )
        ok = (
            payload.get("status") == "done"
            and set(roles) == {"high-support", "boundary-support"}
            and all(row.get("status") == "done" for row in roles.values())
            and non_target.get("status") == "done"
            and unchanged
        )
        return ok, {
            "summary": str(summary_path),
            "role_statuses": {
                key: value.get("status") for key, value in roles.items()
            },
            "non_target_status": non_target.get("status"),
            "checkpoint_unchanged": unchanged,
        }

    actor_stage = run_stage(
        run_dir=run_dir,
        stage=f"actor_metrics_{safe_name}",
        command=actor_command,
        cwd=PROJECT,
        environment=environment(),
        validate=validate_actor,
        timeout_seconds=timeout_seconds,
    )
    actor_metrics = json.loads(
        (actor_output / "summary.json").read_text(encoding="utf-8")
    )
    if sha256_file(checkpoint) != checkpoint_sha:
        raise RuntimeError("read-only formal evaluation changed the checkpoint")
    result = {
        "status": "done",
        "view_name": view_name,
        "checkpoint": _checkpoint_source_contract(checkpoint_audit_payload),
        "registry": str(registry),
        "registry_sha256": sha256_file(registry),
        "selected_actors": selected,
        "heldout_metrics": heldout_metrics,
        "actor_metrics": actor_metrics,
        "resources": {
            "registry": registry_stage,
            "global_eval": eval_stage,
            "actor_eval": actor_stage,
        },
    }
    atomic_json(output / "summary.json", result)
    return result


def _resource_view(
    train_stage: dict[str, Any], duration_seconds: float, *, upper_bound: bool
) -> dict[str, Any]:
    result = {
        key: train_stage[key]
        for key in (
            "peak_gpu_memory_mib_sampled",
            "peak_gpu_memory_mib_torch_log",
            "peak_cgroup_memory_bytes",
        )
    }
    result["duration_seconds"] = duration_seconds
    result["peak_scope"] = (
        "full_30k_arm_upper_bound" if upper_bound else "fixed_step_arm"
    )
    return result


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--formal-contract",
        type=Path,
        default=PROJECT / "configs/worldsim_v3/a2_d1_formal_v1.yaml",
    )
    args = parser.parse_args()
    if not args.preflight_only and args.run_dir is None:
        raise ValueError("--run-dir is required unless --preflight-only is used")
    if args.run_dir is not None and args.run_dir.exists():
        raise FileExistsError(args.run_dir)

    formal = yaml.safe_load(args.formal_contract.read_text(encoding="utf-8"))
    validate_a2_d1_formal_contract(formal)
    base_contract = PROJECT / formal["depends_on"]["base_contract"]
    base = yaml.safe_load(base_contract.read_text(encoding="utf-8"))
    validate_a2_d1_contract(base)
    if sha256_file(base_contract) != formal["depends_on"]["base_contract_sha256"]:
        raise RuntimeError("A2-D1 base contract SHA drift")
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            formal["depends_on"]["implementation_commit"],
            "HEAD",
        ],
        cwd=PROJECT,
        check=True,
    )

    smoke_run = Path(formal["depends_on"]["paired_smoke_run"])
    smoke_terminal = json.loads(
        (smoke_run / "terminal.json").read_text(encoding="utf-8")
    )
    smoke_summary = smoke_run / "summary.json"
    if smoke_terminal.get("status") != "done" or sha256_file(smoke_summary) != formal[
        "depends_on"
    ]["paired_smoke_summary_sha256"]:
        raise RuntimeError("canonical A2-D1 paired smoke evidence drift")

    source_config = Path(formal["paired_design"]["source_config"])
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    minimum_free = float(formal["resource_contract"]["minimum_free_disk_gib"])
    free_gib = shutil.disk_usage("/root/autodl-tmp").free / (1024**3)
    if free_gib < minimum_free:
        raise RuntimeError(
            f"free disk {free_gib:.2f} GiB is below formal gate {minimum_free:.2f} GiB"
        )
    memory_max_path = Path("/sys/fs/cgroup/memory.max")
    memory_max = int(memory_max_path.read_text(encoding="utf-8").strip())
    if memory_max != int(formal["resource_contract"]["cgroup_memory_limit_bytes"]):
        raise RuntimeError(
            f"cgroup memory.max drift: expected "
            f"{formal['resource_contract']['cgroup_memory_limit_bytes']}, got {memory_max}"
        )

    runtime_sources = (
        "configs/worldsim_v3/a2_d1_v1.yaml",
        "configs/worldsim_v3/a2_d1_formal_v1.yaml",
        "motion_proj/worldsim_v3/a2_formal.py",
        "motion_proj/worldsim_v3/actor_quota.py",
        "motion_proj/worldsim_v3/gaussian_ancestry.py",
        "motion_proj/worldsim_v3/actor_metrics.py",
        "scripts/run_worldsim_v3_a2_d1_formal.py",
        "scripts/materialize_worldsim_v3_a2_d1_config.py",
        "scripts/run_worldsim_v3_a2_d1_paired_smoke.py",
        "scripts/prepare_worldsim_v3_a2_d1_drivestudio.py",
        "scripts/build_dr_v2_drivestudio_registry.py",
        "scripts/eval_worldsim_v3_a0_actor_metrics.py",
        "scripts/run_worldsim_v3_a0_scene.py",
        "scripts/run_worldsim_v3_a0_smoke.py",
        "tests/test_worldsim_v3_a2_formal.py",
        "tests/test_materialize_worldsim_v3_a2_d1_formal_config.py",
        "tests/test_worldsim_v3_actor_metrics.py",
    )
    _clean_relevant_sources(runtime_sources)

    compatibility_patch = PROJECT / base["drivestudio"]["compatibility_patch"]
    instrumentation_patch = PROJECT / base["drivestudio"]["instrumentation_patch"]
    quota_patch = PROJECT / base["drivestudio"]["quota_patch"]
    worktree = verify_patched_tree(
        DRIVESTUDIO,
        compatibility_patch,
        instrumentation_patch,
        quota_patch,
    )
    preflight_resources = resource_sample("a2_d1_formal", "preflight")
    if int(preflight_resources["gpu"]["memory_used_mib"]) > int(
        formal["resource_contract"]["gpu_idle_max_mib"]
    ):
        raise RuntimeError(f"GPU is not idle: {preflight_resources['gpu']}")
    if args.preflight_only:
        result = {
            "status": "done",
            "component": "A2-D1 formal read-only preflight",
            "formal_contract_sha256": sha256_file(args.formal_contract),
            "base_contract_sha256": sha256_file(base_contract),
            "paired_smoke_summary_sha256": sha256_file(smoke_summary),
            "project_commit": command_output(
                "git", "rev-parse", "HEAD", cwd=PROJECT
            ),
            "free_disk_gib": free_gib,
            "resources": preflight_resources,
            "patched_drivestudio": worktree,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    assert args.run_dir is not None
    _ACTIVE_RUN_DIR = args.run_dir
    for name in (
        "artifacts",
        "environment",
        "logs",
        "source_snapshot",
        "stages",
        "work_dirs",
    ):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )
    shutil.copy2(args.formal_contract, args.run_dir / "resolved.yaml")
    atomic_json(args.run_dir / "artifacts/worktree.json", worktree)

    snapshot_sources = tuple(PROJECT / path for path in runtime_sources) + (
        compatibility_patch,
        instrumentation_patch,
        quota_patch,
    )
    source_hashes: dict[str, str] = {}
    for source in snapshot_sources:
        relative = source.relative_to(PROJECT)
        destination = args.run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = sha256_file(source)

    design = formal["paired_design"]
    num_iters = int(design["num_iters"])
    checkpoint_interval = int(design["checkpoint_interval"])
    candidate_steps = list(formal["matched_gaussian_budget"]["candidate_steps"])
    frozen_initial_counts = list(base["frozen_scene_0230_reference"]["initial_actor_counts"])
    configs: dict[str, Path] = {}
    materialize_commands: dict[str, list[str]] = {}
    for variant in VARIANTS:
        output = args.run_dir / "artifacts" / f"{variant}.yaml"
        materialize_commands[variant] = materialize(
            source_config=source_config,
            base_contract=base_contract,
            variant=variant,
            num_iters=num_iters,
            checkpoint_interval=checkpoint_interval,
            output=output,
        )
        configs[variant] = output
    matched_configs = normalized_paired_config(configs[VARIANTS[0]]) == normalized_paired_config(
        configs[VARIANTS[1]]
    )
    if not matched_configs:
        raise RuntimeError("D0/D1 formal configs differ beyond frozen variant fields")

    project_commit = command_output("git", "rev-parse", "HEAD", cwd=PROJECT)
    manifest = {
        "schema_version": 1,
        "task_id": "WS-V3-A2-ACTOR-DENSIFY-01",
        "component": "A2-D1 paired formal 30k and matched-RigidNodes-budget view",
        "scene_name": SCENE_NAME,
        "scene_index": SCENE_INDEX,
        "seed": 0,
        "order": list(VARIANTS),
        "num_iters": num_iters,
        "checkpoint_interval": checkpoint_interval,
        "candidate_steps": candidate_steps,
        "formal_contract_sha256": sha256_file(args.formal_contract),
        "base_contract_sha256": sha256_file(base_contract),
        "source_config": str(source_config),
        "source_config_sha256": sha256_file(source_config),
        "materialized_config_sha256": {
            key: sha256_file(value) for key, value in configs.items()
        },
        "matched_configs_after_variant_normalization": matched_configs,
        "materialize_commands": materialize_commands,
        "source_hashes": source_hashes,
        "project_commit": project_commit,
        "project_status": command_output(
            "git", "status", "--short", cwd=PROJECT
        ).splitlines(),
        "patched_drivestudio": worktree,
        "started_at": now(),
    }
    atomic_json(args.run_dir / "manifest.json", manifest)

    train_timeout = float(
        formal["resource_contract"]["train_timeout_seconds_per_arm"]
    )
    eval_timeout = float(
        formal["resource_contract"]["eval_timeout_seconds_per_stage"]
    )
    arm_results: dict[str, dict[str, Any]] = {}
    fixed_evaluations: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        train_command, final_checkpoint = build_train_command(
            args.run_dir, configs[variant], variant, num_iters
        )
        provenance_path = args.run_dir / "artifacts" / f"{variant}-init-provenance.json"
        arm_environment = environment()
        arm_environment["WORLDSIM_V3_INIT_PROVENANCE"] = str(provenance_path)
        arm_environment["WORLDSIM_V3_INIT_SEED"] = "0"
        arm_started_epoch = time.time()
        stage_name = f"train_{variant.replace('-', '_')}_{num_iters}"
        train_stage = run_stage(
            run_dir=args.run_dir,
            stage=stage_name,
            command=train_command,
            cwd=DRIVESTUDIO,
            environment=arm_environment,
            validate=lambda checkpoint=final_checkpoint, provenance=provenance_path: (
                checkpoint.is_file()
                and checkpoint.stat().st_size > 0
                and provenance.is_file()
                and provenance.stat().st_size > 0,
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_bytes": checkpoint.stat().st_size
                    if checkpoint.is_file()
                    else 0,
                    "init_provenance": str(provenance),
                },
            ),
            timeout_seconds=train_timeout,
        )
        grid = checkpoint_grid(
            final_checkpoint,
            steps=candidate_steps,
            variant=variant,
            frozen_initial_counts=frozen_initial_counts,
            arm_started_epoch=arm_started_epoch,
        )
        final_audit = grid[-1]["audit"]
        provenance_payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_payload.get("truth_tier") != design["initialization_provenance"][
            "require_truth_tier"
        ]:
            raise RuntimeError("initialization provenance truth tier drift")
        arm_results[variant] = {
            "status": "done",
            "variant": variant,
            "train_command": train_command,
            "train_stage": train_stage,
            "checkpoint_grid": grid,
            "final_checkpoint_audit": final_audit,
            "initialization_provenance": {
                "path": str(provenance_path),
                "sha256": sha256_file(provenance_path),
                "truth_tier": provenance_payload.get("truth_tier"),
                "initialized_gaussians": provenance_payload.get(
                    "initialized_gaussians"
                ),
            },
        }
        atomic_json(
            args.run_dir / "artifacts" / f"{variant}-train-summary.json",
            arm_results[variant],
        )
        fixed_evaluations[variant] = evaluate_checkpoint(
            run_dir=args.run_dir,
            view_name=f"fixed-{variant}",
            checkpoint_audit_payload=final_audit,
            timeout_seconds=eval_timeout,
        )

    provenance_matched = (
        arm_results[VARIANTS[0]]["initialization_provenance"]["sha256"]
        == arm_results[VARIANTS[1]]["initialization_provenance"]["sha256"]
    )
    if not provenance_matched:
        raise RuntimeError("D0/D1 formal initialization provenance mismatch")

    d0_final = arm_results[VARIANTS[0]]["final_checkpoint_audit"]
    d1_grid = arm_results[VARIANTS[1]]["checkpoint_grid"]
    matched_budget = select_matched_checkpoint(
        int(d0_final["rigid_total"]),
        [
            {
                key: row[key]
                for key in (
                    "step",
                    "checkpoint",
                    "checkpoint_sha256",
                    "rigid_gaussians",
                    "background_gaussians",
                    "total_gaussians",
                    "elapsed_to_checkpoint_seconds",
                )
            }
            for row in d1_grid
        ],
        float(formal["matched_gaussian_budget"]["maximum_relative_gap"]),
    )

    d0_resources = _resource_view(
        arm_results[VARIANTS[0]]["train_stage"],
        float(arm_results[VARIANTS[0]]["train_stage"]["duration_seconds"]),
        upper_bound=False,
    )
    d1_resources = _resource_view(
        arm_results[VARIANTS[1]]["train_stage"],
        float(arm_results[VARIANTS[1]]["train_stage"]["duration_seconds"]),
        upper_bound=False,
    )
    fixed_comparison = compare_view(
        fixed_evaluations[VARIANTS[0]],
        fixed_evaluations[VARIANTS[1]],
        d0_final,
        arm_results[VARIANTS[1]]["final_checkpoint_audit"],
        d0_resources,
        d1_resources,
    )

    matched_comparison: dict[str, Any] | None = None
    matched_evaluation: dict[str, Any] | None = None
    if matched_budget["status"] == "done":
        selected_step = int(matched_budget["selected"]["step"])
        selected_row = next(row for row in d1_grid if row["step"] == selected_step)
        if selected_step == num_iters:
            matched_evaluation = {
                **fixed_evaluations[VARIANTS[1]],
                "exact_alias_of": f"fixed-{VARIANTS[1]}",
            }
        else:
            matched_evaluation = evaluate_checkpoint(
                run_dir=args.run_dir,
                view_name=f"matched-d1-step-{selected_step}",
                checkpoint_audit_payload=selected_row["audit"],
                timeout_seconds=eval_timeout,
            )
        matched_d1_resources = _resource_view(
            arm_results[VARIANTS[1]]["train_stage"],
            float(selected_row["elapsed_to_checkpoint_seconds"]),
            upper_bound=selected_step != num_iters,
        )
        matched_comparison = compare_view(
            fixed_evaluations[VARIANTS[0]],
            matched_evaluation,
            d0_final,
            selected_row["audit"],
            d0_resources,
            matched_d1_resources,
        )
        matched_budget["d1_train_resources"] = matched_d1_resources
        matched_budget["d0_view"] = "exact_alias_of_fixed_d0_final"

    d2_unlocked = matched_budget["status"] == "done"
    summary = {
        "status": "done",
        "task_id": "WS-V3-A2-ACTOR-DENSIFY-01",
        "component": "A2-D1 paired formal comparison",
        "scene_name": SCENE_NAME,
        "scene_index": SCENE_INDEX,
        "seed": 0,
        "formal_contract_sha256": sha256_file(args.formal_contract),
        "project_commit": project_commit,
        "paired_config_matched": matched_configs,
        "initialization_provenance_matched": provenance_matched,
        "arms": arm_results,
        "fixed_step": {
            "status": "done",
            "d0_evaluation": fixed_evaluations[VARIANTS[0]],
            "d1_evaluation": fixed_evaluations[VARIANTS[1]],
            "comparison": fixed_comparison,
        },
        "matched_gaussian_budget": {
            **matched_budget,
            "d1_evaluation": matched_evaluation,
            "comparison": matched_comparison,
        },
        "d2_unlocked": d2_unlocked,
        "next_action": (
            "freeze D2 boundary/residual ordering protocol"
            if d2_unlocked
            else "freeze a prospective D1 matched-budget repair before D2"
        ),
        "claim_boundary": (
            "Formal evidence is limited to scene-0230. A negative or tradeoff "
            "result remains valid evidence; more Gaussians are not an improvement."
        ),
        "completed_at": now(),
    }
    atomic_json(args.run_dir / "summary.json", summary)
    manifest["status"] = "done"
    manifest["completed_at"] = summary["completed_at"]
    atomic_json(args.run_dir / "manifest.json", manifest)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "done", "updated_at": now(), "failure": None},
    )
    _TERMINAL_FINAL = True
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


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
                        "code": "A2_D1_FORMAL_RUN_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
            )
        raise

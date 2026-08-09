#!/usr/bin/env python
"""运行冻结的 scene-0230 A2-D2 formal 单臂与 D1 exact-alias 比较。"""

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

from motion_proj.worldsim_v3.a2_formal import (  # noqa: E402
    compare_view,
    select_matched_checkpoint,
)
from motion_proj.worldsim_v3.boundary_residual import (  # noqa: E402
    validate_a2_d2_contract,
)
from motion_proj.worldsim_v3.d2_formal import (  # noqa: E402
    validate_a2_d2_formal_contract,
)
from scripts.materialize_worldsim_v3_a2_d2_config import (  # noqa: E402
    normalized_pair_payload,
)
from scripts.prepare_worldsim_v3_a2_d2_drivestudio import (  # noqa: E402
    sha256_file,
    verify_patched_tree,
)
from scripts.run_worldsim_v3_a0_scene import run_stage  # noqa: E402
from scripts.run_worldsim_v3_a0_smoke import (  # noqa: E402
    atomic_json,
    command_output,
    now,
    resource_sample,
)
from scripts.run_worldsim_v3_a2_d2_paired_smoke import (  # noqa: E402
    checkpoint_audit,
)
import scripts.run_worldsim_v3_a2_d1_formal as legacy  # noqa: E402


DRIVESTUDIO = Path(
    "/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-d2-r8"
)
DRIVESTUDIO_PYTHON = Path("/root/autodl-tmp/envs/drivestudio/bin/python")
SCENE_NAME = "scene-0230"
SCENE_INDEX = 179
VARIANTS = ("d1-actor-quota", "d2-boundary-residual")
_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False

# D1 formal 的评测器是通用只读实现；绑定 D2 worktree 与 D2 checkpoint audit。
legacy.DRIVESTUDIO = DRIVESTUDIO
legacy.checkpoint_audit = checkpoint_audit


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


def clean_relevant_sources(relative_paths: tuple[str, ...]) -> None:
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


def materialize(
    *,
    source_config: Path,
    protocol: Path,
    variant: str,
    num_iters: int,
    checkpoint_interval: int,
    output: Path,
) -> list[str]:
    command = [
        str(DRIVESTUDIO_PYTHON),
        str(PROJECT / "scripts/materialize_worldsim_v3_a2_d2_config.py"),
        "--source-config",
        str(source_config),
        "--protocol",
        str(protocol),
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


def provenance_fingerprint(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "background_lidar_sample": payload["background_lidar_sample"],
        "instance_lidar_samples": payload["instance_lidar_samples"],
        "initialized_gaussians": payload["initialized_gaussians"],
        "rng_reset": payload["rng_reset"],
        "truth_tier": payload["truth_tier"],
    }


def alias_resource_view(stage: dict[str, Any]) -> dict[str, Any]:
    return legacy._resource_view(  # noqa: SLF001
        stage,
        float(stage["duration_seconds"]),
        upper_bound=False,
    )


def load_d1_alias(
    formal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    alias = formal["d1_reference_alias"]
    run = Path(alias["run"])
    terminal = json.loads((run / "terminal.json").read_text(encoding="utf-8"))
    summary_path = run / "summary.json"
    if terminal.get("status") != "done":
        raise RuntimeError("D1 alias run is not terminal done")
    if sha256_file(summary_path) != alias["summary_sha256"]:
        raise RuntimeError("D1 alias summary SHA drift")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = summary["arms"]["d1-actor-quota"]["final_checkpoint_audit"]
    evaluation = summary["fixed_step"]["d1_evaluation"]
    checkpoint = Path(alias["fixed_checkpoint"])
    expected = alias["fixed_checkpoint_sha256"]
    if (
        audit["checkpoint_sha256"] != expected
        or evaluation["checkpoint"]["sha256"] != expected
        or sha256_file(checkpoint) != expected
    ):
        raise RuntimeError("D1 alias fixed checkpoint SHA drift")
    if (
        audit["rigid_total"] != alias["fixed_rigid_gaussians"]
        or audit["background_total"] != alias["fixed_background_gaussians"]
    ):
        raise RuntimeError("D1 alias Gaussian counts drift")
    provenance = Path(
        summary["arms"]["d1-actor-quota"]["initialization_provenance"][
            "path"
        ]
    )
    if sha256_file(provenance) != alias["initialization_provenance_sha256"]:
        raise RuntimeError("D1 alias initialization provenance drift")
    return summary, {
        "status": "done",
        "mode": alias["mode"],
        "source_run": str(run),
        "source_summary_sha256": alias["summary_sha256"],
        "source_commit": alias["source_commit"],
        "checkpoint": audit,
        "evaluation": evaluation,
        "train_stage": summary["arms"]["d1-actor-quota"]["train_stage"],
        "initialization_provenance": {
            "path": str(provenance),
            "sha256": sha256_file(provenance),
            "fingerprint": provenance_fingerprint(provenance),
        },
    }


def verify_dependencies(
    formal: dict[str, Any], formal_contract: Path
) -> dict[str, Any]:
    validate_a2_d2_formal_contract(formal)
    depends_on = formal["depends_on"]
    protocol_path = PROJECT / depends_on["d2_protocol"]
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    validate_a2_d2_contract(protocol)
    if sha256_file(protocol_path) != depends_on["d2_protocol_sha256"]:
        raise RuntimeError("D2 protocol SHA drift")
    for commit in (
        depends_on["implementation_commit"],
        depends_on["smoke_closeout_commit"],
    ):
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=PROJECT,
            check=True,
        )
    smoke_run = Path(depends_on["paired_smoke_run"])
    terminal = json.loads(
        (smoke_run / "terminal.json").read_text(encoding="utf-8")
    )
    summary = smoke_run / "summary.json"
    manifest = smoke_run / "manifest.json"
    if (
        terminal.get("status") != "done"
        or sha256_file(summary) != depends_on["paired_smoke_summary_sha256"]
        or sha256_file(manifest) != depends_on["paired_smoke_manifest_sha256"]
    ):
        raise RuntimeError("canonical D2 smoke evidence drift")
    return {
        "formal_contract": str(formal_contract),
        "formal_contract_sha256": sha256_file(formal_contract),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "smoke_run": str(smoke_run),
        "smoke_summary_sha256": sha256_file(summary),
        "smoke_manifest_sha256": sha256_file(manifest),
    }


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--formal-contract",
        type=Path,
        default=PROJECT / "configs/worldsim_v3/a2_d2_formal_v1.yaml",
    )
    args = parser.parse_args()
    if not args.preflight_only and args.run_dir is None:
        raise ValueError("--run-dir is required unless --preflight-only is used")
    if args.run_dir is not None and args.run_dir.exists():
        raise FileExistsError(args.run_dir)

    formal = yaml.safe_load(args.formal_contract.read_text(encoding="utf-8"))
    dependency_audit = verify_dependencies(formal, args.formal_contract)
    d1_summary, d1_alias = load_d1_alias(formal)
    design = formal["paired_design"]
    source_config = Path(design["source_config"])
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    free_gib = shutil.disk_usage("/root/autodl-tmp").free / (1024**3)
    resources = formal["resource_contract"]
    if free_gib < float(resources["minimum_free_disk_gib"]):
        raise RuntimeError("free disk is below the D2 formal gate")
    memory_max = int(
        Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
    )
    if memory_max != int(resources["cgroup_memory_limit_bytes"]):
        raise RuntimeError("cgroup memory.max drift")

    runtime_sources = (
        "configs/worldsim_v3/a2_d2_protocol_v1.yaml",
        "configs/worldsim_v3/a2_d2_formal_v1.yaml",
        "motion_proj/worldsim_v3/a2_formal.py",
        "motion_proj/worldsim_v3/actor_metrics.py",
        "motion_proj/worldsim_v3/actor_quota.py",
        "motion_proj/worldsim_v3/boundary_residual.py",
        "motion_proj/worldsim_v3/d2_formal.py",
        "motion_proj/worldsim_v3/gaussian_ancestry.py",
        "scripts/materialize_worldsim_v3_a2_d2_config.py",
        "scripts/prepare_worldsim_v3_a2_d2_drivestudio.py",
        "scripts/run_worldsim_v3_a2_d1_formal.py",
        "scripts/run_worldsim_v3_a2_d2_formal.py",
        "scripts/run_worldsim_v3_a2_d2_paired_smoke.py",
        "scripts/build_dr_v2_drivestudio_registry.py",
        "scripts/eval_worldsim_v3_a0_actor_metrics.py",
        "scripts/run_worldsim_v3_a0_scene.py",
        "scripts/run_worldsim_v3_a0_smoke.py",
        "tests/test_materialize_worldsim_v3_a2_d2_config.py",
        "tests/test_worldsim_v3_a2_formal.py",
        "tests/test_worldsim_v3_d2_formal.py",
    )
    clean_relevant_sources(runtime_sources)

    patches = (
        PROJECT / "compatibility/DriveStudio-2026-08-05.patch",
        PROJECT / "compatibility/DriveStudio-WorldSim-A2-ancestry-v1.patch",
        PROJECT / "compatibility/DriveStudio-WorldSim-A2-D1-quota-v1.patch",
        PROJECT
        / "compatibility/DriveStudio-WorldSim-A2-D2-boundary-residual-v1.patch",
    )
    worktree = verify_patched_tree(DRIVESTUDIO, *patches)
    preflight_resources = resource_sample("a2_d2_formal", "preflight")
    if int(preflight_resources["gpu"]["memory_used_mib"]) > int(
        resources["gpu_idle_max_mib"]
    ):
        raise RuntimeError("GPU is not idle")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "done",
                    "component": "A2-D2 formal read-only preflight",
                    "dependency_audit": dependency_audit,
                    "d1_reference_alias": d1_alias,
                    "project_commit": command_output(
                        "git", "rev-parse", "HEAD", cwd=PROJECT
                    ),
                    "free_disk_gib": free_gib,
                    "resources": preflight_resources,
                    "patched_drivestudio": worktree,
                },
                indent=2,
                sort_keys=True,
            )
        )
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
    atomic_json(args.run_dir / "artifacts/d1-reference-alias.json", d1_alias)

    snapshot_sources = tuple(PROJECT / path for path in runtime_sources) + patches
    source_hashes: dict[str, str] = {}
    for source in snapshot_sources:
        relative = source.relative_to(PROJECT)
        destination = args.run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = sha256_file(source)

    num_iters = int(design["num_iters"])
    checkpoint_interval = int(design["checkpoint_interval"])
    protocol_path = PROJECT / formal["depends_on"]["d2_protocol"]
    configs: dict[str, Path] = {}
    materialize_commands = {}
    for variant in VARIANTS:
        output = args.run_dir / "artifacts" / f"{variant}.yaml"
        materialize_commands[variant] = materialize(
            source_config=source_config,
            protocol=protocol_path,
            variant=variant,
            num_iters=num_iters,
            checkpoint_interval=checkpoint_interval,
            output=output,
        )
        configs[variant] = output
    normalized = [
        normalized_pair_payload(OmegaConf.load(configs[variant]))
        for variant in VARIANTS
    ]
    matched_configs = normalized[0] == normalized[1]
    if not matched_configs:
        raise RuntimeError("D1/D2 formal configs exceed frozen differences")

    project_commit = command_output("git", "rev-parse", "HEAD", cwd=PROJECT)
    manifest = {
        "schema_version": 1,
        "task_id": formal["task_id"],
        "component": "A2-D2 formal with immutable D1 reference alias",
        "scene_name": SCENE_NAME,
        "scene_index": SCENE_INDEX,
        "seed": 0,
        "order": list(design["order"]),
        "trained_arms": list(design["trained_arms"]),
        "exact_alias_arms": list(design["exact_alias_arms"]),
        "num_iters": num_iters,
        "checkpoint_interval": checkpoint_interval,
        "candidate_steps": formal["matched_gaussian_budget"][
            "candidate_steps"
        ],
        "dependency_audit": dependency_audit,
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

    d1_checkpoint = Path(formal["d1_reference_alias"]["fixed_checkpoint"])
    d1_sha_before = sha256_file(d1_checkpoint)
    provenance_path = args.run_dir / "artifacts/d2-init-provenance.json"
    arm_environment = environment()
    arm_environment["WORLDSIM_V3_INIT_PROVENANCE"] = str(provenance_path)
    arm_environment["WORLDSIM_V3_INIT_SEED"] = "0"
    train_command, final_checkpoint = legacy.build_train_command(
        args.run_dir,
        configs["d2-boundary-residual"],
        "d2-boundary-residual",
        num_iters,
    )
    started_epoch = time.time()
    train_stage = run_stage(
        run_dir=args.run_dir,
        stage="train_d2_boundary_residual_30000",
        command=train_command,
        cwd=DRIVESTUDIO,
        environment=arm_environment,
        validate=lambda: (
            final_checkpoint.is_file()
            and final_checkpoint.stat().st_size > 0
            and provenance_path.is_file()
            and provenance_path.stat().st_size > 0,
            {
                "checkpoint": str(final_checkpoint),
                "provenance": str(provenance_path),
            },
        ),
        timeout_seconds=float(resources["train_timeout_seconds_per_arm"]),
    )
    frozen_counts = d1_summary["arms"]["d1-actor-quota"][
        "final_checkpoint_audit"
    ]["quota_initial_counts"]
    grid = legacy.checkpoint_grid(
        final_checkpoint,
        steps=formal["matched_gaussian_budget"]["candidate_steps"],
        variant="d2-boundary-residual",
        frozen_initial_counts=frozen_counts,
        arm_started_epoch=started_epoch,
    )
    d2_final_audit = grid[-1]["audit"]
    d2_provenance_sha = sha256_file(provenance_path)
    expected_provenance_sha = formal["d1_reference_alias"][
        "initialization_provenance_sha256"
    ]
    if d2_provenance_sha != expected_provenance_sha:
        raise RuntimeError("D2 initialization provenance differs from D1 alias")
    d2_arm = {
        "status": "done",
        "variant": "d2-boundary-residual",
        "train_command": train_command,
        "train_stage": train_stage,
        "checkpoint_grid": grid,
        "final_checkpoint_audit": d2_final_audit,
        "initialization_provenance": {
            "path": str(provenance_path),
            "sha256": d2_provenance_sha,
            "fingerprint": provenance_fingerprint(provenance_path),
        },
    }
    atomic_json(args.run_dir / "artifacts/d2-train-summary.json", d2_arm)

    eval_timeout = float(resources["eval_timeout_seconds_per_stage"])
    d2_fixed_evaluation = legacy.evaluate_checkpoint(
        run_dir=args.run_dir,
        view_name="fixed-d2-boundary-residual",
        checkpoint_audit_payload=d2_final_audit,
        timeout_seconds=eval_timeout,
    )
    d1_audit = d1_alias["checkpoint"]
    d1_evaluation = d1_alias["evaluation"]
    d1_resources = alias_resource_view(d1_alias["train_stage"])
    d2_resources = legacy._resource_view(  # noqa: SLF001
        train_stage,
        float(train_stage["duration_seconds"]),
        upper_bound=False,
    )
    fixed_comparison = compare_view(
        d1_evaluation,
        d2_fixed_evaluation,
        d1_audit,
        d2_final_audit,
        d1_resources,
        d2_resources,
    )

    matched_contract = formal["matched_gaussian_budget"]
    matched_budget = select_matched_checkpoint(
        int(matched_contract["target_count"]),
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
            for row in grid
        ],
        float(matched_contract["maximum_relative_gap"]),
    )
    matched_evaluation = None
    matched_comparison = None
    if matched_budget["status"] == "done":
        selected_step = int(matched_budget["selected"]["step"])
        selected_row = next(row for row in grid if row["step"] == selected_step)
        if selected_step == num_iters:
            matched_evaluation = {
                **d2_fixed_evaluation,
                "exact_alias_of": "fixed-d2-boundary-residual",
            }
        else:
            matched_evaluation = legacy.evaluate_checkpoint(
                run_dir=args.run_dir,
                view_name=f"matched-d2-step-{selected_step}",
                checkpoint_audit_payload=selected_row["audit"],
                timeout_seconds=eval_timeout,
            )
        matched_d2_resources = legacy._resource_view(  # noqa: SLF001
            train_stage,
            float(selected_row["elapsed_to_checkpoint_seconds"]),
            upper_bound=selected_step != num_iters,
        )
        matched_comparison = compare_view(
            d1_evaluation,
            matched_evaluation,
            d1_audit,
            selected_row["audit"],
            d1_resources,
            matched_d2_resources,
        )
        matched_budget["d1_view"] = "exact_alias_of_fixed_d1_final"
        matched_budget["d2_train_resources"] = matched_d2_resources

    d1_sha_after = sha256_file(d1_checkpoint)
    if d1_sha_before != d1_sha_after:
        raise RuntimeError("D1 exact-alias checkpoint changed during D2 formal")
    summary = {
        "status": "done",
        "task_id": formal["task_id"],
        "component": "A2-D2 formal comparison",
        "scene_name": SCENE_NAME,
        "scene_index": SCENE_INDEX,
        "seed": 0,
        "formal_contract_sha256": sha256_file(args.formal_contract),
        "project_commit": project_commit,
        "comparison_role_mapping": {
            "d0_fields_in_shared_comparator": "d1-reference-baseline",
            "d1_fields_in_shared_comparator": "d2-candidate",
        },
        "paired_config_matched": matched_configs,
        "initialization_provenance_matched": True,
        "d1_reference_alias": {
            **d1_alias,
            "checkpoint_sha256_before": d1_sha_before,
            "checkpoint_sha256_after": d1_sha_after,
            "checkpoint_unchanged": True,
        },
        "d2_arm": d2_arm,
        "fixed_step": {
            "status": "done",
            "d1_evaluation": d1_evaluation,
            "d2_evaluation": d2_fixed_evaluation,
            "comparison": fixed_comparison,
        },
        "matched_gaussian_budget": {
            **matched_budget,
            "d1_evaluation": d1_evaluation,
            "d2_evaluation": matched_evaluation,
            "comparison": matched_comparison,
        },
        "d2_formal_complete": True,
        "d3_unlocked": False,
        "next_action": (
            "review D2 fixed/matched Pareto and freeze A2 selected variant; "
            "D3 remains conditional on reliable registered depth/normal inputs"
        ),
        "claim_boundary": (
            "Formal evidence is limited to scene-0230. Negative, abstain, or "
            "tradeoff results are valid; more Gaussians are not improvements."
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
                        "code": "A2_D2_FORMAL_RUN_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
            )
        raise

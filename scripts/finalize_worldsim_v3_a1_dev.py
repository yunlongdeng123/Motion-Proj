#!/usr/bin/env python
"""冻结并正式汇总 WorldSim V3 A1 开发场景 C* 选择。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_worldsim_v3_a0_scene import common_environment, run_stage
from scripts.run_worldsim_v3_a0_smoke import atomic_json, command_output, now, sha256_file


PROJECT = Path("/root/autodl-tmp/motion_proj")
MOTIONPROJ_PYTHON = Path("/root/autodl-tmp/envs/motionproj/bin/python")
VARIANTS = (
    "c0-off",
    "c1-native",
    "c2-factorized-isp",
    "c3-bounded-pose",
)
ROLES = ("boundary-support", "high-support")
DECISION_STATUS = {
    "c0-off": "done_off",
    "c1-native": "done_native",
    "c2-factorized-isp": "done_enhanced",
    "c3-bounded-pose": "done_enhanced",
}
_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False


def nested(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(path)
        value = value[key]
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_done(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    terminal_path = run_dir / "terminal.json"
    summary_path = run_dir / "summary.json"
    if not terminal_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"run 缺少 terminal/summary：{run_dir}")
    terminal = load_json(terminal_path)
    summary = load_json(summary_path)
    if terminal.get("status") != "done" or summary.get("status") != "done":
        raise RuntimeError(f"run 不是 done：{run_dir}")
    return terminal, summary


def validate_selection_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema_version") != 1:
        raise ValueError("selection schema_version 必须为 1")
    if contract.get("selection_version") != "A1-S0-v1":
        raise ValueError("selection_version 必须为 A1-S0-v1")
    scope = contract.get("scope", {})
    if scope.get("baseline_variant") != "c0-off":
        raise ValueError("A1-S0-v1 baseline 必须为 c0-off")
    alternatives = tuple(scope.get("alternatives", []))
    if alternatives != VARIANTS[1:]:
        raise ValueError(f"alternatives 必须为 {VARIANTS[1:]}")
    disclosure = contract.get("result_access_disclosure", {})
    if disclosure.get("operationalized_after_development_results") is not True:
        raise ValueError("必须披露开发结果已可见")
    if disclosure.get("frozen_before_confirmation_scenes") is not True:
        raise ValueError("必须在确认场景前冻结")
    if disclosure.get("numerical_tolerances_introduced") is not False:
        raise ValueError("A1-S0-v1 不允许数值容差")
    decision = contract.get("decision", {})
    if decision.get("comparator") != "strict_floating_point_order_without_tolerance":
        raise ValueError("只允许无容差严格比较")
    if decision.get("fallback_variant") != "c0-off":
        raise ValueError("fallback 必须为 c0-off")
    source = contract.get("source_contract", {})
    if set(source.get("training_runs", {})) != set(VARIANTS):
        raise ValueError("training_runs 必须覆盖 C0-C3")
    if set(source.get("endpoint_runs", {})) != set(VARIANTS):
        raise ValueError("endpoint_runs 必须覆盖 C0-C3")


def resolve_sources(contract: dict[str, Any], task_runs_root: Path) -> dict[str, Any]:
    source = contract["source_contract"]
    return {
        "training": {
            variant: task_runs_root / source["training_runs"][variant]
            for variant in VARIANTS
        },
        "endpoint": {
            variant: task_runs_root / source["endpoint_runs"][variant]
            for variant in VARIANTS
        },
        "diagnostic": task_runs_root / source["diagnostic_run"],
    }


def actor_metrics(source: dict[str, Any], role: str) -> dict[str, Any]:
    row = source["actor_metrics"]["roles"][role]
    if row.get("status") != "done":
        raise RuntimeError(f"actor role 不是 done：{role}")
    actor = row["actor_region"]
    boundary = row["boundary_band"]
    if actor.get("status") != "done" or boundary.get("status") != "done":
        raise RuntimeError(f"actor/boundary 指标不是 done：{role}")
    return {
        "effect_pixel_coverage": row["effect_pixel_coverage"],
        "actor_psnr": actor["psnr"],
        "actor_ssim": actor["ssim"],
        "actor_lpips": actor["masked_lpips_alex_tight_crop_256px"],
        "boundary_psnr": boundary["psnr"],
        "boundary_ssim": boundary["ssim"],
        "boundary_lpips": boundary["masked_lpips_alex_tight_crop_256px"],
    }


def endpoint_role(source: dict[str, Any], role: str) -> dict[str, Any]:
    row = source["endpoint_metrics"]["e2"]["roles"][role]
    if row.get("status") != "done":
        raise RuntimeError(f"E2 role 不是 done：{role}")
    distance = row["normalized_bidirectional_distance"]
    return {
        "candidate_count": row["candidate_count"],
        "valid_count": row["valid_count"],
        "coverage": row["coverage"],
        "mean": distance["mean"],
        "median": distance["median"],
        "p90": distance["p90"],
    }


def compact_diagnostic(source: dict[str, Any]) -> dict[str, Any]:
    diagnostic = source["diagnostic"]
    if diagnostic.get("checkpoint_unchanged") is not True:
        raise RuntimeError("诊断前后 checkpoint 不一致")
    return {
        "pose_translation_median_m": nested(
            diagnostic, "pose.overall.translation_norm_m.median"
        ),
        "pose_translation_p90_m": nested(
            diagnostic, "pose.overall.translation_norm_m.p90"
        ),
        "pose_rotation_median_deg": nested(
            diagnostic, "pose.overall.rotation_angle_deg.median"
        ),
        "pose_rotation_p90_deg": nested(
            diagnostic, "pose.overall.rotation_angle_deg.p90"
        ),
        "pose_translation_first_p90_m": nested(
            diagnostic, "pose.first_difference.translation_delta_norm_m.p90"
        ),
        "pose_translation_second_p90_m": nested(
            diagnostic, "pose.second_difference.translation_jitter_norm_m.p90"
        ),
        "pose_rotation_first_p90_deg": nested(
            diagnostic, "pose.first_difference.rotation_delta_deg.p90"
        ),
        "pose_rotation_second_p90_deg": nested(
            diagnostic, "pose.second_difference.rotation_jitter_deg.p90"
        ),
        "isp_residual_median": nested(diagnostic, "isp.overall.residual_l2.median"),
        "isp_residual_p90": nested(diagnostic, "isp.overall.residual_l2.p90"),
        "isp_temporal_first_p90": nested(
            diagnostic, "isp.temporal_first_difference_l2.p90"
        ),
        "isp_temporal_second_p90": nested(
            diagnostic, "isp.temporal_second_difference_l2.p90"
        ),
    }


def validate_and_build_rows(
    contract: dict[str, Any], sources: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    expected = contract["source_contract"]
    scene = contract["scope"]
    _, diagnostic_wrapper = require_done(sources["diagnostic"])
    if diagnostic_wrapper.get("diagnostic_config_sha256") != expected["diagnostic_config_sha256"]:
        raise RuntimeError("diagnostic config SHA 不匹配")
    diagnostic_summary = diagnostic_wrapper["diagnostic_metrics"]
    if diagnostic_summary.get("paired_initialization_sha256") != expected["paired_initialization_sha256"]:
        raise RuntimeError("诊断中的配对初始化 SHA 不匹配")
    diagnostics = diagnostic_summary.get("variants", {})
    if set(diagnostics) != set(VARIANTS):
        raise RuntimeError("诊断变体集合不匹配")

    rows: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        train_dir = sources["training"][variant]
        endpoint_dir = sources["endpoint"][variant]
        _, train = require_done(train_dir)
        _, endpoint = require_done(endpoint_dir)
        if (
            train.get("variant") != variant
            or train.get("scene_name") != scene["scene_name"]
            or train.get("scene_index") != scene["scene_index"]
            or train.get("formal") is not True
            or train.get("num_iters") != 30000
        ):
            raise RuntimeError(f"training contract 不匹配：{variant}")
        checkpoint = train["checkpoint"]
        if checkpoint.get("step") != 30000 or not Path(checkpoint["checkpoint"]).is_file():
            raise RuntimeError(f"checkpoint contract 不匹配：{variant}")
        initialization = train.get("initialization_provenance", {})
        if initialization.get("sha256") != expected["paired_initialization_sha256"]:
            raise RuntimeError(f"配对初始化 SHA 不匹配：{variant}")
        if (
            endpoint.get("variant") != variant
            or endpoint.get("scene_name") != scene["scene_name"]
            or endpoint.get("endpoint_config_sha256") != expected["endpoint_config_sha256"]
            or Path(endpoint["source_run_dir"]).name != train_dir.name
        ):
            raise RuntimeError(f"endpoint contract 不匹配：{variant}")
        endpoint_metrics = endpoint["endpoint_metrics"]
        if (
            endpoint_metrics.get("checkpoint_sha256_before") != checkpoint["sha256"]
            or endpoint_metrics.get("checkpoint_sha256_after") != checkpoint["sha256"]
        ):
            raise RuntimeError(f"endpoint 改变或错绑 checkpoint：{variant}")
        diagnostic = diagnostics[variant]
        if Path(diagnostic["source"]["source_run_dir"]).name != train_dir.name:
            raise RuntimeError(f"diagnostic source 错绑：{variant}")
        if diagnostic["source"]["checkpoint"]["sha256"] != checkpoint["sha256"]:
            raise RuntimeError(f"diagnostic checkpoint 错绑：{variant}")

        heldout = train["heldout_metrics"]
        e1 = endpoint_metrics["e1"]
        if e1.get("status") != "done":
            raise RuntimeError(f"E1 不是 done：{variant}")
        e1_residual = e1["residual"]
        role_metrics = {role: actor_metrics(train, role) for role in ROLES}
        e2_metrics = {role: endpoint_role(endpoint, role) for role in ROLES}
        rows[variant] = {
            "variant": variant,
            "training_run_id": train_dir.name,
            "endpoint_run_id": endpoint_dir.name,
            "checkpoint_sha256": checkpoint["sha256"],
            "global_psnr": heldout["image_metrics/test/psnr"],
            "global_ssim": heldout["image_metrics/test/ssim"],
            "global_lpips": heldout["image_metrics/test/lpips"],
            "background_gaussians": checkpoint["background_gaussians"],
            "rigid_gaussians": checkpoint["rigid_gaussians"],
            "total_gaussians": checkpoint["background_gaussians"]
            + checkpoint["rigid_gaussians"],
            "checkpoint_bytes": checkpoint["bytes"],
            "train_seconds": train["train_resources"]["duration_seconds"],
            "train_peak_gpu_mib": train["train_resources"][
                "peak_gpu_memory_mib_sampled"
            ],
            "actors": role_metrics,
            "e1": {
                "candidate_count": e1["candidate_count"],
                "valid_count": e1["valid_count"],
                "coverage": e1["coverage"],
                "median": e1_residual["median"],
                "p90": e1_residual["p90"],
            },
            "e2": e2_metrics,
            "diagnostic": compact_diagnostic(diagnostic),
        }
    return rows


def e1_state(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, bool]:
    errors = ("median", "p90")
    support_ok = candidate["e1"]["coverage"] >= baseline["e1"]["coverage"]
    non_degraded = support_ok and all(
        candidate["e1"][metric] <= baseline["e1"][metric] for metric in errors
    )
    improved = support_ok and all(
        candidate["e1"][metric] < baseline["e1"][metric] for metric in errors
    )
    return {"improved": improved, "non_degraded": non_degraded, "support_ok": support_ok}


def e2_state(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    role_states: dict[str, dict[str, bool]] = {}
    for role in ROLES:
        support_ok = candidate["e2"][role]["coverage"] >= baseline["e2"][role]["coverage"]
        non_degraded = support_ok and all(
            candidate["e2"][role][metric] <= baseline["e2"][role][metric]
            for metric in ("mean", "p90")
        )
        improved = support_ok and all(
            candidate["e2"][role][metric] < baseline["e2"][role][metric]
            for metric in ("mean", "p90")
        )
        role_states[role] = {
            "improved": improved,
            "non_degraded": non_degraded,
            "support_ok": support_ok,
        }
    non_degraded = all(state["non_degraded"] for state in role_states.values())
    improved = non_degraded and any(state["improved"] for state in role_states.values())
    return {"improved": improved, "non_degraded": non_degraded, "roles": role_states}


def appearance_state(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    values = {
        "global_lpips": (candidate["global_lpips"], baseline["global_lpips"]),
        "boundary_actor_lpips": (
            candidate["actors"]["boundary-support"]["actor_lpips"],
            baseline["actors"]["boundary-support"]["actor_lpips"],
        ),
        "boundary_band_lpips": (
            candidate["actors"]["boundary-support"]["boundary_lpips"],
            baseline["actors"]["boundary-support"]["boundary_lpips"],
        ),
        "high_actor_lpips": (
            candidate["actors"]["high-support"]["actor_lpips"],
            baseline["actors"]["high-support"]["actor_lpips"],
        ),
        "high_boundary_lpips": (
            candidate["actors"]["high-support"]["boundary_lpips"],
            baseline["actors"]["high-support"]["boundary_lpips"],
        ),
    }
    per_metric = {name: current <= reference for name, (current, reference) in values.items()}
    return {"acceptable": all(per_metric.values()), "per_metric": per_metric}


def assess_alternative(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    e1 = e1_state(candidate, baseline)
    e2 = e2_state(candidate, baseline)
    appearance = appearance_state(candidate, baseline)
    primary_count = int(e1["improved"]) + int(e2["improved"])
    if e1["improved"] and not e2["improved"]:
        other_non_degraded = e2["non_degraded"]
    elif e2["improved"] and not e1["improved"]:
        other_non_degraded = e1["non_degraded"]
    elif e1["improved"] and e2["improved"]:
        other_non_degraded = True
    else:
        other_non_degraded = False
    eligible = primary_count > 0 and other_non_degraded and appearance["acceptable"]
    reasons: list[str] = []
    if primary_count == 0:
        reasons.append("NO_PRIMARY_ENDPOINT_IMPROVEMENT")
    if primary_count > 0 and not other_non_degraded:
        reasons.append("OTHER_PRIMARY_ENDPOINT_DEGRADED")
    if not appearance["acceptable"]:
        reasons.append("APPEARANCE_LPIPS_DEGRADED")
    return {
        "eligible": eligible,
        "improved_primary_endpoint_count": primary_count,
        "e1": e1,
        "e2": e2,
        "other_primary_endpoint_non_degraded": other_non_degraded,
        "appearance": appearance,
        "reasons": reasons,
    }


def select_candidate(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = rows["c0-off"]
    assessments = {
        variant: assess_alternative(rows[variant], baseline) for variant in VARIANTS[1:]
    }
    eligible = [variant for variant in VARIANTS[1:] if assessments[variant]["eligible"]]
    eligible.sort(
        key=lambda variant: (
            -assessments[variant]["improved_primary_endpoint_count"],
            rows[variant]["global_lpips"],
            rows[variant]["total_gaussians"],
            rows[variant]["train_seconds"],
            VARIANTS.index(variant),
        )
    )
    selected = eligible[0] if eligible else "c0-off"
    selected_is_alias = selected in ("c0-off", "c1-native")
    confirmation_unique_training_runs = 4 + 2 * (2 if selected_is_alias else 3)
    return {
        "selected_variant": selected,
        "decision_status": DECISION_STATUS[selected],
        "eligible_alternatives": eligible,
        "fallback_used": not eligible,
        "assessments": assessments,
        "confirmation": {
            "logical_matrix_entries": 10,
            "unique_training_runs": confirmation_unique_training_runs,
            "selected_c_star_is_exact_alias": selected_is_alias,
            "alias_source_variant": selected if selected_is_alias else None,
            "confirmation_variants": ["c0-off", "c1-native", selected],
        },
    }


def csv_rows(rows: dict[str, dict[str, Any]], decision: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for variant in VARIANTS:
        row = rows[variant]
        assessment = decision["assessments"].get(variant, {})
        result.append(
            {
                "variant": variant,
                "selected": variant == decision["selected_variant"],
                "eligible": assessment.get("eligible"),
                "e1_median": row["e1"]["median"],
                "e1_p90": row["e1"]["p90"],
                "e1_coverage": row["e1"]["coverage"],
                "e2_boundary_mean": row["e2"]["boundary-support"]["mean"],
                "e2_boundary_p90": row["e2"]["boundary-support"]["p90"],
                "e2_boundary_coverage": row["e2"]["boundary-support"]["coverage"],
                "e2_high_mean": row["e2"]["high-support"]["mean"],
                "e2_high_p90": row["e2"]["high-support"]["p90"],
                "e2_high_coverage": row["e2"]["high-support"]["coverage"],
                "global_psnr": row["global_psnr"],
                "global_ssim": row["global_ssim"],
                "global_lpips": row["global_lpips"],
                "boundary_actor_lpips": row["actors"]["boundary-support"]["actor_lpips"],
                "boundary_band_lpips": row["actors"]["boundary-support"]["boundary_lpips"],
                "high_actor_lpips": row["actors"]["high-support"]["actor_lpips"],
                "high_boundary_lpips": row["actors"]["high-support"]["boundary_lpips"],
                "background_gaussians": row["background_gaussians"],
                "rigid_gaussians": row["rigid_gaussians"],
                "total_gaussians": row["total_gaussians"],
                "train_seconds": row["train_seconds"],
                "train_peak_gpu_mib": row["train_peak_gpu_mib"],
                "pose_translation_p90_m": row["diagnostic"]["pose_translation_p90_m"],
                "pose_rotation_p90_deg": row["diagnostic"]["pose_rotation_p90_deg"],
                "isp_residual_p90": row["diagnostic"]["isp_residual_p90"],
                "reasons": ";".join(assessment.get("reasons", [])),
            }
        )
    return result


def markdown_report(rows: dict[str, dict[str, Any]], decision: dict[str, Any]) -> str:
    lines = [
        "# WorldSim V3 A1 开发场景选择",
        "",
        "本判定在 scene-0230 结果已可见后、确认场景启动前冻结；使用无容差严格 Pareto 比较。",
        "",
        "| 变体 | E1 median / P90 / coverage | E2 boundary mean / P90 | E2 high mean / P90 | global PSNR / LPIPS | total GS | train min | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANTS:
        row = rows[variant]
        assessment = decision["assessments"].get(variant)
        conclusion = "selected baseline" if variant == "c0-off" else (
            "eligible" if assessment["eligible"] else ", ".join(assessment["reasons"])
        )
        lines.append(
            "| {variant} | {e1m:.5f} / {e1p:.5f} / {e1c:.3%} | "
            "{e2bm:.6f} / {e2bp:.6f} | {e2hm:.6f} / {e2hp:.6f} | "
            "{psnr:.3f} / {lpips:.4f} | {gs:,} | {minutes:.2f} | {conclusion} |".format(
                variant=variant,
                e1m=row["e1"]["median"],
                e1p=row["e1"]["p90"],
                e1c=row["e1"]["coverage"],
                e2bm=row["e2"]["boundary-support"]["mean"],
                e2bp=row["e2"]["boundary-support"]["p90"],
                e2hm=row["e2"]["high-support"]["mean"],
                e2hp=row["e2"]["high-support"]["p90"],
                psnr=row["global_psnr"],
                lpips=row["global_lpips"],
                gs=row["total_gaussians"],
                minutes=row["train_seconds"] / 60.0,
                conclusion=conclusion,
            )
        )
    lines.extend(
        [
            "",
            f"冻结结论：`C*={decision['selected_variant']}`，终态 `{decision['decision_status']}`。",
            "",
            "- C2 仅改善 boundary role E2，high role E2 与 actor/boundary LPIPS 退化，不能算整个 E2 端点改善。",
            "- C3 的全图、actor/boundary 画质和位姿修正稳定性最好，但 E1 与两个 E2 role 均未严格优于 C0。",
            "- 因无替代候选通过主端点合同，回退 C0；这是正式负结果，不把复杂度当作改进。",
            "- 确认场景的 C* 项登记为 C0 source run/checkpoint 的 exact alias；逻辑矩阵仍为 10 项，唯一训练为 8 个。",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_selection(
    contract: dict[str, Any], task_runs_root: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    sources = resolve_sources(contract, task_runs_root)
    rows = validate_and_build_rows(contract, sources)
    decision = select_candidate(rows)
    payload = {
        "status": "done",
        "task_id": contract["task_id"],
        "selection_version": contract["selection_version"],
        "scene_name": contract["scope"]["scene_name"],
        "scene_index": contract["scope"]["scene_index"],
        "truth_boundary": contract["result_access_disclosure"],
        "source_contract": contract["source_contract"],
        "rows": rows,
        "decision": decision,
        "completed_at": now(),
    }
    output_dir.mkdir(parents=True)
    atomic_json(output_dir / "selection.json", payload)
    flat_rows = csv_rows(rows, decision)
    with (output_dir / "pareto_matrix.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    (output_dir / "report.md").write_text(
        markdown_report(rows, decision), encoding="utf-8"
    )
    return payload


def resource_guard(run_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(run_root)
    if usage.free < 10 * 1024**3:
        raise RuntimeError(f"{run_root} 剩余空间少于 10 GiB")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if processes:
        raise RuntimeError(f"GPU 存在活动计算进程：{processes}")
    return {
        "free_disk_bytes": usage.free,
        "minimum_free_disk_bytes": 10 * 1024**3,
        "active_gpu_compute_processes": processes,
    }


def formal_main(args: argparse.Namespace) -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    if args.run_dir is None:
        raise ValueError("formal 模式必须提供 --run-dir")
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    contract = yaml.safe_load(args.selection_config.read_text(encoding="utf-8"))
    validate_selection_contract(contract)
    sources = resolve_sources(contract, args.task_runs_root)
    guard = resource_guard(args.run_dir.parent)
    _ACTIVE_RUN_DIR = args.run_dir
    for name in ("artifacts", "environment", "logs", "source_snapshot", "stages"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    resolved = args.run_dir / "resolved.yaml"
    shutil.copy2(args.selection_config, resolved)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )
    source_files = (
        PROJECT / "scripts/finalize_worldsim_v3_a1_dev.py",
        PROJECT / "scripts/run_worldsim_v3_a0_scene.py",
        PROJECT / "scripts/run_worldsim_v3_a0_smoke.py",
        args.selection_config,
    )
    source_hashes: dict[str, str] = {}
    for source in source_files:
        relative = source.relative_to(PROJECT)
        destination = args.run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[str(relative)] = sha256_file(source)
    evidence_hashes: dict[str, Any] = {"training": {}, "endpoint": {}}
    for kind in ("training", "endpoint"):
        for variant, run_dir in sources[kind].items():
            evidence_hashes[kind][variant] = {
                "run_dir": str(run_dir),
                "summary_sha256": sha256_file(run_dir / "summary.json"),
                "terminal_sha256": sha256_file(run_dir / "terminal.json"),
            }
    diagnostic_dir = sources["diagnostic"]
    evidence_hashes["diagnostic"] = {
        "run_dir": str(diagnostic_dir),
        "summary_sha256": sha256_file(diagnostic_dir / "summary.json"),
        "terminal_sha256": sha256_file(diagnostic_dir / "terminal.json"),
    }
    commit = command_output("git", "rev-parse", "HEAD", cwd=PROJECT)
    fingerprint = {
        "schema_version": 1,
        "task_id": contract["task_id"],
        "project_commit": commit,
        "selection_config_sha256": sha256_file(resolved),
        "source_hashes": source_hashes,
        "evidence_hashes": evidence_hashes,
    }
    atomic_json(args.run_dir / "fingerprint.json", fingerprint)
    output_dir = args.run_dir / "artifacts/a1_dev_selection"
    command = [
        str(MOTIONPROJ_PYTHON),
        str(PROJECT / "scripts/finalize_worldsim_v3_a1_dev.py"),
        "--worker-output",
        str(output_dir),
        "--selection-config",
        str(resolved),
        "--task-runs-root",
        str(args.task_runs_root),
    ]
    manifest = {
        "schema_version": 1,
        "task_id": contract["task_id"],
        "component": "A1 development-scene Pareto selection",
        "status": "running",
        "scene_name": contract["scope"]["scene_name"],
        "scene_index": contract["scope"]["scene_index"],
        "seed": contract["scope"]["seed"],
        "split": contract["scope"]["split"],
        "project_commit": commit,
        "project_status": command_output("git", "status", "--short", cwd=PROJECT).splitlines(),
        "resource_guard": guard,
        "source_runs": contract["source_contract"],
        "command": command,
        "started_at": now(),
    }
    atomic_json(args.run_dir / "manifest.json", manifest)

    def validate() -> tuple[bool, dict[str, Any]]:
        selection_path = output_dir / "selection.json"
        csv_path = output_dir / "pareto_matrix.csv"
        report_path = output_dir / "report.md"
        if not selection_path.is_file() or not csv_path.is_file() or not report_path.is_file():
            return False, {"selection": str(selection_path), "csv": str(csv_path), "report": str(report_path)}
        payload = load_json(selection_path)
        decision = payload.get("decision", {})
        ok = (
            payload.get("status") == "done"
            and set(payload.get("rows", {})) == set(VARIANTS)
            and decision.get("selected_variant") in VARIANTS
            and decision.get("decision_status") in set(DECISION_STATUS.values())
        )
        return ok, {
            "selection": str(selection_path),
            "selection_sha256": sha256_file(selection_path),
            "pareto_csv": str(csv_path),
            "report": str(report_path),
            "selected_variant": decision.get("selected_variant"),
            "decision_status": decision.get("decision_status"),
        }

    stage = run_stage(
        run_dir=args.run_dir,
        stage="a1_dev_selection",
        command=command,
        cwd=PROJECT,
        environment=common_environment(),
        validate=validate,
        timeout_seconds=args.timeout_seconds,
    )
    selection = load_json(output_dir / "selection.json")
    shutil.copy2(output_dir / "selection.json", args.run_dir / "metrics.json")
    shutil.copy2(output_dir / "report.md", args.run_dir / "summary.md")
    artifacts = {
        "selection": str(output_dir / "selection.json"),
        "selection_sha256": sha256_file(output_dir / "selection.json"),
        "pareto_matrix_csv": str(output_dir / "pareto_matrix.csv"),
        "pareto_matrix_csv_sha256": sha256_file(output_dir / "pareto_matrix.csv"),
        "report": str(output_dir / "report.md"),
        "report_sha256": sha256_file(output_dir / "report.md"),
    }
    atomic_json(args.run_dir / "artifacts.json", artifacts)
    summary = {
        "status": "done",
        "task_id": contract["task_id"],
        "component": "A1 development-scene frozen C* selection",
        "scene_name": contract["scope"]["scene_name"],
        "scene_index": contract["scope"]["scene_index"],
        "seed": contract["scope"]["seed"],
        "selection_config_sha256": fingerprint["selection_config_sha256"],
        "selected_variant": selection["decision"]["selected_variant"],
        "decision_status": selection["decision"]["decision_status"],
        "decision": selection["decision"],
        "artifacts": artifacts,
        "resources": {
            key: stage[key]
            for key in (
                "duration_seconds",
                "peak_gpu_memory_mib_sampled",
                "peak_gpu_memory_mib_torch_log",
                "peak_cgroup_memory_bytes",
            )
        },
        "completed_at": now(),
    }
    atomic_json(args.run_dir / "summary.json", summary)
    atomic_json(
        args.run_dir / "terminal.json",
        {"status": "done", "updated_at": now(), "failure": None},
    )
    _TERMINAL_FINAL = True
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=PROJECT / "configs/worldsim_v3/a1_dev_selection_v1.yaml",
    )
    parser.add_argument(
        "--task-runs-root",
        type=Path,
        default=Path("/root/autodl-tmp/runs/worldsim_v3/WS-V3-A1-CALIBRATION-01"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.worker_output is not None:
        contract = yaml.safe_load(args.selection_config.read_text(encoding="utf-8"))
        validate_selection_contract(contract)
        payload = evaluate_selection(contract, args.task_runs_root, args.worker_output)
        print(json.dumps(payload["decision"], indent=2, sort_keys=True))
        return
    formal_main(args)


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
                        "code": "A1_DEV_SELECTION_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
            )
        raise

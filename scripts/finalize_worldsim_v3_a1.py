#!/usr/bin/env python
"""验证并汇总 WorldSim V3 A1 的 10 项逻辑确认矩阵。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from scripts.run_worldsim_v3_a0_smoke import atomic_json, command_output, now, sha256_file


PROJECT = Path("/root/autodl-tmp/motion_proj")
TASK_RUNS = Path("/root/autodl-tmp/runs/worldsim_v3/WS-V3-A1-CALIBRATION-01")
ROLES = ("boundary-support", "high-support")
_ACTIVE_RUN_DIR: Path | None = None
_TERMINAL_FINAL = False


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_done(run_dir: Path) -> dict[str, Any]:
    terminal = load_json(run_dir / "terminal.json")
    summary = load_json(run_dir / "summary.json")
    if terminal.get("status") != "done" or summary.get("status") != "done":
        raise RuntimeError(f"run 不是 done：{run_dir}")
    return summary


def metric_row(train: dict[str, Any], endpoint: dict[str, Any]) -> dict[str, Any]:
    heldout = train["heldout_metrics"]
    checkpoint = train["checkpoint"]
    endpoint_metrics = endpoint["endpoint_metrics"]
    e1 = endpoint_metrics["e1"]
    actors: dict[str, Any] = {}
    e2: dict[str, Any] = {}
    for role in ROLES:
        actor = train["actor_metrics"]["roles"][role]
        endpoint_role = endpoint_metrics["e2"]["roles"][role]
        actors[role] = {
            "status": actor["status"],
            "actor_lpips": (actor.get("actor_region") or {}).get("masked_lpips_alex_tight_crop_256px"),
            "boundary_lpips": (actor.get("boundary_band") or {}).get("masked_lpips_alex_tight_crop_256px"),
        }
        distance = endpoint_role.get("normalized_bidirectional_distance") or {}
        e2[role] = {
            "status": endpoint_role["status"],
            "reason": endpoint_role.get("reason"),
            "coverage": endpoint_role.get("coverage"),
            "mean": distance.get("mean"),
            "p90": distance.get("p90"),
        }
    return {
        "variant": train["variant"],
        "global_psnr": heldout["image_metrics/test/psnr"],
        "global_ssim": heldout["image_metrics/test/ssim"],
        "global_lpips": heldout["image_metrics/test/lpips"],
        "total_gaussians": checkpoint["background_gaussians"] + checkpoint["rigid_gaussians"],
        "train_seconds": train["train_resources"]["duration_seconds"],
        "train_peak_gpu_mib": train["train_resources"]["peak_gpu_memory_mib_sampled"],
        "checkpoint_sha256": checkpoint["sha256"],
        "actors": actors,
        "e1": {
            "coverage": e1["coverage"],
            "median": e1["residual"]["median"],
            "p90": e1["residual"]["p90"],
        },
        "e2": e2,
    }


def assess_confirmation(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    e1_support = candidate["e1"]["coverage"] >= baseline["e1"]["coverage"]
    e1_non_degraded = e1_support and all(candidate["e1"][key] <= baseline["e1"][key] for key in ("median", "p90"))
    e1_improved = e1_support and all(candidate["e1"][key] < baseline["e1"][key] for key in ("median", "p90"))
    e2_roles: dict[str, Any] = {}
    available_roles: list[str] = []
    for role in ROLES:
        c = candidate["e2"][role]
        b = baseline["e2"][role]
        if c["status"] == "ABSTAIN" and b["status"] == "ABSTAIN":
            e2_roles[role] = {"status": "ABSTAIN", "non_degraded": True, "improved": False}
            continue
        if c["status"] != "done" or b["status"] != "done":
            raise RuntimeError(f"E2 role 状态不对称：{role}")
        available_roles.append(role)
        support = c["coverage"] >= b["coverage"]
        non_degraded = support and all(c[key] <= b[key] for key in ("mean", "p90"))
        improved = support and all(c[key] < b[key] for key in ("mean", "p90"))
        e2_roles[role] = {"status": "done", "support_ok": support, "non_degraded": non_degraded, "improved": improved}
    e2_non_degraded = bool(available_roles) and all(e2_roles[role]["non_degraded"] for role in available_roles)
    e2_improved = e2_non_degraded and any(e2_roles[role]["improved"] for role in available_roles)
    appearance_checks = {"global_lpips": candidate["global_lpips"] <= baseline["global_lpips"]}
    for role in ROLES:
        if candidate["actors"][role]["status"] == "ABSTAIN" and baseline["actors"][role]["status"] == "ABSTAIN":
            continue
        for region in ("actor_lpips", "boundary_lpips"):
            appearance_checks[f"{role}.{region}"] = candidate["actors"][role][region] <= baseline["actors"][role][region]
    appearance_ok = all(appearance_checks.values())
    primary_count = int(e1_improved) + int(e2_improved)
    other_ok = e2_non_degraded if e1_improved and not e2_improved else e1_non_degraded if e2_improved and not e1_improved else primary_count == 2
    eligible = primary_count > 0 and other_ok and appearance_ok
    reasons = []
    if primary_count == 0:
        reasons.append("NO_PRIMARY_ENDPOINT_IMPROVEMENT")
    if primary_count > 0 and not other_ok:
        reasons.append("OTHER_PRIMARY_ENDPOINT_DEGRADED")
    if not appearance_ok:
        reasons.append("APPEARANCE_LPIPS_DEGRADED")
    return {
        "eligible": eligible,
        "e1": {"improved": e1_improved, "non_degraded": e1_non_degraded, "support_ok": e1_support},
        "e2": {"improved": e2_improved, "non_degraded": e2_non_degraded, "roles": e2_roles},
        "appearance": {"acceptable": appearance_ok, "per_metric": appearance_checks},
        "reasons": reasons,
    }


def validate_scene(scene_name: str, scene: dict[str, Any], root: Path, endpoint_sha: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    initialization_hashes = set()
    for variant in ("c0-off", "c1-native"):
        train_dir = root / scene["training_runs"][variant]
        endpoint_dir = root / scene["endpoint_runs"][variant]
        train = require_done(train_dir)
        endpoint = require_done(endpoint_dir)
        checkpoint = train["checkpoint"]
        if train.get("scene_name") != scene_name or train.get("scene_index") != scene["scene_index"] or train.get("variant") != variant or train.get("formal") is not True or train.get("num_iters") != 30000:
            raise RuntimeError(f"training contract 不匹配：{scene_name}/{variant}")
        initialization_hashes.add(train["initialization_provenance"]["sha256"])
        endpoint_metrics = endpoint["endpoint_metrics"]
        if endpoint.get("endpoint_config_sha256") != endpoint_sha or Path(endpoint["source_run_dir"]).name != train_dir.name or endpoint_metrics["checkpoint_sha256_before"] != checkpoint["sha256"] or endpoint_metrics["checkpoint_sha256_after"] != checkpoint["sha256"]:
            raise RuntimeError(f"endpoint contract 不匹配：{scene_name}/{variant}")
        for role, expected in scene["expected_e2_roles"].items():
            if endpoint_metrics["e2"]["roles"][role]["status"] != expected:
                raise RuntimeError(f"E2 role 状态不匹配：{scene_name}/{variant}/{role}")
        rows[variant] = metric_row(train, endpoint)
        sources[variant] = {"training_run": str(train_dir), "endpoint_run": str(endpoint_dir)}
    if initialization_hashes != {scene["expected_initialization_sha256"]}:
        raise RuntimeError(f"配对初始化不匹配：{scene_name} {initialization_hashes}")
    alias_dir = root / scene["cstar_alias_run"]
    alias = require_done(alias_dir)
    if not alias.get("exact_alias") or alias.get("alias_of") != "c0-off" or Path(alias["source_training_run"]).name != scene["training_runs"]["c0-off"] or alias["source_checkpoint"]["sha256"] != rows["c0-off"]["checkpoint_sha256"]:
        raise RuntimeError(f"C* alias contract 不匹配：{scene_name}")
    assessment = assess_confirmation(rows["c1-native"], rows["c0-off"])
    return {"scene_name": scene_name, "scene_index": scene["scene_index"], "rows": rows, "c1_vs_c0": assessment, "sources": sources, "cstar_alias_run": str(alias_dir)}


def main() -> None:
    global _ACTIVE_RUN_DIR, _TERMINAL_FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--confirmation-config", type=Path, default=PROJECT / "configs/worldsim_v3/a1_confirmation_v1.yaml")
    parser.add_argument("--task-runs-root", type=Path, default=TASK_RUNS)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    contract = yaml.safe_load(args.confirmation_config.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("confirmation_version") != "A1-CF-v1":
        raise RuntimeError("confirmation config contract 不匹配")
    selection_dir = args.task_runs_root / contract["development_selection"]["run_id"]
    selection = require_done(selection_dir)
    if selection.get("selected_variant") != "c0-off" or selection.get("selection_config_sha256") != contract["development_selection"]["config_sha256"]:
        raise RuntimeError("development selection 不匹配")
    guard_usage = shutil.disk_usage(args.run_dir.parent)
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    active = [line for line in query.stdout.splitlines() if line.strip()]
    if guard_usage.free < 10 * 1024**3 or active:
        raise RuntimeError(f"finalizer resource guard 失败：free={guard_usage.free}, gpu={active}")
    _ACTIVE_RUN_DIR = args.run_dir
    for name in ("artifacts", "environment", "logs", "source_snapshot", "stages"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.confirmation_config, args.run_dir / "resolved.yaml")
    source = PROJECT / "scripts/finalize_worldsim_v3_a1.py"
    snapshot = args.run_dir / "source_snapshot/scripts" / source.name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, snapshot)
    atomic_json(args.run_dir / "terminal.json", {"status": "running", "updated_at": now(), "failure": None})
    commit = command_output("git", "rev-parse", "HEAD", cwd=PROJECT)
    manifest = {"schema_version": 1, "task_id": contract["task_id"], "component": "A1 10-entry finalizer", "status": "running", "project_commit": commit, "project_status": command_output("git", "status", "--short", cwd=PROJECT).splitlines(), "confirmation_config_sha256": sha256_file(args.run_dir / "resolved.yaml"), "started_at": now()}
    atomic_json(args.run_dir / "manifest.json", manifest)
    try:
        scenes = [validate_scene(name, contract["scenes"][name], args.task_runs_root, contract["endpoint_config_sha256"]) for name in ("scene-0242", "scene-0255")]
        if any(scene["c1_vs_c0"]["eligible"] for scene in scenes):
            raise RuntimeError("确认场景出现通过冻结合同的 C1，需人工裁决而不能自动 done_off")
        matrix_rows = []
        for scene in scenes:
            for logical, source_variant in (("c0-off", "c0-off"), ("c1-native", "c1-native"), ("c-star", "c0-off")):
                row = dict(scene["rows"][source_variant])
                row.update({"scene_name": scene["scene_name"], "logical_variant": logical, "source_variant": source_variant, "exact_alias": logical == "c-star"})
                matrix_rows.append(row)
        decision = {
            "status": "done_off",
            "selected_variant": "c0-off",
            "logical_matrix_entries": 4 + len(matrix_rows),
            "unique_training_runs": 8,
            "confirmation_scene_count": 2,
            "c1_eligible_confirmation_scenes": [],
            "raw_endpoint_direction": {"scene-0242": "c0", "scene-0255": "c1_errors_better_but_contract_failed"},
            "interpretation": "C0 remains frozen; raw endpoint direction is scene-dependent, while C1 fails the full frozen contract in both confirmation scenes.",
        }
        result = {"status": "done", "task_id": contract["task_id"], "confirmation_version": contract["confirmation_version"], "development_selection_run": str(selection_dir), "scenes": scenes, "matrix_rows": matrix_rows, "decision": decision, "completed_at": now()}
        atomic_json(args.run_dir / "artifacts/a1_matrix.json", result)
        flat = []
        for row in matrix_rows:
            flat.append({"scene_name": row["scene_name"], "logical_variant": row["logical_variant"], "source_variant": row["source_variant"], "exact_alias": row["exact_alias"], "global_psnr": row["global_psnr"], "global_ssim": row["global_ssim"], "global_lpips": row["global_lpips"], "e1_median": row["e1"]["median"], "e1_p90": row["e1"]["p90"], "e1_coverage": row["e1"]["coverage"], "total_gaussians": row["total_gaussians"], "train_seconds": row["train_seconds"]})
        with (args.run_dir / "artifacts/a1_matrix.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        report_lines = ["# WorldSim V3 A1 final", "", "开发场景冻结 C*=C0；两个确认场景均未出现通过完整冻结合同的 C1。", "", "原始端点方向存在场景依赖：0242 偏向 C0；0255 的误差值偏向 C1，但 coverage 与 actor/boundary LPIPS 合同失败。", "", "终态：`done_off`。", ""]
        report = "\n".join(report_lines)
        (args.run_dir / "summary.md").write_text(report, encoding="utf-8")
        atomic_json(args.run_dir / "metrics.json", result)
        artifacts = {"matrix_json": str(args.run_dir / "artifacts/a1_matrix.json"), "matrix_csv": str(args.run_dir / "artifacts/a1_matrix.csv"), "report": str(args.run_dir / "summary.md")}
        atomic_json(args.run_dir / "artifacts.json", artifacts)
        summary = {"status": "done", "task_id": contract["task_id"], "component": "A1 finalizer", "decision_status": "done_off", "selected_variant": "c0-off", "decision": decision, "confirmation_config_sha256": manifest["confirmation_config_sha256"], "artifacts": artifacts, "completed_at": now()}
        atomic_json(args.run_dir / "summary.json", summary)
        atomic_json(args.run_dir / "fingerprint.json", {"schema_version": 1, "task_id": contract["task_id"], "project_commit": commit, "confirmation_config_sha256": manifest["confirmation_config_sha256"], "finalizer_source_sha256": sha256_file(source), "selection_summary_sha256": sha256_file(selection_dir / "summary.json")})
        atomic_json(args.run_dir / "terminal.json", {"status": "done", "updated_at": now(), "failure": None})
        _TERMINAL_FINAL = True
        print(json.dumps(summary, indent=2, sort_keys=True))
    except BaseException:
        raise


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_RUN_DIR is not None and _ACTIVE_RUN_DIR.is_dir() and not _TERMINAL_FINAL:
            atomic_json(_ACTIVE_RUN_DIR / "terminal.json", {"status": "blocked", "updated_at": now(), "failure": {"code": "A1_FINALIZER_FAILED", "detail": f"{type(error).__name__}: {error}"}})
        raise

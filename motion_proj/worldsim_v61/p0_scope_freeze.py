"""WorldSim V6.1 P0：冻结范围、R10 基线与 ME-0 证据分层。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


TASK_ID = "WS-V61-P0-SCOPE-FREEZE-01"
RUNS_ROOT = Path("/root/autodl-tmp/runs")


class P0ContractError(RuntimeError):
    """P0 冻结合同被破坏。"""


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _resolve_uri(repo_root: Path, uri: str) -> Path:
    if uri.startswith("runs://"):
        relative = Path(uri.removeprefix("runs://"))
        if ".." in relative.parts:
            raise P0ContractError("runs URI 不得包含上级路径")
        return (RUNS_ROOT / relative).resolve()
    if uri.startswith("repo://"):
        relative = Path(uri.removeprefix("repo://"))
        if ".." in relative.parts:
            raise P0ContractError("repo URI 不得包含上级路径")
        return (repo_root / relative).resolve()
    return Path(uri).resolve()


def _verify(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise P0ContractError(f"冻结源漂移: {path}")


def _evaluate_r10_baseline(
    metrics: Mapping[str, Any], decisions: Iterable[Mapping[str, Any]], expected: Mapping[str, Any]
) -> dict[str, Any]:
    rows = list(decisions)
    accepted = sorted(str(row["case_id"]) for row in rows if row["overall_decision"] == "ACCEPT")
    abstained = sorted(str(row["case_id"]) for row in rows if row["overall_decision"] == "ABSTAIN")
    rejected = sorted(str(row["case_id"]) for row in rows if row["overall_decision"] == "REJECT")
    false_safe = [str(row["case_id"]) for row in rows if bool(row["false_safe"])]
    checks = {
        "case_count_exact": len(rows) == int(expected["case_count"]) == int(metrics["case_count"]),
        "case_ids_unique": len({row["case_id"] for row in rows}) == len(rows),
        "accept_count_exact": len(accepted) == int(expected["accept_count"]) == int(metrics["accept_count"]),
        "abstain_count_exact": len(abstained) == int(expected["abstain_count"]) == int(metrics["abstain_count"]),
        "reject_count_exact": len(rejected) == int(expected["reject_count"]) == int(metrics["reject_count"]),
        "false_safe_zero": not false_safe and int(metrics["false_safe_count"]) == 0,
        "accepted_mask_pixels_exact": int(metrics["accepted_mask_pixels"])
        == int(expected["accepted_mask_pixels"]),
        "accepted_ids_exact": accepted == sorted(str(value) for value in expected["accepted_case_ids"]),
    }
    checks["passed"] = all(checks.values())
    return {
        "schema_version": "worldsim_v61.r10_baseline.v1",
        "checks": checks,
        "accepted_case_ids": accepted,
        "abstained_case_ids": abstained,
        "rejected_case_ids": rejected,
        "false_safe_case_ids": false_safe,
        "metrics": dict(metrics),
    }


def _source_index(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method_paths: set[str] = set()
    eval_paths: set[str] = set()
    cohort = config["cohort"]
    for scene_name, scene_spec in sorted(config["raw_evidence"].items()):
        scene_root = Path(scene_spec["processed_scene_root"])
        for relative in (
            "extrinsics/000_0.txt",
            "instances/instances_info.json",
            "instances/frame_instances.json",
        ):
            path = scene_root / relative
            if not path.is_file():
                raise P0ContractError(f"缺少 raw evidence: {path}")
            rows.append({"tier": "shared_metadata", "scene": scene_name, "path": str(path), "bytes": path.stat().st_size})
        for target_frame in cohort["frame_indices"]:
            split = scene_spec["sweep_offsets_by_target"][int(target_frame)]
            for tier, offsets in (("O_method", split["method"]), ("O_eval", split["eval"])):
                for offset in offsets:
                    frame = int(target_frame) + int(offset)
                    for kind, suffix in (("lidar", ".bin"), ("lidar_pose", ".txt")):
                        path = scene_root / kind / f"{frame:03d}{suffix}"
                        if not path.is_file():
                            raise P0ContractError(f"缺少 {tier} evidence: {path}")
                        rows.append(
                            {
                                "tier": tier,
                                "scene": scene_name,
                                "target_frame": int(target_frame),
                                "source_frame": frame,
                                "kind": kind,
                                "path": str(path),
                                "bytes": path.stat().st_size,
                            }
                        )
                        (method_paths if tier == "O_method" else eval_paths).add(str(path))
    disjoint = method_paths.isdisjoint(eval_paths)
    return rows, {
        "method_file_count": len(method_paths),
        "eval_file_count": len(eval_paths),
        "method_eval_path_disjoint": disjoint,
        "method_index_sha256": _content_sha256(sorted(method_paths)),
        "eval_index_sha256": _content_sha256(sorted(eval_paths)),
    }


def _prepare_run_root(run_root: Path) -> float:
    """先建立精确 namespace，避免首次路线启动在资源审计前失败。"""
    run_root.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(run_root).free / 1024**3


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise P0ContractError("正式 P0 run 要求干净工作树")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise P0ContractError("P0 task_id 漂移")

    source_paths: dict[str, Path] = {}
    for name, spec in config["sources"].items():
        path = _resolve_uri(repo_root, spec["uri"])
        _verify(path, spec["sha256"])
        source_paths[name] = path

    r10_run = source_paths["r10_manifest"].parent
    r9_run = source_paths["r9_manifest"].parent
    metrics = json.loads((r10_run / "METRICS.json").read_text(encoding="utf-8"))
    decisions = _read_jsonl(r10_run / "FACTORIZED_DECISIONS.jsonl")
    r9_cases = _read_jsonl(r9_run / "CASES.jsonl")
    baseline = _evaluate_r10_baseline(metrics, decisions, config["r10_baseline"])
    r9_ids = sorted(row["case_id"] for row in r9_cases)
    r10_ids = sorted(row["case_id"] for row in decisions)
    index_rows, tier_audit = _source_index(config)

    free_gib = _prepare_run_root(run_root)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__scope-freeze-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        checks = {
            "plan_and_v6_authorities_exact": all(
                _sha256(path) == config["sources"][name]["sha256"]
                for name, path in source_paths.items()
            ),
            "r10_baseline_exact": bool(baseline["checks"]["passed"]),
            "r9_r10_case_identity_exact": r9_ids == r10_ids,
            "method_eval_source_paths_disjoint": bool(tier_audit["method_eval_path_disjoint"]),
            "scene_mapping_exact": {
                name: int(spec["scene_index"]) for name, spec in config["raw_evidence"].items()
            }
            == config["expected_scene_mapping"],
            "failure_ledger_refs_present": len(config["failure_ledger_refs"]) >= 5,
            "confirmation_locked": bool(config["cohort"]["confirmation_locked"]),
            "no_generator_training_or_gpu_started": True,
            "disk_budget_sufficient": free_gib >= float(config["resources"]["minimum_disk_free_gib"]),
        }
        wall_seconds = time.monotonic() - started
        checks["wall_within_budget"] = wall_seconds <= float(config["resources"]["maximum_wall_seconds"])
        checks["passed"] = all(checks.values())

        scope = {
            "schema_version": "worldsim_v61.scope_freeze.v1",
            "task_id": TASK_ID,
            "source_commit": source_commit,
            "north_star": config["north_star"],
            "minimum_experiment_question": config["minimum_experiment_question"],
            "stage_order": config["stage_order"],
            "stop_rules": config["stop_rules"],
            "failure_ledger_refs": config["failure_ledger_refs"],
            "cohort": config["cohort"],
            "truth_tiers": config["truth_tiers"],
            "forbidden_shortcuts": config["forbidden_shortcuts"],
        }
        _write_json(run_dir / "SCOPE_FREEZE.json", scope)
        _write_json(run_dir / "R10_BASELINE.json", baseline)
        _write_json(
            run_dir / "SOURCE_INDEX.json",
            {
                "schema_version": "worldsim_v61.source_index.v1",
                "files": index_rows,
                "tier_audit": tier_audit,
                "index_sha256": _content_sha256(index_rows),
            },
        )
        _write_json(
            run_dir / "P0_GATE.json",
            {
                "schema_version": "worldsim_v61.p0_gate.v1",
                "checks": checks,
                "decision": "proceed_to_sceneir_o_and_me0" if checks["passed"] else "repair_scope_or_sources",
            },
        )
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v61.p0_resource_audit.v1",
                "gpu_used": False,
                "training_started": False,
                "generator_started": False,
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
            },
        )
        status = "done" if checks["passed"] else "rejected"
        _write_json(
            run_dir / "SUMMARY.json",
            {
                "schema_version": "worldsim_v61.p0_summary.v1",
                "task_id": TASK_ID,
                "hypothesis_id": config["hypothesis_id"],
                "status": status,
                "hypothesis_outcome": "accepted_scope_freeze" if checks["passed"] else "rejected",
                "source_commit": source_commit,
                "r10_accept_count": metrics["accept_count"],
                "r10_case_count": metrics["case_count"],
                "r10_false_safe_count": metrics["false_safe_count"],
                "method_eval_path_disjoint": tier_audit["method_eval_path_disjoint"],
                "failure_ledger_delta": "none" if checks["passed"] else "pending_rejected_closeout",
                "next": "WS-V61-ME0-OCCIR-01" if checks["passed"] else "repair_scope_or_sources",
            },
        )
        tracked = [
            "SCOPE_FREEZE.json",
            "R10_BASELINE.json",
            "SOURCE_INDEX.json",
            "P0_GATE.json",
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v61.p0_manifest.v1",
                "files": {
                    name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
                    for name in tracked
                },
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v61.terminal.v1",
                "status": status,
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
            },
        )
        print(run_dir, flush=True)
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v61.terminal.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise

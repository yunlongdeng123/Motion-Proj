"""WorldSim V6 R13 实景 SceneIR 类型化动态编辑实验。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from motion_proj.worldsim_v6.r12_dynamic_logsim import _replay_once


TASK_ID = "WS-V6-R13-WORLDSIM-01"
FACTOR_NAMES = ("actor_states", "trajectories", "semantic_labels", "collision_labels")


class R13DynamicEditError(RuntimeError):
    """R13 dynamic-edit 正式合同失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R13DynamicEditError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def _factor_view(replay: Mapping[str, Any]) -> dict[str, Any]:
    return {name: replay[name] for name in FACTOR_NAMES}


def _collision_rows(actor_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_time: dict[int, list[dict[str, Any]]] = {}
    for row in actor_states:
        by_time.setdefault(int(row["timestamp_us"]), []).append(row)
    result: list[dict[str, Any]] = []
    for timestamp, states in sorted(by_time.items()):
        ordered = sorted(states, key=lambda row: row["actor_id"])
        for left_index in range(len(ordered)):
            left = ordered[left_index]
            left_min = np.asarray(left["aabb_min_world_m"], dtype=np.float64)
            left_max = np.asarray(left["aabb_max_world_m"], dtype=np.float64)
            for right in ordered[left_index + 1 :]:
                right_min = np.asarray(right["aabb_min_world_m"], dtype=np.float64)
                right_max = np.asarray(right["aabb_max_world_m"], dtype=np.float64)
                overlap = bool(
                    np.all(np.minimum(left_max, right_max) >= np.maximum(left_min, right_min))
                )
                result.append(
                    {
                        "timestamp_us": timestamp,
                        "actor_pair": [left["actor_id"], right["actor_id"]],
                        "aabb_overlap": overlap,
                    }
                )
    return result


def _translated(values: list[float], delta: list[float]) -> list[float]:
    return (np.asarray(values, dtype=np.float64) + np.asarray(delta, dtype=np.float64)).tolist()


def _sort_factors(replay: dict[str, Any]) -> None:
    replay["actor_states"].sort(key=lambda row: (row["actor_id"], int(row["timestamp_us"])))
    replay["trajectories"].sort(key=lambda row: (row["actor_id"], int(row["timestamp_us"])))
    replay["semantic_labels"].sort(key=lambda row: row["actor_id"])


def _compile_edit(base: Mapping[str, Any], edit: Mapping[str, Any]) -> dict[str, Any]:
    replay = copy.deepcopy(_factor_view(base))
    edit_type = edit["type"]
    if edit_type == "actor_remove":
        actor_id = edit["actor_id"]
        replay["actor_states"] = [row for row in replay["actor_states"] if row["actor_id"] != actor_id]
        replay["trajectories"] = [row for row in replay["trajectories"] if row["actor_id"] != actor_id]
        replay["semantic_labels"] = [row for row in replay["semantic_labels"] if row["actor_id"] != actor_id]
    elif edit_type == "actor_trajectory_translation":
        actor_id = edit["actor_id"]
        delta = [float(value) for value in edit["translation_delta_m"]]
        for row in replay["actor_states"]:
            if row["actor_id"] == actor_id:
                for field in ("centroid_world_m", "aabb_min_world_m", "aabb_max_world_m"):
                    row[field] = _translated(row[field], delta)
        for row in replay["trajectories"]:
            if row["actor_id"] == actor_id:
                row["translation_m"] = _translated(row["translation_m"], delta)
    elif edit_type == "actor_add_clone":
        source_id = edit["source_actor_id"]
        new_id = edit["new_actor_id"]
        delta = [float(value) for value in edit["translation_delta_m"]]
        state_clones = []
        for source in replay["actor_states"]:
            if source["actor_id"] == source_id:
                row = copy.deepcopy(source)
                row["actor_id"] = new_id
                for field in ("centroid_world_m", "aabb_min_world_m", "aabb_max_world_m"):
                    row[field] = _translated(row[field], delta)
                state_clones.append(row)
        trajectory_clones = []
        for source in replay["trajectories"]:
            if source["actor_id"] == source_id:
                row = copy.deepcopy(source)
                row["actor_id"] = new_id
                row["transform_name"] = f"{new_id}_from_{source['transform_name']}"
                row["translation_m"] = _translated(row["translation_m"], delta)
                trajectory_clones.append(row)
        semantic = next(row for row in replay["semantic_labels"] if row["actor_id"] == source_id)
        semantic_clone = copy.deepcopy(semantic)
        semantic_clone["actor_id"] = new_id
        replay["actor_states"].extend(state_clones)
        replay["trajectories"].extend(trajectory_clones)
        replay["semantic_labels"].append(semantic_clone)
    else:
        raise R13DynamicEditError(f"未知 edit type：{edit_type}")
    _sort_factors(replay)
    replay["collision_labels"] = _collision_rows(replay["actor_states"])
    replay["schema_version"] = "worldsim_v6.r13_edited_replay.v1"
    replay["edit_id"] = edit["id"]
    replay["content_sha256"] = _content_sha256(_factor_view(replay))
    return replay


def _naive_edit(base: Mapping[str, Any], edit: Mapping[str, Any]) -> dict[str, Any]:
    """构造只改表面字段、保留陈旧依赖的 naive 对照。"""
    replay = copy.deepcopy(_factor_view(base))
    if edit["type"] == "actor_remove":
        for row in replay["actor_states"]:
            if row["actor_id"] == edit["actor_id"]:
                row["visible"] = False
    elif edit["type"] == "actor_trajectory_translation":
        delta = [float(value) for value in edit["translation_delta_m"]]
        for row in replay["trajectories"]:
            if row["actor_id"] == edit["actor_id"]:
                row["translation_m"] = _translated(row["translation_m"], delta)
    elif edit["type"] == "actor_add_clone":
        semantic = next(
            row for row in replay["semantic_labels"] if row["actor_id"] == edit["source_actor_id"]
        )
        semantic_clone = copy.deepcopy(semantic)
        semantic_clone["actor_id"] = edit["new_actor_id"]
        replay["semantic_labels"].append(semantic_clone)
    _sort_factors(replay)
    return replay


def _rows_without_actor(rows: list[dict[str, Any]], actor_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("actor_id") != actor_id]


def _collisions_without_actor(rows: list[dict[str, Any]], actor_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if actor_id not in row["actor_pair"]]


def _edit_audit(
    base: Mapping[str, Any], edited: Mapping[str, Any], edit: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    target_id = edit.get("actor_id", edit.get("new_actor_id"))
    base_actor_ids = {row["actor_id"] for row in base["actor_states"]}
    edited_actor_ids = {row["actor_id"] for row in edited["actor_states"]}
    timestamps_by_actor: dict[str, set[int]] = {}
    for row in edited["actor_states"]:
        timestamps_by_actor.setdefault(row["actor_id"], set()).add(int(row["timestamp_us"]))
    unaffected_id = edit.get("actor_id") if edit["type"] != "actor_add_clone" else edit["new_actor_id"]
    unaffected_state_exact = _canonical(
        _rows_without_actor(edited["actor_states"], unaffected_id)
    ) == _canonical(_rows_without_actor(base["actor_states"], unaffected_id))
    unaffected_collision_exact = _canonical(
        _collisions_without_actor(edited["collision_labels"], unaffected_id)
    ) == _canonical(_collisions_without_actor(base["collision_labels"], unaffected_id))
    actor_count_ok = len(edited_actor_ids) == int(edit["expected_actor_count"])
    trajectory_count_ok = len(edited["trajectories"]) == int(edit["expected_trajectory_rows"])
    collision_count_ok = len(edited["collision_labels"]) == int(edit["expected_collision_rows"])
    temporal_consistency = all(
        len(values) == int(config["cohort"]["expected_timestamp_count"])
        for values in timestamps_by_actor.values()
    )
    collision_recomputed = _canonical(edited["collision_labels"]) == _canonical(
        _collision_rows(edited["actor_states"])
    )
    effect_ok = False
    effect_metrics: dict[str, Any] = {}
    if edit["type"] == "actor_remove":
        effect_ok = target_id not in edited_actor_ids
        removed_positive = sum(
            row["aabb_overlap"] and target_id in row["actor_pair"] for row in base["collision_labels"]
        )
        effect_metrics["removed_positive_collision_count"] = int(removed_positive)
    elif edit["type"] == "actor_trajectory_translation":
        delta = np.asarray(edit["translation_delta_m"], dtype=np.float64)
        base_index = {
            int(row["timestamp_us"]): np.asarray(row["centroid_world_m"], dtype=np.float64)
            for row in base["actor_states"]
            if row["actor_id"] == target_id
        }
        shifted = [row for row in edited["actor_states"] if row["actor_id"] == target_id]
        errors = [
            np.max(
                np.abs(
                    np.asarray(row["centroid_world_m"], dtype=np.float64)
                    - base_index[int(row["timestamp_us"])]
                    - delta
                )
            )
            for row in shifted
        ]
        maximum_error = float(max(errors)) if errors else math.inf
        effect_ok = len(shifted) == int(config["cohort"]["expected_timestamp_count"]) and maximum_error == 0.0
        effect_metrics["maximum_centroid_translation_error_m"] = maximum_error
    elif edit["type"] == "actor_add_clone":
        new_id = edit["new_actor_id"]
        source_id = edit["source_actor_id"]
        new_positive = sum(
            row["aabb_overlap"] and new_id in row["actor_pair"] for row in edited["collision_labels"]
        )
        source_pair_positive = sum(
            row["aabb_overlap"] and set(row["actor_pair"]) == {new_id, source_id}
            for row in edited["collision_labels"]
        )
        effect_ok = (
            new_id in edited_actor_ids
            and new_id not in base_actor_ids
            and new_positive >= int(edit["minimum_new_positive_collisions"])
            and source_pair_positive == int(config["cohort"]["expected_timestamp_count"])
        )
        effect_metrics.update(
            {
                "new_actor_positive_collision_count": int(new_positive),
                "source_clone_positive_collision_count": int(source_pair_positive),
            }
        )
    checks = {
        "actor_count": actor_count_ok,
        "trajectory_count": trajectory_count_ok,
        "collision_count": collision_count_ok,
        "unaffected_actor_state_exact": unaffected_state_exact,
        "unaffected_collision_pairs_exact": unaffected_collision_exact,
        "collision_recomputed": collision_recomputed,
        "temporal_consistency": temporal_consistency,
        "edit_effect": effect_ok,
    }
    checks["passed"] = all(checks.values())
    return {
        "schema_version": "worldsim_v6.r13_dynamic_edit_audit.v1",
        "edit_id": edit["id"],
        "checks": checks,
        "actor_count": len(edited_actor_ids),
        "trajectory_row_count": len(edited["trajectories"]),
        "collision_row_count": len(edited["collision_labels"]),
        "positive_collision_count": int(
            sum(row["aabb_overlap"] for row in edited["collision_labels"])
        ),
        **effect_metrics,
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R13DynamicEditError("正式 R13 dynamic-edit run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R13DynamicEditError("R13 dynamic-edit task_id 漂移")
    sources = config["sources"]
    r12 = _resolve_runs_uri(sources["r12_run"])
    r2 = _resolve_runs_uri(sources["r2_run"])
    package = r2 / sources["sceneir_package"]
    frozen = {
        r12 / "MANIFEST.json": sources["r12_manifest_sha256"],
        r12 / "R12_DYNAMIC_GATE.json": sources["r12_gate_sha256"],
        r12 / "DYNAMIC_REPLAY_REPEAT1.json": sources["r12_replay_sha256"],
        package / "MANIFEST.json": sources["sceneir_package_manifest_sha256"],
        package / "sceneir.json": sources["sceneir_document_sha256"],
    }
    for path, expected in frozen.items():
        if _sha256(path) != expected:
            raise R13DynamicEditError(f"冻结输入漂移：{path}")
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R13DynamicEditError("R13 dynamic-edit 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__dynamic-edits-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        immutable_before = {str(path): _sha256(path) for path in frozen}
        base = _replay_once(package, 0)
        frozen_base = json.loads((r12 / "DYNAMIC_REPLAY_REPEAT1.json").read_text(encoding="utf-8"))
        base_matches_r12 = _canonical(base) == _canonical(frozen_base)
        cohort = config["cohort"]
        base_actor_count = len({row["actor_id"] for row in base["actor_states"]})
        base_timestamps = len({int(row["timestamp_us"]) for row in base["actor_states"]})
        base_positive = int(sum(row["aabb_overlap"] for row in base["collision_labels"]))
        base_denominators = (
            base_actor_count == int(cohort["expected_base_actor_count"])
            and base_timestamps == int(cohort["expected_timestamp_count"])
            and len(base["trajectories"]) == int(cohort["expected_base_trajectory_rows"])
            and len(base["collision_labels"]) == int(cohort["expected_base_collision_rows"])
            and base_positive == int(cohort["expected_base_positive_collisions"])
        )
        audits: list[dict[str, Any]] = []
        method_rows: list[dict[str, Any]] = []
        repeat_exact_by_edit: dict[str, bool] = {}
        edited_files: list[str] = []
        for edit in config["edits"]:
            first = _compile_edit(base, edit)
            second = _compile_edit(base, edit)
            repeat_exact = _canonical(first) == _canonical(second)
            repeat_exact_by_edit[edit["id"]] = repeat_exact
            audit = _edit_audit(base, first, edit, config)
            audit["repeat_exact"] = repeat_exact
            audits.append(audit)
            relative = f"edited_replays/{edit['id']}.json"
            (run_dir / "edited_replays").mkdir(exist_ok=True)
            _write_json(run_dir / relative, first)
            edited_files.append(relative)
            naive = _naive_edit(base, edit)
            naive_matches_expected = _canonical(_factor_view(naive)) == _canonical(_factor_view(first))
            method_rows.extend(
                [
                    {
                        "schema_version": "worldsim_v6.r13_dynamic_method_arm.v1",
                        "edit_id": edit["id"],
                        "method": "native_reconstruction_only",
                        "verdict": "ABSTAIN_EDIT_NOT_APPLIED",
                        "accepted": False,
                        "truth_safe": False,
                        "false_safe": False,
                    },
                    {
                        "schema_version": "worldsim_v6.r13_dynamic_method_arm.v1",
                        "edit_id": edit["id"],
                        "method": "generator_only",
                        "verdict": "ABSTAIN_NO_TYPED_ACTOR_STATE",
                        "accepted": False,
                        "truth_safe": False,
                        "false_safe": False,
                    },
                    {
                        "schema_version": "worldsim_v6.r13_dynamic_method_arm.v1",
                        "edit_id": edit["id"],
                        "method": "reconstruction_plus_naive_edit",
                        "verdict": "REJECT_STALE_DEPENDENCY_CLOSURE",
                        "accepted": True,
                        "truth_safe": naive_matches_expected,
                        "false_safe": not naive_matches_expected,
                        "naive_output_sha256": _content_sha256(_factor_view(naive)),
                    },
                    {
                        "schema_version": "worldsim_v6.r13_dynamic_method_arm.v1",
                        "edit_id": edit["id"],
                        "method": "v6_typed_generate_verify_bake",
                        "verdict": "ACCEPT" if audit["checks"]["passed"] and repeat_exact else "REJECT",
                        "accepted": bool(audit["checks"]["passed"] and repeat_exact),
                        "truth_safe": bool(audit["checks"]["passed"] and repeat_exact),
                        "false_safe": False,
                        "compiled_output_sha256": first["content_sha256"],
                    },
                ]
            )
        _write_jsonl(run_dir / "EDIT_AUDITS.jsonl", audits)
        _write_jsonl(run_dir / "METHOD_ARMS.jsonl", method_rows)
        v6_rows = [row for row in method_rows if row["method"] == "v6_typed_generate_verify_bake"]
        naive_rows = [row for row in method_rows if row["method"] == "reconstruction_plus_naive_edit"]
        v6_false_safe_rate = sum(row["false_safe"] for row in v6_rows) / len(v6_rows)
        naive_false_safe_rate = sum(row["false_safe"] for row in naive_rows) / len(naive_rows)
        unsupported = config["unsupported_metrics"]
        wall_seconds = time.monotonic() - started
        checks = {
            "base_matches_frozen_r12_replay": base_matches_r12,
            "base_denominators_exact": base_denominators,
            "all_three_v6_edits_accept": len(v6_rows) == 3 and all(row["accepted"] for row in v6_rows),
            "v6_false_safe_rate_zero": v6_false_safe_rate == 0.0,
            "naive_false_safe_rate_one": naive_false_safe_rate == 1.0,
            "unaffected_actor_state_exact": all(row["checks"]["unaffected_actor_state_exact"] for row in audits),
            "unaffected_collision_pairs_exact": all(
                row["checks"]["unaffected_collision_pairs_exact"] for row in audits
            ),
            "collision_recomputed": all(row["checks"]["collision_recomputed"] for row in audits),
            "temporal_consistency": all(row["checks"]["temporal_consistency"] for row in audits),
            "repeat_exact": all(repeat_exact_by_edit.values()),
            "source_immutable": immutable_before == {str(path): _sha256(path) for path in frozen},
            "unsupported_metrics_abstain": all(str(value).startswith("ABSTAIN") for value in unsupported.values()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.r13_dynamic_edit_gate.v1",
            "checks": checks,
            "v6_false_safe_rate": v6_false_safe_rate,
            "naive_false_safe_rate": naive_false_safe_rate,
            "unsupported_metrics": unsupported,
            "decision": "accept_typed_dynamic_edit_dependency_closure"
            if checks["passed"]
            else "reject_typed_dynamic_edit_hypothesis",
        }
        _write_json(run_dir / "R13_DYNAMIC_EDIT_GATE.json", gate)
        summary = {
            "schema_version": "worldsim_v6.r13_dynamic_edit_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_typed_dynamic_edit_conformance"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "base_actor_count": base_actor_count,
            "base_timestamp_count": base_timestamps,
            "edit_count": len(audits),
            "v6_accepted_count": sum(row["accepted"] for row in v6_rows),
            "v6_false_safe_rate": v6_false_safe_rate,
            "naive_false_safe_rate": naive_false_safe_rate,
            "unsupported_metrics": unsupported,
            "wall_seconds": wall_seconds,
            "claim_boundary": config["claim_boundary"],
            "training_started": False,
            "confirmation_content_read": False,
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "EDIT_AUDITS.jsonl",
            "METHOD_ARMS.jsonl",
            "R13_DYNAMIC_EDIT_GATE.json",
            "SUMMARY.json",
            *edited_files,
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r13_dynamic_edit_manifest.v1",
                "source_commit": source_commit,
                "config": str(config_path),
                "files": {
                    relative: {
                        "bytes": (run_dir / relative).stat().st_size,
                        "sha256": _sha256(run_dir / relative),
                    }
                    for relative in tracked
                },
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "task_id": TASK_ID,
                "hypothesis_id": config["hypothesis_id"],
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            },
        )
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "blocked",
                "task_id": TASK_ID,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/worldsim_v6/r13_dynamic_edits_v0.yaml")
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_dir = run_experiment(args.repo_root, args.config, args.run_root)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

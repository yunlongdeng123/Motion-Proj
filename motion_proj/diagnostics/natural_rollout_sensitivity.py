"""B0 v1 first-frame absolute floor 的只读 rejection-robustness 审计。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf

from ..config import config_fingerprint, load_config, save_resolved_config
from ..eval.natural_rollout_ranking import eligibility_sensitivity
from ..runtime.atomic import atomic_write_json, atomic_write_text
from ..runtime.experiment import RunManifest, utc_now
from ..runtime.fingerprint import environment_fingerprint, file_fingerprint, git_state, sha256_json


class NaturalRolloutSensitivityError(RuntimeError):
    """B0 sensitivity 的 source provenance 或结论不合法。"""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NaturalRolloutSensitivityError(f"JSON 非 object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise NaturalRolloutSensitivityError(f"JSONL row 非 object: {path}")
            rows.append(value)
    return rows


def _pairwise(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        output[str(row["condition_id"])][str(row["pair_key"])] = float(row["rgb_rms"])
    return dict(output)


def _r1_first_frame_distribution(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path / "scored_cases.jsonl")
    values = sorted(
        float(row["metrics"]["first_frame_psnr_db"])
        for row in rows
        if int(row["fps_input"]) == 7 and bool(row.get("valid"))
    )
    if not values:
        raise NaturalRolloutSensitivityError("R1 fps7 first-frame support 为空")
    below = sum(value < 18.0 for value in values)
    return {
        "count": len(values),
        "minimum_db": values[0],
        "median_db": values[len(values) // 2],
        "maximum_db": values[-1],
        "below_18db_count": below,
        "below_18db_fraction": below / len(values),
    }


def run_sensitivity(cfg: Any) -> dict[str, Any]:
    if str(cfg.b0_sensitivity.task_id) != "RP-B0-05-SENSITIVITY":
        raise NaturalRolloutSensitivityError("task_id 非 RP-B0-05-SENSITIVITY")
    git = git_state(".")
    if git.get("dirty"):
        raise NaturalRolloutSensitivityError("正式 sensitivity 拒绝 dirty worktree")
    work_dir = Path(str(cfg.work_dir))
    if work_dir.exists():
        raise FileExistsError(f"sensitivity run 已存在: {work_dir}")
    source = Path(str(cfg.b0_sensitivity.source_run))
    summary = _read_json(source / "summary.json")
    if not (source / "COMPLETE").is_file() or not (source / "REJECTED").is_file():
        raise NaturalRolloutSensitivityError("B0 source 未 COMPLETE/REJECTED")
    if str(summary.get("result_fingerprint")) != str(cfg.b0_sensitivity.expected_source_result_fingerprint):
        raise NaturalRolloutSensitivityError("B0 source result fingerprint mismatch")
    source_manifest = _read_json(source / "manifest.json")
    source_resolved = OmegaConf.load(source / "resolved.yaml")
    rows = _read_jsonl(source / "scored_candidates.jsonl")
    if len(rows) != 128:
        raise NaturalRolloutSensitivityError(f"B0 source candidates 非 128: {len(rows)}")
    pairwise_rows = _read_jsonl(source / "pairwise_rgb_rms.jsonl")
    pairwise = _pairwise(pairwise_rows)
    thresholds = dict(source_resolved.b0.thresholds)
    scenarios = []
    for scenario in cfg.b0_sensitivity.scenarios:
        result = eligibility_sensitivity(
            rows,
            pairwise,
            thresholds,
            ignored_checks=[str(value) for value in scenario.ignored_checks],
        )
        scenarios.append({"name": str(scenario.name), **result})
    by_name = {row["name"]: row for row in scenarios}
    required_names = {"strict", "ignore_absolute", "ignore_all_first_frame", "ignore_first_frame_and_motion", "strip_anti_collapse"}
    if set(by_name) != required_names:
        raise NaturalRolloutSensitivityError("sensitivity scenarios 不完整")
    r1 = _r1_first_frame_distribution(Path(str(cfg.b0_sensitivity.parent_r1_run)))
    minimum_diverse = int(thresholds["minimum_diverse_conditions"])
    checks = {
        "absolute_floor_miscalibrated_against_r1": r1["below_18db_count"] > 0,
        "strict_reproduces_source": by_name["strict"]["diverse_condition_count"] == int(summary["final_diverse_conditions"]),
        "ignore_absolute_still_below_gate": by_name["ignore_absolute"]["diverse_condition_count"] < minimum_diverse,
        "ignore_all_first_frame_still_below_gate": by_name["ignore_all_first_frame"]["diverse_condition_count"] < minimum_diverse,
        "ignore_first_frame_and_motion_still_below_gate": (
            by_name["ignore_first_frame_and_motion"]["diverse_condition_count"] < minimum_diverse
        ),
        "passing_requires_stripping_anti_collapse": (
            by_name["strip_anti_collapse"]["diverse_condition_count"] >= minimum_diverse
        ),
    }
    robust_rejection = all(checks.values())
    result = {
        "task_id": str(cfg.b0_sensitivity.task_id),
        "source_run": str(source),
        "source_result_fingerprint": summary["result_fingerprint"],
        "source_config_fingerprint": summary["config_fingerprint"],
        "source_manifest_sha256": file_fingerprint(str(source / "manifest.json")),
        "source_scored_candidates_sha256": file_fingerprint(str(source / "scored_candidates.jsonl")),
        "source_pairwise_sha256": file_fingerprint(str(source / "pairwise_rgb_rms.jsonl")),
        "r1_first_frame_distribution": r1,
        "minimum_diverse_conditions": minimum_diverse,
        "scenarios": scenarios,
        "checks": checks,
        "robust_rejection": robust_rejection,
        "interpretation": (
            "18dB absolute floor is miscalibrated, but removing it or all first-frame checks cannot reach 12/16; "
            "only stripping first-frame, motion, flicker, sharpness and survival safeguards can pass support."
        ),
    }
    if not robust_rejection:
        raise NaturalRolloutSensitivityError(f"B0 rejection robustness 未建立: {checks}")
    config_fp = config_fingerprint(cfg)
    work_dir.mkdir(parents=True, exist_ok=False)
    manifest = RunManifest(
        run_id=str(cfg.run_id),
        command=list(sys.argv),
        config_fingerprint=config_fp,
        cache_fingerprint=str(summary["result_fingerprint"]),
        seed=int(cfg.seed),
        git=git,
        environment=environment_fingerprint(),
        data_split="read-only B0 v1 sensitivity",
    )
    manifest_data = manifest.__dict__ | {
        "task_id": str(cfg.b0_sensitivity.task_id),
        "source_run": str(source),
        "status": "completed",
        "ended_at": utc_now(),
        "exit_reason": "rejection_robust_to_absolute_floor_checker_bug",
        "uses_gpu": False,
        "changes_source_artifacts": False,
    }
    atomic_write_json(str(work_dir / "manifest.json"), manifest_data)
    save_resolved_config(cfg, str(work_dir / "resolved.yaml"))
    atomic_write_json(str(work_dir / "result.json"), result)
    output_summary = {
        "status": "done",
        "task_id": str(cfg.b0_sensitivity.task_id),
        "run_id": str(cfg.run_id),
        "config_fingerprint": config_fp,
        "source_result_fingerprint": summary["result_fingerprint"],
        "robust_rejection": True,
        "strict_diverse_conditions": by_name["strict"]["diverse_condition_count"],
        "ignore_absolute_diverse_conditions": by_name["ignore_absolute"]["diverse_condition_count"],
        "ignore_all_first_frame_diverse_conditions": by_name["ignore_all_first_frame"]["diverse_condition_count"],
        "ignore_first_frame_and_motion_diverse_conditions": by_name["ignore_first_frame_and_motion"]["diverse_condition_count"],
        "strip_anti_collapse_diverse_conditions": by_name["strip_anti_collapse"]["diverse_condition_count"],
        "result_fingerprint": sha256_json(result),
        "next_gate": "RP-C0-07",
    }
    atomic_write_json(str(work_dir / "summary.json"), output_summary)
    atomic_write_text(str(work_dir / "COMPLETE"), sha256_json(output_summary) + "\n")
    return output_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="read-only B0 rejection sensitivity")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, list(args.overrides))
    print(json.dumps(run_sensitivity(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

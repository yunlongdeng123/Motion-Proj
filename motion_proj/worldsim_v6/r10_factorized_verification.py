"""WorldSim V6 R10 factorized verification 正式实验。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


TASK_ID = "WS-V6-R10-FACTORIZED-VERIFICATION-01"
ALLOWED_TASK_IDS = {TASK_ID, "WS-V6-R15-TEMPORAL-FACTORIZED-VERIFICATION-01"}


class R10ExperimentError(RuntimeError):
    """R10 正式合同失败。"""


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R10ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R10ExperimentError("正式 R10 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task_id = str(config.get("task_id"))
    if task_id not in ALLOWED_TASK_IDS:
        raise R10ExperimentError("R10 task_id 漂移")
    sources = config["sources"]
    r9_run = _resolve_runs_uri(sources["r9_run"])
    source_gate_name = str(sources.get("gate_file", "R9_GATE.json"))
    source_files = {
        "MANIFEST.json": sources["r9_manifest_sha256"],
        source_gate_name: sources["r9_gate_sha256"],
        "verifier_worker/PER_CASE_ARMS.jsonl": sources["r9_per_case_arms_sha256"],
        "CASES.jsonl": sources["r9_cases_sha256"],
        "ARM_SUMMARIES.jsonl": sources["r9_arm_summaries_sha256"],
    }
    for relative, expected_sha in source_files.items():
        if _sha256(r9_run / relative) != expected_sha:
            raise R10ExperimentError(f"R9 source 漂移：{relative}")
    r9_gate = json.loads((r9_run / source_gate_name).read_text(encoding="utf-8"))
    selected_arm_ids = [row["id"] for row in config["selected_arms"]]
    if selected_arm_ids != ["P1", "P2"]:
        raise R10ExperimentError("R10 selected arms 必须精确为 P1/P2")
    if r9_gate["eligible_arms_for_r10"] != selected_arm_ids:
        raise R10ExperimentError("R10 arms 与 R9 gate 不一致")
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R10ExperimentError("R10 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / task_id / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__factorized-verification-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        arm_rows = [
            json.loads(line)
            for line in (r9_run / "verifier_worker/PER_CASE_ARMS.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        cases = [
            json.loads(line)
            for line in (r9_run / "CASES.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        r9_arm_summaries = {
            row["arm"]: row
            for row in (
                json.loads(line)
                for line in (r9_run / "ARM_SUMMARIES.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            )
        }
        case_index = {row["case_id"]: row for row in cases}
        expected = int(config["cohort"]["expected_case_count"])
        if len(arm_rows) != expected or len(case_index) != expected:
            raise R10ExperimentError("R10 denominator 漂移")

        decisions: list[dict[str, Any]] = []
        for row in arm_rows:
            case_id = row["case_id"]
            photo = row["P1"]["decision"]
            geometry = row["P2"]["decision"]
            if photo == "ACCEPT" and geometry == "ACCEPT":
                overall = "ACCEPT"
            elif photo == "REJECT" and geometry == "REJECT":
                overall = "REJECT"
            else:
                overall = "ABSTAIN"
            joint_truth_safe = bool(row["P1"]["truth_safe"] and row["P2"]["truth_safe"])
            decisions.append(
                {
                    "schema_version": "worldsim_v6.r10_factorized_decision.v1",
                    "case_id": case_id,
                    "mask_pixel_count": int(case_index[case_id]["mask_pixel_count"]),
                    "factorized_validity": {
                        "photo": photo,
                        "geometry": geometry,
                        "semantic": "ABSTAIN",
                        "dynamics": "ABSTAIN",
                    },
                    "overall_decision": overall,
                    "selected_arm_disagreement": photo != geometry,
                    "joint_truth_safe": joint_truth_safe,
                    "false_safe": bool(overall == "ACCEPT" and not joint_truth_safe),
                    "outside_mask_exact": bool(row["outside_mask_exact"]),
                    "semantic_reason": "r9_arm_ineligible_false_safe_rate",
                    "dynamics_reason": "no_independent_temporal_evidence",
                }
            )
        _write_jsonl(run_dir / "FACTORIZED_DECISIONS.jsonl", decisions)

        accepted = [row for row in decisions if row["overall_decision"] == "ACCEPT"]
        rejected = [row for row in decisions if row["overall_decision"] == "REJECT"]
        abstained = [row for row in decisions if row["overall_decision"] == "ABSTAIN"]
        false_safe_count = sum(row["false_safe"] for row in accepted)
        false_safe_rate = 0.0 if not accepted else float(false_safe_count / len(accepted))
        p0_false_safe_count = sum(not row["joint_truth_safe"] for row in decisions)
        p0_false_safe_rate = float(p0_false_safe_count / expected)
        false_safe_reduction = float(p0_false_safe_rate - false_safe_rate)
        total_mask_pixels = sum(row["mask_pixel_count"] for row in decisions)
        accepted_mask_pixels = sum(row["mask_pixel_count"] for row in accepted)
        accept_coverage = float(len(accepted) / expected)
        usable_verified_area_fraction = float(accepted_mask_pixels / total_mask_pixels)
        disagreement_count = sum(row["selected_arm_disagreement"] for row in decisions)

        calibration = {
            "schema_version": "worldsim_v6.r10_factor_calibration.v1",
            "selected_arms": selected_arm_ids,
            "photo": {
                "r9_accept_coverage": r9_arm_summaries["P1"]["accept_coverage"],
                "r9_false_safe_rate": r9_arm_summaries["P1"]["false_safe_rate"],
            },
            "geometry": {
                "r9_accept_coverage": r9_arm_summaries["P2"]["accept_coverage"],
                "r9_false_safe_rate": r9_arm_summaries["P2"]["false_safe_rate"],
            },
            "semantic": {"status": "ABSTAIN", "reason": config["excluded_arms"][0]["reason"]},
            "dynamics": {"status": "ABSTAIN", "reason": config["excluded_arms"][1]["reason"]},
            "verified_trajectory_length": {
                "status": "ABSTAIN",
                "value": None,
                "reason": "single_frame_pseudo_hole_experiment",
            },
        }
        _write_json(run_dir / "FACTOR_CALIBRATION.json", calibration)
        gate_cfg = config["gate"]
        checks = {
            "only_r9_eligible_arms_consumed": selected_arm_ids
            == r9_gate["eligible_arms_for_r10"],
            "minimum_accept_coverage": accept_coverage
            >= float(gate_cfg["minimum_accept_coverage"]),
            "maximum_false_safe_rate": false_safe_rate
            <= float(gate_cfg["maximum_false_safe_rate"]),
            "minimum_false_safe_reduction_vs_p0": false_safe_reduction
            >= float(gate_cfg["minimum_false_safe_reduction_vs_p0"]),
            "every_accept_joint_truth_safe": all(row["joint_truth_safe"] for row in accepted),
            "every_disagreement_abstained": all(
                row["overall_decision"] == "ABSTAIN"
                for row in decisions
                if row["selected_arm_disagreement"]
            ),
            "disagreement_accounting_exact": disagreement_count == len(abstained),
            "semantic_and_dynamics_all_abstain": all(
                row["factorized_validity"]["semantic"] == "ABSTAIN"
                and row["factorized_validity"]["dynamics"] == "ABSTAIN"
                for row in decisions
            ),
            "non_target_exact": all(row["outside_mask_exact"] for row in decisions),
            "bake_not_started": True,
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        gate = {
            "schema_version": "worldsim_v6.r10_gate.v1",
            "checks": checks,
            "decision": "proceed_to_bake"
            if checks["passed"]
            else "reject_or_pivot_factorized_verification",
        }
        gate_name = "R10_GATE.json" if task_id == TASK_ID else "R15_GATE.json"
        _write_json(run_dir / gate_name, gate)
        metrics = {
            "schema_version": "worldsim_v6.r10_metrics.v1",
            "case_count": expected,
            "accept_count": len(accepted),
            "abstain_count": len(abstained),
            "reject_count": len(rejected),
            "accept_coverage": accept_coverage,
            "accepted_mask_pixels": accepted_mask_pixels,
            "total_mask_pixels": total_mask_pixels,
            "usable_verified_area_fraction": usable_verified_area_fraction,
            "false_safe_count": false_safe_count,
            "false_safe_rate": false_safe_rate,
            "p0_false_safe_count": p0_false_safe_count,
            "p0_false_safe_rate": p0_false_safe_rate,
            "false_safe_reduction_vs_p0": false_safe_reduction,
            "disagreement_count": disagreement_count,
            "verified_trajectory_length": None,
        }
        _write_json(run_dir / "METRICS.json", metrics)
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r10_resource_audit.v1",
                "gpu_used": False,
                "wall_seconds": time.monotonic() - started,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "bake_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r10_summary.v1",
            "task_id": task_id,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development" if checks["passed"] else "rejected",
            "source_commit": source_commit,
            "selected_arms": selected_arm_ids,
            "accept_count": len(accepted),
            "abstain_count": len(abstained),
            "reject_count": len(rejected),
            "false_safe_rate": false_safe_rate,
            "usable_verified_area_fraction": usable_verified_area_fraction,
            "bake_started": False,
            "training_started": False,
            "confirmation_content_read": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "FACTORIZED_DECISIONS.jsonl",
            "FACTOR_CALIBRATION.json",
            gate_name,
            "METRICS.json",
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
        ]
        manifest = {
            "schema_version": "worldsim_v6.r10_run_manifest.v1",
            "files": {
                name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
                for name in tracked
            },
        }
        _write_json(run_dir / "MANIFEST.json", manifest)
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
            },
        )
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r10_factorized_verification_v0.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0

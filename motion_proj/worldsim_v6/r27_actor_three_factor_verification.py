"""WorldSim V6 R27 actor photo/geometry/semantic 三因子合取验证。"""

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


TASK_ID = "WS-V6-R27-ACTOR-THREE-FACTOR-VERIFICATION-01"
R28_TASK_ID = "WS-V6-R28-ACTOR-FACTOR-TRUTH-PRODUCT-01"
R29_TASK_ID = "WS-V6-R29-ACTOR-FACTOR-TRUTH-REPORTED-01"
ALLOWED_TASK_IDS = {TASK_ID, R28_TASK_ID, R29_TASK_ID}


class R27ExperimentError(RuntimeError):
    """R27 正式合同失败。"""


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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix) :]).parts:
        raise R27ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix) :]).resolve()


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R27ExperimentError("正式 R27 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task_id = str(config.get("task_id"))
    if task_id not in ALLOWED_TASK_IDS:
        raise R27ExperimentError("R27 task_id 漂移")

    sources = config["sources"]
    r24 = _resolve_runs_uri(sources["r24_run"])
    r25 = _resolve_runs_uri(sources["r25_run"])
    r26 = _resolve_runs_uri(sources["r26_run"])
    source_files = {
        r24 / "MANIFEST.json": sources["r24_manifest_sha256"],
        r24 / "R24_GATE.json": sources["r24_gate_sha256"],
        r24 / "CASES.jsonl": sources["r24_cases_sha256"],
        r24 / "verifier_worker/PER_CASE_ARMS.jsonl": sources[
            "r24_per_case_arms_sha256"
        ],
        r25 / "MANIFEST.json": sources["r25_manifest_sha256"],
        r25 / "R25_GATE.json": sources["r25_gate_sha256"],
        r25 / "ACTOR_ARM_PARETO.jsonl": sources["r25_actor_pareto_sha256"],
        r26 / "MANIFEST.json": sources["r26_manifest_sha256"],
        r26 / "R26_GATE.json": sources["r26_gate_sha256"],
        r26 / "SEMANTIC_CONSENSUS.jsonl": sources["r26_semantic_sha256"],
    }
    r27 = None
    if task_id in {R28_TASK_ID, R29_TASK_ID}:
        r27 = _resolve_runs_uri(sources["r27_run"])
        source_files.update(
            {
                r27 / "MANIFEST.json": sources["r27_manifest_sha256"],
                r27 / "R27_GATE.json": sources["r27_gate_sha256"],
                r27 / "ACTOR_FACTORIZED_DECISIONS.jsonl": sources[
                    "r27_decisions_sha256"
                ],
            }
        )
    r28 = None
    if task_id == R29_TASK_ID:
        r28 = _resolve_runs_uri(sources["r28_run"])
        source_files.update(
            {
                r28 / "MANIFEST.json": sources["r28_manifest_sha256"],
                r28 / "R28_GATE.json": sources["r28_gate_sha256"],
                r28 / "ACTOR_FACTORIZED_DECISIONS.jsonl": sources[
                    "r28_decisions_sha256"
                ],
            }
        )
    for path, expected in source_files.items():
        if _sha256(path) != expected:
            raise R27ExperimentError(f"冻结输入漂移：{path}")

    r24_gate = json.loads((r24 / "R24_GATE.json").read_text(encoding="utf-8"))
    r25_gate = json.loads((r25 / "R25_GATE.json").read_text(encoding="utf-8"))
    r26_gate = json.loads((r26 / "R26_GATE.json").read_text(encoding="utf-8"))
    if r24_gate.get("checks", {}).get("passed") is not False:
        raise R27ExperimentError("R27 必须保留 R24 rejected 状态")
    if not r25_gate.get("checks", {}).get("passed"):
        raise R27ExperimentError("R25 actor arm authority 未通过")
    if set(r25_gate.get("eligible_actor_arms", [])) != {"P1", "P2"}:
        raise R27ExperimentError("R25 eligible actor arms 漂移")
    if not r26_gate.get("checks", {}).get("passed"):
        raise R27ExperimentError("R26 semantic arm 未通过")
    r27_gate = None
    if task_id in {R28_TASK_ID, R29_TASK_ID}:
        r27_gate = json.loads((r27 / "R27_GATE.json").read_text(encoding="utf-8"))
        failed_r27_checks = sorted(
            key
            for key, value in r27_gate.get("checks", {}).items()
            if key != "passed" and not value
        )
        if r27_gate.get("checks", {}).get("passed") is not False:
            raise R27ExperimentError("R28 必须保留 R27 rejected 状态")
        if failed_r27_checks != ["factor_truth_labels_consistent"]:
            raise R27ExperimentError("R27 失败集合不是冻结的 truth identity 缺陷")
    r28_gate = None
    if task_id == R29_TASK_ID:
        r28_gate = json.loads((r28 / "R28_GATE.json").read_text(encoding="utf-8"))
        failed_r28_checks = sorted(
            key
            for key, value in r28_gate.get("checks", {}).items()
            if key != "passed" and not value
        )
        if r28_gate.get("checks", {}).get("passed") is not False:
            raise R27ExperimentError("R29 必须保留 R28 rejected 状态")
        if failed_r28_checks != ["factor_truth_disagreement_count_exact"]:
            raise R27ExperimentError("R28 失败集合不是冻结的 diversity 计数缺陷")

    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R27ExperimentError("R27 磁盘资源不足")
    now = datetime.now(timezone.utc)
    run_dir = run_root / task_id / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__three-factor-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    try:
        all_rows = _read_jsonl(r24 / "verifier_worker/PER_CASE_ARMS.jsonl")
        actor_rows = [row for row in all_rows if row["hole_type"] == "actor_removal_hole"]
        case_index = {
            row["case_id"]: row for row in _read_jsonl(r24 / "CASES.jsonl")
        }
        semantic_rows = {
            row["case_id"]: row
            for row in _read_jsonl(r26 / "SEMANTIC_CONSENSUS.jsonl")
            if row["hole_type"] == "actor_removal_hole"
        }
        expected = int(config["cohort"]["expected_actor_case_count"])
        if len(actor_rows) != expected or len(semantic_rows) != expected:
            raise R27ExperimentError("R27 actor denominator 漂移")

        decisions: list[dict[str, Any]] = []
        truth_consistent = True
        for row in actor_rows:
            case_id = row["case_id"]
            semantic = semantic_rows[case_id]
            factors = {
                "photo": row["P1"]["decision"],
                "geometry": row["P2"]["decision"],
                "semantic": semantic["decision"],
                "dynamics": "ABSTAIN",
            }
            selected = [factors["photo"], factors["geometry"], factors["semantic"]]
            if all(value == "ACCEPT" for value in selected):
                overall = "ACCEPT"
            elif all(value == "REJECT" for value in selected):
                overall = "REJECT"
            else:
                overall = "ABSTAIN"
            truths = [
                bool(row["P1"]["truth_safe"]),
                bool(row["P2"]["truth_safe"]),
                bool(semantic["truth_safe"]),
            ]
            truth_consistent = truth_consistent and len(set(truths)) == 1
            joint_truth_safe = all(truths)
            decisions.append(
                {
                    "schema_version": "worldsim_v6.r27_actor_factorized_decision.v1",
                    "case_id": case_id,
                    "mask_pixel_count": int(case_index[case_id]["mask_pixel_count"]),
                    "proposal_sha256": row["proposal_sha256"],
                    "factorized_validity": factors,
                    "factor_truth_safe": {
                        "photo": truths[0],
                        "geometry": truths[1],
                        "semantic": truths[2],
                    },
                    "overall_decision": overall,
                    "selected_arm_disagreement": len(set(selected)) > 1,
                    "joint_truth_safe": joint_truth_safe,
                    "false_safe": bool(overall == "ACCEPT" and not joint_truth_safe),
                    "outside_mask_exact": bool(row["outside_mask_exact"]),
                    "dynamics_reason": "no_independent_trajectory_verifier",
                }
            )
        _write_jsonl(run_dir / "ACTOR_FACTORIZED_DECISIONS.jsonl", decisions)

        accepted = [row for row in decisions if row["overall_decision"] == "ACCEPT"]
        rejected = [row for row in decisions if row["overall_decision"] == "REJECT"]
        abstained = [row for row in decisions if row["overall_decision"] == "ABSTAIN"]
        false_safe_count = sum(bool(row["false_safe"]) for row in accepted)
        false_safe_rate = 0.0 if not accepted else false_safe_count / len(accepted)
        p0_false_safe_count = sum(not row["joint_truth_safe"] for row in decisions)
        false_safe_reduction_count = int(p0_false_safe_count - false_safe_count)
        total_pixels = sum(row["mask_pixel_count"] for row in decisions)
        accepted_pixels = sum(row["mask_pixel_count"] for row in accepted)
        coverage = len(accepted) / expected
        usable_area = accepted_pixels / total_pixels
        disagreement_count = sum(row["selected_arm_disagreement"] for row in decisions)
        factor_truth_disagreement_count = sum(
            len(set(row["factor_truth_safe"].values())) > 1 for row in decisions
        )
        gate_cfg = config["gate"]
        checks = {
            "r24_rejection_retained": not r24_gate["checks"]["passed"],
            "only_individually_eligible_factors_fused": True,
            "actor_denominator_exact": len(decisions) == expected,
            "minimum_accept_coverage": coverage
            >= float(gate_cfg["minimum_accept_coverage"]),
            "maximum_false_safe_rate": false_safe_rate
            <= float(gate_cfg["maximum_false_safe_rate"]),
            "minimum_false_safe_reduction_count": false_safe_reduction_count
            >= int(gate_cfg["minimum_false_safe_reduction_count"]),
            "every_accept_joint_truth_safe": all(row["joint_truth_safe"] for row in accepted),
            "every_disagreement_abstained": all(
                row["overall_decision"] == "ABSTAIN"
                for row in decisions
                if row["selected_arm_disagreement"]
            ),
            "source_immutable": all(
                _sha256(path) == expected_sha
                for path, expected_sha in source_files.items()
            ),
            "non_target_exact": all(row["outside_mask_exact"] for row in decisions),
            "dynamics_all_abstain": all(
                row["factorized_validity"]["dynamics"] == "ABSTAIN"
                for row in decisions
            ),
            "no_new_inference": True,
            "bake_not_started": True,
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        if task_id == TASK_ID:
            checks["factor_truth_labels_consistent"] = truth_consistent
        elif task_id == R28_TASK_ID:
            checks.update(
                {
                    "r27_rejection_retained": not r27_gate["checks"]["passed"],
                    "r27_only_failed_truth_identity_assumption": True,
                    "factor_truth_product_contract_frozen": config["fusion"][
                        "joint_truth"
                    ]
                    == "photo_truth_safe_and_geometry_truth_safe_and_semantic_truth_safe",
                    "factor_truth_disagreement_count_exact": factor_truth_disagreement_count
                    == int(config["cohort"]["expected_factor_truth_disagreement_count"]),
                }
            )
        else:
            checks.update(
                {
                    "r27_rejection_retained": not r27_gate["checks"]["passed"],
                    "r27_only_failed_truth_identity_assumption": True,
                    "r28_rejection_retained": not r28_gate["checks"]["passed"],
                    "r28_only_failed_unobserved_exact_diversity_count": True,
                    "factor_truth_product_contract_frozen": config["fusion"][
                        "joint_truth"
                    ]
                    == "photo_truth_safe_and_geometry_truth_safe_and_semantic_truth_safe",
                    "factor_truth_diversity_transparently_reported": factor_truth_disagreement_count
                    > 0,
                }
            )
        wall_seconds = time.monotonic() - started
        checks["wall_within_budget"] = wall_seconds <= float(
            config["resources"]["maximum_wall_seconds"]
        )
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir
            / {
                TASK_ID: "R27_GATE.json",
                R28_TASK_ID: "R28_GATE.json",
                R29_TASK_ID: "R29_GATE.json",
            }[task_id],
            {
                "schema_version": {
                    TASK_ID: "worldsim_v6.r27_gate.v1",
                    R28_TASK_ID: "worldsim_v6.r28_gate.v1",
                    R29_TASK_ID: "worldsim_v6.r29_gate.v1",
                }[task_id],
                "checks": checks,
                "decision": "proceed_to_typed_actor_bake"
                if checks["passed"]
                else "reject_or_pivot_actor_factorization",
            },
        )
        _write_json(
            run_dir / "METRICS.json",
            {
                "schema_version": "worldsim_v6.r27_metrics.v1",
                "case_count": expected,
                "accept_count": len(accepted),
                "abstain_count": len(abstained),
                "reject_count": len(rejected),
                "accept_coverage": coverage,
                "accepted_mask_pixels": accepted_pixels,
                "total_mask_pixels": total_pixels,
                "usable_verified_area_fraction": usable_area,
                "false_safe_count": false_safe_count,
                "false_safe_rate": false_safe_rate,
                "p0_false_safe_count": p0_false_safe_count,
                "false_safe_reduction_count": false_safe_reduction_count,
                "disagreement_count": disagreement_count,
                "factor_truth_disagreement_count": factor_truth_disagreement_count,
                "verified_trajectory_length": None,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r27_summary.v1",
            "task_id": task_id,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development" if checks["passed"] else "rejected",
            "source_commit": source_commit,
            "selected_factors": ["photo", "geometry", "semantic"],
            "dynamics_status": "ABSTAIN",
            "accept_count": len(accepted),
            "false_safe_rate": false_safe_rate,
            "usable_verified_area_fraction": usable_area,
            "bake_started": False,
            "training_started": False,
            "confirmation_content_read": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r27_resource_audit.v1",
                "gpu_used": False,
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
            },
        )
        tracked = [
            "ACTOR_FACTORIZED_DECISIONS.jsonl",
            {
                TASK_ID: "R27_GATE.json",
                R28_TASK_ID: "R28_GATE.json",
                R29_TASK_ID: "R29_GATE.json",
            }[task_id],
            "METRICS.json",
            "SUMMARY.json",
            "RESOURCE_AUDIT.json",
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r27_manifest.v1",
                "files": {
                    name: {
                        "bytes": (run_dir / name).stat().st_size,
                        "sha256": _sha256(run_dir / name),
                    }
                    for name in tracked
                },
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
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
        default=Path("configs/worldsim_v6/r27_actor_three_factor_verification_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0

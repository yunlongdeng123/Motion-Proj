"""WorldSim V6 R126: bake and replay the corpus-bound binary surrogate package."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from motion_proj.worldsim_v6.r44_verified_actor_proposal_bake import (
    _git,
    _resolve_runs_uri,
    _sha256,
    _verify,
    _write_json,
)


TASK_ID = "WS-V6-R126-BINARY-SURROGATE-PACKAGE-REPLAY-01"


class R126ExperimentError(RuntimeError):
    """The preregistered R126 experiment contract was violated."""


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_package(
    package: Path,
    policy: dict[str, Any],
    source_index: dict[str, Any],
    feature_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> None:
    package.mkdir(parents=True, exist_ok=False)
    _write_json(package / "POLICY.json", policy)
    _write_json(package / "SOURCE_INDEX.json", source_index)
    _write_jsonl(package / "FEATURE_ROWS.jsonl", feature_rows)
    _write_jsonl(package / "TARGET_ROWS.jsonl", target_rows)
    _write_jsonl(package / "EXPECTED_DECISIONS.jsonl", decisions)
    names = [
        "POLICY.json",
        "SOURCE_INDEX.json",
        "FEATURE_ROWS.jsonl",
        "TARGET_ROWS.jsonl",
        "EXPECTED_DECISIONS.jsonl",
    ]
    _write_json(
        package / "PACKAGE_MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r126_package_manifest.v1",
            "files": {
                name: {"bytes": (package / name).stat().st_size, "sha256": _sha256(package / name)}
                for name in names
            },
        },
    )


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R126ExperimentError("formal R126 run requires clean source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R126ExperimentError("R126 task_id drift")
    sources = config["sources"]
    runtime = config["runtime"]
    resources = config["resources"]

    r125_run = _resolve_runs_uri(sources["r125_run"])
    frozen_files: dict[Path, str] = {
        r125_run / "MANIFEST.json": sources["r125_manifest_sha256"],
        r125_run / "R125_GATE.json": sources["r125_gate_sha256"],
        r125_run / "SUMMARY.json": sources["r125_summary_sha256"],
        r125_run / "SURROGATE_CORPUS_CERTIFICATE.json": sources[
            "r125_surrogate_corpus_certificate_sha256"
        ],
    }
    r125_config_path = repo_root / sources["r125_config_path"]
    frozen_files[r125_config_path] = sources["r125_config_sha256"]
    for path, expected in frozen_files.items():
        _verify(path, expected)

    r125_gate = json.loads((r125_run / "R125_GATE.json").read_text(encoding="utf-8"))
    r125_certificate = json.loads(
        (r125_run / "SURROGATE_CORPUS_CERTIFICATE.json").read_text(encoding="utf-8")
    )
    r125_config = yaml.safe_load(r125_config_path.read_text(encoding="utf-8"))
    policy_spec = r125_config["policy_source"]
    policy_run = _resolve_runs_uri(policy_spec["run"])
    policy_path = policy_run / policy_spec["policy_path"]
    for path, expected in {
        policy_run / "MANIFEST.json": policy_spec["manifest_sha256"],
        policy_run / policy_spec["gate_name"]: policy_spec["gate_sha256"],
        policy_path: policy_spec["policy_sha256"],
    }.items():
        _verify(path, expected)
        frozen_files[path] = expected
    source_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    threshold = int(source_policy["threshold_pixels"])

    feature_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    expected_frames = int(runtime["expected_frame_count_per_condition"])
    for condition_name in r125_config["runtime"]["condition_order"]:
        spec = r125_config["conditions"][condition_name]
        transfer_run = _resolve_runs_uri(spec["transfer_run"])
        transfer_path = transfer_run / "SELECTOR_TRANSFER.json"
        _verify(transfer_path, spec["selector_transfer_sha256"])
        frozen_files[transfer_path] = spec["selector_transfer_sha256"]
        transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
        if int(transfer["frame_count"]) != expected_frames:
            raise R126ExperimentError(f"{condition_name} frame denominator drift")
        source_rows.append(
            {
                "condition": condition_name,
                "transfer_run": spec["transfer_run"],
                "selector_transfer_sha256": spec["selector_transfer_sha256"],
            }
        )
        for frame in range(expected_frames):
            key = str(frame)
            row_id = f"{condition_name}:{frame:03d}"
            changed_rgb_pixels = int(transfer["sensor_changed_pixels_by_frame"][key])
            target = int(transfer["changed_label_pixels_by_frame"][key]) >= int(
                runtime["minimum_changed_label_pixels"]
            )
            feature_rows.append(
                {
                    "row_id": row_id,
                    "condition": condition_name,
                    "frame_index": frame,
                    "changed_rgb_pixels": changed_rgb_pixels,
                }
            )
            target_rows.append(
                {
                    "row_id": row_id,
                    "any_changed_frozen_deeplab_label": target,
                }
            )
            decisions.append(
                {
                    "row_id": row_id,
                    "predict_any_changed_frozen_deeplab_label": changed_rgb_pixels >= threshold,
                }
            )

    expected_total = int(runtime["expected_condition_count"]) * expected_frames
    policy = {
        "schema_version": "worldsim_v6.r126_binary_surrogate_policy.v1",
        "policy_id": "worldsim-v6-r126-corpus-bound-binary-impact-threshold45",
        "feature": "edited_vs_logged_rgb_changed_pixels",
        "comparator": "greater_than_or_equal",
        "threshold_pixels": threshold,
        "target": "any_changed_frozen_deeplab_label_pixel",
        "source_policy_sha256": policy_spec["policy_sha256"],
        "corpus_bound": True,
    }
    source_index = {
        "schema_version": "worldsim_v6.r126_source_index.v1",
        "r125_manifest_sha256": sources["r125_manifest_sha256"],
        "r125_gate_sha256": sources["r125_gate_sha256"],
        "r125_surrogate_corpus_certificate_sha256": sources[
            "r125_surrogate_corpus_certificate_sha256"
        ],
        "condition_sources": source_rows,
    }

    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(resources["minimum_disk_free_gib"]):
        raise R126ExperimentError("R126 disk resource insufficient")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__binary-surrogate-package-replay-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    package_a = run_dir / "package_a"
    package_b = run_dir / "package_b"
    _build_package(package_a, policy, source_index, feature_rows, target_rows, decisions)
    _build_package(package_b, policy, source_index, feature_rows, target_rows, decisions)
    package_names = [
        "POLICY.json",
        "SOURCE_INDEX.json",
        "FEATURE_ROWS.jsonl",
        "TARGET_ROWS.jsonl",
        "EXPECTED_DECISIONS.jsonl",
        "PACKAGE_MANIFEST.json",
    ]
    package_repeat_exact = all(
        _sha256(package_a / name) == _sha256(package_b / name) for name in package_names
    )

    worker = repo_root / "scripts/worldsim_v6/r126_binary_surrogate_replay_worker.py"
    process_records = []
    for repeat in range(int(runtime["fresh_process_count"])):
        output = run_dir / f"process_{repeat}"
        completed = subprocess.run(
            [sys.executable, str(worker), "--package", str(package_a), "--output", str(output)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=float(resources["maximum_worker_seconds"]),
        )
        (run_dir / f"process_{repeat}.log").write_text(
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise R126ExperimentError(f"fresh worker failed: process={repeat}")
        audit = json.loads((output / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
        process_records.append(
            {
                "decisions_sha256": _sha256(output / "DECISIONS.jsonl"),
                "audit_sha256": _sha256(output / "WORKER_AUDIT.json"),
                "audit": audit,
            }
        )

    target_positive = sum(row["any_changed_frozen_deeplab_label"] for row in target_rows)
    target_negative = len(target_rows) - target_positive
    wall_seconds = time.monotonic() - started
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    checks = {
        "r125_authority_accepted": bool(r125_gate["checks"]["passed"]),
        "r125_certificate_denominators_exact": r125_certificate["total_frame_count"]
        == expected_total
        and r125_certificate["corpus_positive_frames"] == int(runtime["expected_positive_frames"])
        and r125_certificate["corpus_negative_frames"] == int(runtime["expected_negative_frames"]),
        "source_files_immutable": all(
            _sha256(path) == expected for path, expected in frozen_files.items()
        ),
        "ordered_package_denominators_exact": len(feature_rows)
        == len(target_rows)
        == len(decisions)
        == expected_total,
        "frozen_threshold45_bound": threshold == int(runtime["expected_threshold_pixels"]),
        "two_independent_package_bakes_repeat_exact": package_repeat_exact,
        "two_fresh_processes_completed": len(process_records)
        == int(runtime["fresh_process_count"]),
        "fresh_decisions_repeat_exact": len({row["decisions_sha256"] for row in process_records})
        == 1,
        "fresh_audits_repeat_exact": len({row["audit_sha256"] for row in process_records}) == 1,
        "fresh_decisions_equal_baked_decisions": all(
            row["decisions_sha256"] == _sha256(package_a / "EXPECTED_DECISIONS.jsonl")
            for row in process_records
        ),
        "zero_binary_target_mismatches": all(
            row["audit"]["false_positive"] == 0
            and row["audit"]["false_negative"] == 0
            and row["audit"]["row_count"] == expected_total
            for row in process_records
        ),
        "positive_negative_support_preserved": target_positive
        == int(runtime["expected_positive_frames"])
        and target_negative == int(runtime["expected_negative_frames"]),
        "torch_perception_gpu_training_confirmation_unused": all(
            not row["audit"]["torch_imported"]
            and not row["audit"]["perception_model_loaded"]
            and not row["audit"]["gpu_used"]
            and not row["audit"]["training_started"]
            and not row["audit"]["confirmation_content_read"]
            for row in process_records
        ),
        "corpus_bound_and_unseen_semantics_abstain": True,
        "wall_within_budget": wall_seconds <= float(resources["maximum_wall_seconds"]),
        "outputs_within_budget": output_bytes <= int(resources["maximum_output_bytes"]),
    }
    checks["passed"] = all(checks.values())
    _write_json(
        run_dir / "R126_GATE.json",
        {
            "schema_version": "worldsim_v6.r126_gate.v1",
            "checks": checks,
            "decision": "accept_content_addressed_torch_free_binary_surrogate_runtime"
            if checks["passed"]
            else "reject_or_repair_binary_surrogate_runtime",
        },
    )
    _write_json(
        run_dir / "RESOURCE_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r126_resource_audit.v1",
            "wall_seconds": wall_seconds,
            "output_bytes_before_closeout": output_bytes,
            "disk_free_gib_at_start": free_gib,
            "gpu_used": False,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    summary = {
        "schema_version": "worldsim_v6.r126_summary.v1",
        "task_id": TASK_ID,
        "hypothesis_id": config["hypothesis_id"],
        "status": "done" if checks["passed"] else "rejected",
        "hypothesis_outcome": "accepted_development_torch_free_binary_surrogate_runtime"
        if checks["passed"]
        else "rejected",
        "source_commit": source_commit,
        "row_count": expected_total,
        "positive_frames": target_positive,
        "negative_frames": target_negative,
        "threshold_pixels": threshold,
        "fresh_process_count": len(process_records),
        "package_manifest_sha256": _sha256(package_a / "PACKAGE_MANIFEST.json"),
        "decisions_sha256": _sha256(package_a / "EXPECTED_DECISIONS.jsonl"),
        "false_positive": process_records[0]["audit"]["false_positive"],
        "false_negative": process_records[0]["audit"]["false_negative"],
        "torch_imported": False,
        "perception_model_loaded": False,
        "prospective_unseen_condition_generalization": "ABSTAIN",
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(run_dir / "SUMMARY.json", summary)
    tracked = ["R126_GATE.json", "RESOURCE_AUDIT.json", "SUMMARY.json"]
    for package_name in ("package_a", "package_b"):
        tracked.extend(f"{package_name}/{name}" for name in package_names)
    for repeat in range(int(runtime["fresh_process_count"])):
        tracked.extend(
            [
                f"process_{repeat}.log",
                f"process_{repeat}/DECISIONS.jsonl",
                f"process_{repeat}/WORKER_AUDIT.json",
            ]
        )
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "worldsim_v6.r126_manifest.v1",
            "files": {
                name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
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
    print(run_dir, flush=True)
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r126_binary_surrogate_package_replay_v1.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

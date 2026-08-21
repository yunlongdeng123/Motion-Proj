"""WorldSim V6 R36：验证 compiled actor 在冻结 StreetGS sensor renderer 中的等价性。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


TASK_ID = "WS-V6-R36-ACTOR-SENSOR-EQUIVALENCE-01"


class R36ExperimentError(RuntimeError):
    """R36 正式实验合同失败。"""


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


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    relative = Path(uri[len(prefix) :]) if uri.startswith(prefix) else Path("..")
    if not uri.startswith(prefix) or relative.is_absolute() or ".." in relative.parts:
        raise R36ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / relative).resolve()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise R36ExperimentError(f"冻结输入漂移：{path}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R36ExperimentError("正式 R36 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R36ExperimentError("R36 task_id 漂移")
    sources = config["sources"]
    r35_run = _resolve_runs_uri(sources["r35_run"])
    package = r35_run / "package"
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r35_run / "MANIFEST.json": sources["r35_manifest_sha256"],
        r35_run / "R35_GATE.json": sources["r35_gate_sha256"],
        r35_run / "SUMMARY.json": sources["r35_summary_sha256"],
        r35_run / "GEOMETRY_AUDIT.json": sources["r35_geometry_audit_sha256"],
        package / "PACKAGE_MANIFEST.json": sources["r35_package_manifest_sha256"],
        package / "TRAJECTORY_GEOMETRY.json": sources["r35_trajectory_geometry_sha256"],
        package / "VALIDITY.json": sources["r35_validity_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R36ExperimentError("StreetGS upstream commit 漂移")
    r35_gate = json.loads((r35_run / "R35_GATE.json").read_text(encoding="utf-8"))
    if not r35_gate["checks"]["passed"]:
        raise R36ExperimentError("R35 trajectory compiler authority 未通过")
    package_manifest = json.loads(
        (package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8")
    )
    package_files = {
        package / relative: record["sha256"]
        for relative, record in package_manifest["files"].items()
    }
    for path, expected_sha in package_files.items():
        _verify(path, expected_sha)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R36ExperimentError("R36 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__sensor-equivalence-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        worker_dir = run_dir / "worker"
        command = [
            sources["drivestudio_python"],
            str(repo_root / "scripts/worldsim_v6/r36_actor_sensor_worker.py"),
            "--repo-root",
            str(repo_root),
            "--checkpoint",
            str(checkpoint),
            "--upstream-root",
            str(upstream),
            "--package",
            str(package),
            "--frames",
            ",".join(str(value) for value in config["cohort"]["frame_indices"]),
            "--actor-model-index",
            str(config["cohort"]["actor_model_index"]),
            "--output",
            str(worker_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=float(config["resources"]["maximum_wall_seconds"]),
        )
        (run_dir / "worker.log").write_text(
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise R36ExperimentError(f"StreetGS sensor worker 失败：{completed.returncode}")
        rows = _read_jsonl(worker_dir / "FRAME_METRICS.jsonl")
        audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
        for row in rows:
            _verify(worker_dir / row["sensor_path"], row["sensor_sha256"])
        thresholds = config["thresholds"]
        expected_frames = [int(value) for value in config["cohort"]["frame_indices"]]
        frame_denominator_exact = [int(row["frame_index"]) for row in rows] == expected_frames
        actor_geometry_equivalent = all(
            row["native_actor_field_max_error"]["means_m"]
            <= float(thresholds["maximum_means_error_m"])
            and row["native_actor_field_max_error"]["quaternions_wxyz"]
            <= float(thresholds["maximum_quaternion_error"])
            and row["native_actor_field_max_error"]["scales_m"]
            <= float(thresholds["maximum_static_field_error"])
            and row["native_actor_field_max_error"]["opacities"]
            <= float(thresholds["maximum_static_field_error"])
            and row["native_actor_field_max_error"]["view_dependent_rgb"]
            <= float(thresholds["maximum_static_field_error"])
            for row in rows
        )
        sensor_equivalent = all(
            row["full_sensor_rgb_mae"] <= float(thresholds["maximum_rgb_mae"])
            and row["full_sensor_rgb_p99_absolute_error"]
            <= float(thresholds["maximum_rgb_p99_absolute_error"])
            and row["full_sensor_depth_mae_m"]
            <= float(thresholds["maximum_depth_mae_m"])
            and row["full_sensor_opacity_mae"]
            <= float(thresholds["maximum_opacity_mae"])
            for row in rows
        )
        wall_seconds = time.monotonic() - started
        peak_mib = float(audit["peak_torch_reserved_bytes"]) / (1024**2)
        checks = {
            "r35_authority_accepted": r35_gate["checks"]["passed"],
            "frame_denominator_exact": frame_denominator_exact,
            "actor_visible_support_nontrivial": all(
                int(row["actor_effect_pixels"])
                >= int(thresholds["minimum_actor_effect_pixels"])
                for row in rows
            ),
            "compiled_actor_matches_native_fields": actor_geometry_equivalent,
            "full_sensor_equivalent": sensor_equivalent,
            "compiled_sensor_repeat_exact": all(row["compiled_repeat_exact"] for row in rows),
            "checkpoint_immutable": audit["checkpoint_sha256_before"]
            == audit["checkpoint_sha256_after"]
            == sources["streetgs_checkpoint_sha256"],
            "compiled_package_immutable": audit["package_manifest_sha256_before"]
            == audit["package_manifest_sha256_after"]
            == sources["r35_package_manifest_sha256"],
            "upstream_commit_exact": audit["upstream_commit"]
            == sources["streetgs_upstream_commit"],
            "source_immutable": all(
                _sha256(path) == expected_sha for path, expected_sha in frozen_files.items()
            )
            and all(_sha256(path) == expected_sha for path, expected_sha in package_files.items()),
            "gpu_within_budget": peak_mib
            <= float(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds
            <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": not audit["training_started"],
            "confirmation_not_read": not audit["confirmation_content_read"],
        }
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir / "R36_GATE.json",
            {
                "schema_version": "worldsim_v6.r36_gate.v1",
                "checks": checks,
                "decision": "accept_compiled_actor_sensor_renderer_integration"
                if checks["passed"]
                else "reject_or_repair_actor_sensor_integration",
            },
        )
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r36_resource_audit.v1",
                "gpu_used": True,
                "peak_torch_allocated_mib": float(audit["peak_torch_allocated_bytes"])
                / (1024**2),
                "peak_torch_reserved_mib": peak_mib,
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r36_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_compiled_actor_sensor_equivalence"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "frame_count": len(rows),
            "actor_id": "actor_0000",
            "maximum_rgb_mae": max(row["full_sensor_rgb_mae"] for row in rows),
            "maximum_rgb_p99_absolute_error": max(
                row["full_sensor_rgb_p99_absolute_error"] for row in rows
            ),
            "maximum_depth_mae_m": max(row["full_sensor_depth_mae_m"] for row in rows),
            "minimum_actor_effect_pixels": min(row["actor_effect_pixels"] for row in rows),
            "compiled_repeat_exact": all(row["compiled_repeat_exact"] for row in rows),
            "full_worldsim_coverage": False,
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "R36_GATE.json",
            "SUMMARY.json",
            "RESOURCE_AUDIT.json",
            "worker.log",
            "worker/FRAME_METRICS.jsonl",
            "worker/WORKER_AUDIT.json",
            *[f"worker/{row['sensor_path']}" for row in rows],
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r36_manifest.v1",
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
        default=Path("configs/worldsim_v6/r36_actor_sensor_equivalence_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


"""WorldSim V6 R37：反事实 actor 轨迹平移的 compiled/native sensor 对照。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


TASK_ID = "WS-V6-R37-ACTOR-TRAJECTORY-EDIT-SENSOR-EQUIVALENCE-01"


class R37ExperimentError(RuntimeError):
    """R37 正式实验合同失败。"""


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
        raise R37ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / relative).resolve()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise R37ExperimentError(f"冻结输入漂移：{path}")


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R37ExperimentError("正式 R37 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R37ExperimentError("R37 task_id 漂移")
    sources = config["sources"]
    r36_run = _resolve_runs_uri(sources["r36_run"])
    r35_run = _resolve_runs_uri(sources["r35_run"])
    package = r35_run / "package"
    checkpoint = Path(sources["streetgs_checkpoint"])
    upstream = Path(sources["streetgs_upstream_root"])
    baseline_sensor = r36_run / sources["r36_baseline_sensor"]
    frozen_files = {
        r36_run / "MANIFEST.json": sources["r36_manifest_sha256"],
        r36_run / "R36_GATE.json": sources["r36_gate_sha256"],
        r36_run / "SUMMARY.json": sources["r36_summary_sha256"],
        r36_run / "worker/FRAME_METRICS.jsonl": sources["r36_frame_metrics_sha256"],
        baseline_sensor: sources["r36_baseline_sensor_sha256"],
        r35_run / "MANIFEST.json": sources["r35_manifest_sha256"],
        package / "PACKAGE_MANIFEST.json": sources["r35_package_manifest_sha256"],
        checkpoint: sources["streetgs_checkpoint_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R37ExperimentError("StreetGS upstream commit 漂移")
    r36_gate = json.loads((r36_run / "R36_GATE.json").read_text(encoding="utf-8"))
    if not r36_gate["checks"]["passed"]:
        raise R37ExperimentError("R36 sensor equivalence authority 未通过")
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
        raise R37ExperimentError("R37 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__trajectory-edit-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        baseline_rgb = np.load(baseline_sensor, allow_pickle=False)["native_rgb"].astype(np.float32)
        rows: list[dict[str, Any]] = []
        peak_reserved_bytes = 0
        worker_wall_seconds = 0.0
        for intervention in config["interventions"]:
            worker_dir = run_dir / intervention["id"]
            delta = [float(value) for value in intervention["translation_delta_m"]]
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
                str(config["cohort"]["frame_index"]),
                "--actor-model-index",
                str(config["cohort"]["actor_model_index"]),
                "--translation-delta-m",
                ",".join(str(value) for value in delta),
                "--output",
                str(worker_dir),
            ]
            completed = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=float(config["resources"]["maximum_worker_seconds"]),
            )
            (run_dir / f"{intervention['id']}.log").write_text(
                completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise R37ExperimentError(
                    f"trajectory edit worker 失败：{intervention['id']} rc={completed.returncode}"
                )
            metric_rows = [
                json.loads(line)
                for line in (worker_dir / "FRAME_METRICS.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            if len(metric_rows) != 1:
                raise R37ExperimentError("每个 intervention 必须恰有一帧")
            row = metric_rows[0]
            audit = json.loads((worker_dir / "WORKER_AUDIT.json").read_text(encoding="utf-8"))
            sensor_path = worker_dir / row["sensor_path"]
            _verify(sensor_path, row["sensor_sha256"])
            edited_rgb = np.load(sensor_path, allow_pickle=False)["native_rgb"].astype(np.float32)
            changed_pixels = int(
                np.count_nonzero(
                    np.mean(np.abs(edited_rgb - baseline_rgb), axis=-1)
                    > float(config["thresholds"]["counterfactual_rgb_change_threshold"])
                )
            )
            row.update(
                {
                    "intervention_id": intervention["id"],
                    "counterfactual_changed_pixels_vs_logged": changed_pixels,
                    "worker_audit_sha256": _sha256(worker_dir / "WORKER_AUDIT.json"),
                    "frame_metrics_sha256": _sha256(worker_dir / "FRAME_METRICS.jsonl"),
                }
            )
            rows.append(row)
            peak_reserved_bytes = max(peak_reserved_bytes, int(audit["peak_torch_reserved_bytes"]))
            worker_wall_seconds += float(audit["wall_seconds"])
            if not (
                audit["checkpoint_sha256_before"]
                == audit["checkpoint_sha256_after"]
                == sources["streetgs_checkpoint_sha256"]
            ):
                raise R37ExperimentError("worker checkpoint 漂移")
        (run_dir / "INTERVENTION_METRICS.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        thresholds = config["thresholds"]
        actor_fields_pass = all(
            row["native_actor_field_max_error"]["means_m"]
            <= float(thresholds["maximum_means_error_m"])
            and row["native_actor_field_max_error"]["quaternions_wxyz"]
            <= float(thresholds["maximum_quaternion_error"])
            and all(
                row["native_actor_field_max_error"][key]
                <= float(thresholds["maximum_static_field_error"])
                for key in ("scales_m", "opacities", "view_dependent_rgb")
            )
            for row in rows
        )
        sensors_pass = all(
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
        peak_mib = peak_reserved_bytes / (1024**2)
        checks = {
            "r36_authority_accepted": r36_gate["checks"]["passed"],
            "intervention_denominator_exact": len(rows) == len(config["interventions"])
            and [row["intervention_id"] for row in rows]
            == [row["id"] for row in config["interventions"]],
            "counterfactual_effect_nontrivial": all(
                row["counterfactual_changed_pixels_vs_logged"]
                >= int(thresholds["minimum_counterfactual_changed_pixels"])
                for row in rows
            ),
            "edited_actor_visible_support_nontrivial": all(
                row["actor_effect_pixels"] >= int(thresholds["minimum_actor_effect_pixels"])
                for row in rows
            ),
            "compiled_edits_match_native_actor_fields": actor_fields_pass,
            "compiled_edits_match_native_full_sensors": sensors_pass,
            "compiled_repeats_exact": all(row["compiled_repeat_exact"] for row in rows),
            "native_translation_state_restored_exact": all(
                row["native_translation_state_restored_exact"] for row in rows
            ),
            "source_immutable": all(
                _sha256(path) == expected_sha for path, expected_sha in frozen_files.items()
            )
            and all(_sha256(path) == expected_sha for path, expected_sha in package_files.items()),
            "gpu_within_budget": peak_mib
            <= float(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds
            <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir / "R37_GATE.json",
            {
                "schema_version": "worldsim_v6.r37_gate.v1",
                "checks": checks,
                "decision": "accept_actor_trajectory_edit_sensor_compiler"
                if checks["passed"]
                else "reject_or_repair_actor_trajectory_edit_sensor_compiler",
            },
        )
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r37_resource_audit.v1",
                "gpu_used": True,
                "peak_torch_reserved_mib": peak_mib,
                "worker_wall_seconds_sum": worker_wall_seconds,
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r37_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_actor_trajectory_edit_sensor_compiler"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "intervention_count": len(rows),
            "minimum_counterfactual_changed_pixels": min(
                row["counterfactual_changed_pixels_vs_logged"] for row in rows
            ),
            "maximum_rgb_mae": max(row["full_sensor_rgb_mae"] for row in rows),
            "maximum_depth_mae_m": max(row["full_sensor_depth_mae_m"] for row in rows),
            "compiled_repeats_exact": all(row["compiled_repeat_exact"] for row in rows),
            "physical_trajectory_validity": "ABSTAIN",
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["R37_GATE.json", "SUMMARY.json", "RESOURCE_AUDIT.json", "INTERVENTION_METRICS.jsonl"]
        for intervention, row in zip(config["interventions"], rows):
            tracked.extend(
                [
                    f"{intervention['id']}.log",
                    f"{intervention['id']}/FRAME_METRICS.jsonl",
                    f"{intervention['id']}/WORKER_AUDIT.json",
                    f"{intervention['id']}/{row['sensor_path']}",
                ]
            )
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r37_manifest.v1",
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
        default=Path("configs/worldsim_v6/r37_actor_trajectory_edit_sensor_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


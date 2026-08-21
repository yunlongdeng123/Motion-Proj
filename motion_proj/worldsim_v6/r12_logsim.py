"""WorldSim V6 R12 静态 LogSim 正式实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


TASK_ID = "WS-V6-R12-LOGSIM-01"
ALLOWED_TASK_IDS = {TASK_ID, "WS-V6-R19-RGBD-TEMPORAL-STATIC-LOGSIM-01"}


class R12ExperimentError(RuntimeError):
    """R12 正式合同失败。"""


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


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    if not uri.startswith(prefix) or ".." in Path(uri[len(prefix):]).parts:
        raise R12ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / uri[len(prefix):]).resolve()


def _run_checked(command: list[str], cwd: Path, log_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "20260821"
    # CUDA 10.2+ 的确定性 cuBLAS 路径要求在子进程启动前冻结 workspace 配置。
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise R12ExperimentError(f"感知 worker 失败，见 {log_path}")


def run_experiment(
    repo_root: Path, config_path: Path, run_root: Path, semantic_model_root: Path
) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R12ExperimentError("正式 R12 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task_id = str(config.get("task_id"))
    if task_id not in ALLOWED_TASK_IDS:
        raise R12ExperimentError("R12 task_id 漂移")
    sources = config["sources"]
    r11_run = _resolve_runs_uri(sources["r11_run"])
    r9_run = _resolve_runs_uri(sources["r9_run"])
    r11_gate_name = str(sources.get("r11_gate_file", "R11_GATE.json"))
    proposal_directory = str(
        sources.get("proposal_directory", "cross_frontend_reconstruction_proposals")
    )
    frozen_files = {
        r11_run / "MANIFEST.json": sources["r11_manifest_sha256"],
        r11_run / r11_gate_name: sources["r11_gate_sha256"],
        r11_run / "package/PACKAGE_MANIFEST.json": sources["r11_package_manifest_sha256"],
        r9_run / "MANIFEST.json": sources["r9_manifest_sha256"],
        semantic_model_root / config["verifier_model"]["model_file"]: config["verifier_model"]["model_sha256"],
    }
    for path, expected in frozen_files.items():
        if _sha256(path) != expected:
            raise R12ExperimentError(f"冻结输入漂移：{path}")
    if shutil.disk_usage(run_root).free / (1024**3) < float(config["resources"]["minimum_disk_free_gib"]):
        raise R12ExperimentError("R12 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / task_id / f"{now.strftime('%Y%m%dT%H%M%SZ')}__logsim-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        replay_dir = run_dir / "replays"
        perception_dir = run_dir / "perception"
        replay_dir.mkdir()
        assets = _read_jsonl(r11_run / "package/ASSET_REGISTRY.jsonl")
        provenance = {
            row["asset_id"]: row
            for row in _read_jsonl(r11_run / "package/PROVENANCE.jsonl")
        }
        expected_count = int(config["cohort"]["expected_static_chunk_count"])
        if len(assets) != expected_count:
            raise R12ExperimentError("R11 静态资产分母漂移")
        repeat_count = int(config["cohort"]["replay_count"])
        index_rows: list[dict[str, Any]] = []
        replay_rows: list[dict[str, Any]] = []
        immutable_before = {str(path): _sha256(path) for path in frozen_files}
        for asset in assets:
            case_id = asset["case_id"]
            payload = r11_run / "package" / asset["payload"]
            if _sha256(payload) != asset["payload_sha256"]:
                raise R12ExperimentError(f"bake payload 漂移：{case_id}")
            verifier_input = r9_run / "verifier_inputs" / f"{case_id}.npz"
            proposal = (
                r9_run
                / proposal_directory
                / f"{case_id}__repeat1.npy"
            )
            if _sha256(proposal) != provenance[asset["asset_id"]]["source_proposal_sha256"]:
                raise R12ExperimentError(f"源 proposal 漂移：{case_id}")
            with np.load(verifier_input, allow_pickle=False) as archive:
                input_image = np.asarray(archive["input_image"], dtype=np.uint8)
            expected_image = np.load(proposal, allow_pickle=False).astype(np.uint8)
            case_hashes: list[str] = []
            for repeat_index in range(1, repeat_count + 1):
                with np.load(payload, allow_pickle=False) as archive:
                    coordinates = np.asarray(archive["coordinates_yx"], dtype=np.int64)
                    rgb = np.asarray(archive["rgb_uint8"], dtype=np.uint8)
                    height = int(np.asarray(archive["canvas_height"]).item())
                    width = int(np.asarray(archive["canvas_width"]).item())
                if (height, width) != input_image.shape[:2]:
                    raise R12ExperimentError(f"replay canvas 漂移：{case_id}")
                replay = input_image.copy()
                replay[coordinates[:, 0], coordinates[:, 1]] = rgb
                replay_path = replay_dir / f"{case_id}__repeat{repeat_index}.npy"
                np.save(replay_path, replay, allow_pickle=False)
                array_hash = _array_sha256(replay)
                case_hashes.append(array_hash)
                absolute = np.abs(replay.astype(np.int16) - expected_image.astype(np.int16))
                replay_rows.append(
                    {
                        "schema_version": "worldsim_v6.r12_replay.v1",
                        "case_id": case_id,
                        "repeat_index": repeat_index,
                        "replay_array_sha256": array_hash,
                        "replay_file_sha256": _sha256(replay_path),
                        "expected_array_sha256": _array_sha256(expected_image),
                        "sensor_rgb_mae": float(absolute.mean()),
                        "sensor_rgb_max_absolute_error": int(absolute.max()),
                        "source_proposal_exact": bool(np.array_equal(replay, expected_image)),
                    }
                )
                index_rows.append(
                    {
                        "case_id": case_id,
                        "repeat_index": repeat_index,
                        "replay_path": str(replay_path),
                    }
                )
            if len(set(case_hashes)) != 1:
                raise R12ExperimentError(f"重复回放不 exact：{case_id}")
        _write_jsonl(run_dir / "REPLAY_INDEX.jsonl", index_rows)
        _write_jsonl(run_dir / "REPLAY_METRICS.jsonl", replay_rows)
        _run_checked(
            [
                "/root/autodl-tmp/envs/motionproj/bin/python",
                str(repo_root / "scripts/worldsim_v6/r12_perception_worker.py"),
                "--index",
                str(run_dir / "REPLAY_INDEX.jsonl"),
                "--model-root",
                str(semantic_model_root),
                "--output-dir",
                str(perception_dir),
            ],
            repo_root,
            run_dir / "perception.log",
        )
        worker = json.loads(
            (perception_dir / "WORKER_RESULT.json").read_text(encoding="utf-8")
        )
        perception_rows = _read_jsonl(perception_dir / "PERCEPTION_OUTPUTS.jsonl")
        grouped: dict[str, list[str]] = {}
        for row in perception_rows:
            grouped.setdefault(row["case_id"], []).append(row["label_array_sha256"])
        perception_repeat_exact = all(
            len(values) == repeat_count and len(set(values)) == 1
            for values in grouped.values()
        )
        sensor_exact = all(
            row["source_proposal_exact"] and row["sensor_rgb_mae"] == 0.0
            for row in replay_rows
        )
        repeat_exact = all(
            len(
                {
                    row["replay_array_sha256"]
                    for row in replay_rows
                    if row["case_id"] == case_id
                }
            )
            == 1
            for case_id in grouped
        )
        unsupported = config["unsupported_factors"]
        unsupported_abstain = all(
            str(value).startswith("ABSTAIN") for value in unsupported.values()
        )
        immutable_after = {str(path): _sha256(path) for path in frozen_files}
        wall_seconds = time.monotonic() - started
        checks = {
            "all_static_assets_replayed": len(grouped) == expected_count,
            "sensor_exact": sensor_exact,
            "repeated_run_exact": repeat_exact,
            "perception_repeat_exact": perception_repeat_exact,
            "source_immutable": immutable_before == immutable_after,
            "no_generator_runtime_dependency": all(
                not row["runtime_generator_dependency"] for row in assets
            ),
            "unsupported_factors_abstain": unsupported_abstain,
            "gpu_within_budget": worker["peak_gpu_memory_mib"]
            <= int(config["resources"]["maximum_peak_gpu_memory_mib"]),
            "wall_within_budget": wall_seconds
            <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir / ("R12_GATE.json" if task_id == TASK_ID else "R19_GATE.json"),
            {
                "schema_version": "worldsim_v6.r12_gate.v1",
                "checks": checks,
                "supported_scope": "static_rgb_sensor_and_frozen_semantic_output_replay",
                "full_logsim_coverage": False,
                "unsupported_factors": unsupported,
                "decision": "proceed_to_dynamic_logsim_evidence_search"
                if checks["passed"]
                else "reject_static_logsim_hypothesis",
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r12_summary.v1",
            "task_id": task_id,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_static_scope"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "static_case_count": len(grouped),
            "replay_count_per_case": repeat_count,
            "sensor_exact_case_count": sum(
                all(
                    row["source_proposal_exact"]
                    for row in replay_rows
                    if row["case_id"] == case_id
                )
                for case_id in grouped
            ),
            "perception_exact_case_count": sum(
                len(set(values)) == 1 for values in grouped.values()
            ),
            "peak_gpu_memory_mib": worker["peak_gpu_memory_mib"],
            "wall_seconds": wall_seconds,
            "unsupported_factors": unsupported,
            "claim_boundary": config["claim_boundary"],
            "training_started": False,
            "confirmation_content_read": False,
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = [
            "R12_GATE.json" if task_id == TASK_ID else "R19_GATE.json",
            "SUMMARY.json",
            "REPLAY_INDEX.jsonl",
            "REPLAY_METRICS.jsonl",
            "perception/PERCEPTION_OUTPUTS.jsonl",
            "perception/WORKER_RESULT.json",
            "perception.log",
        ]
        tracked += [
            str(path.relative_to(run_dir)) for path in sorted(replay_dir.glob("*.npy"))
        ]
        tracked += [
            str(path.relative_to(run_dir))
            for path in sorted(perception_dir.glob("*.npy"))
        ]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r12_manifest.v1",
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
                "task_id": task_id,
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
                "task_id": task_id,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/worldsim_v6/r12_logsim_v0.yaml")
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    parser.add_argument(
        "--semantic-model-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/models/worldsim_v6/r9_semantic_deeplab_cityscapes"
        ),
    )
    args = parser.parse_args()
    run_dir = run_experiment(
        args.repo_root, args.config, args.run_root, args.semantic_model_root
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

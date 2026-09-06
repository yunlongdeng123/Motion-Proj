"""将 V7.1 已暴露 Actor cache 转为 AdaPoinTr 的独立诊断缓存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_proj.worldsim_v72.evaluation.surface_metrics import (
    deterministic_farthest_point_sample,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _fixed_count(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        raise ValueError("不能从空点集构造固定点数张量")
    if len(points) >= int(count):
        return deterministic_farthest_point_sample(points, int(count))
    full_repeats, remainder = divmod(int(count), len(points))
    parts = [np.tile(points, (full_repeats, 1))]
    if remainder:
        parts.append(deterministic_farthest_point_sample(points, remainder))
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _scene_to_log(path: Path) -> dict[str, str]:
    return {
        str(row["name"]): str(row["log_token"])
        for row in json.loads(path.read_text(encoding="utf-8"))
    }


def run(config_path: Path, run_id: str) -> dict[str, Any]:
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    if config.get("data_role") != "legacy_diagnostic":
        raise ValueError("V7.1 adapter 只允许 legacy_diagnostic")
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    run_dir = Path(config["runs_root"]) / "worldsim_v72" / config["task_id"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "status.json", {"status": "running", "phase": "export"})
    started = time.monotonic()
    try:
        git_commit = _git_commit()
        resolved = {
            **config,
            "run_id": run_id,
            "git_commit": git_commit,
            "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        scene_to_log = _scene_to_log(Path(config["scene_metadata"]))
        source_paths = sorted(Path(config["cache_root"]).glob("*/*.npz"))[
            : int(config["maximum_source_actors"])
        ]
        eligible: list[Path] = []
        for path in source_paths:
            with np.load(path, allow_pickle=False) as payload:
                if len(payload["candidates"]):
                    eligible.append(path)
        holdout_stride = int(config["holdout_stride"])
        samples: list[dict[str, Any]] = []
        for index, path in enumerate(eligible):
            split = "holdout" if index % holdout_stride == 0 else "train"
            with np.load(path, allow_pickle=False) as payload:
                partial = np.asarray(payload["canonical"], dtype=np.float32)
                target = np.asarray(payload["target"], dtype=np.float32)
                size_lwh_m = np.asarray(payload["size_lwh_m"], dtype=np.float32)
                scene_name = str(payload["scene_name"])
                track_id = str(payload["track_id"])
                category = str(payload["category"])
                hazardous = bool(payload["hazardous"])
            scale_m = float(np.max(size_lwh_m) * 0.5)
            if not np.isfinite(scale_m) or scale_m <= 0.0:
                raise ValueError(f"非法 Actor 尺度: {path}")
            partial_fixed = _fixed_count(partial, int(config["input_point_count"]))
            target_fixed = _fixed_count(target, int(config["target_point_count"]))
            output_path = output_root / split / scene_name / f"{track_id}.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                schema_version=np.asarray("worldsim_v72.adapointr_legacy_example.v1"),
                partial_normalized=(partial_fixed / scale_m).astype(np.float32),
                target_normalized=(target_fixed / scale_m).astype(np.float32),
                size_lwh_m=size_lwh_m,
                scale_m=np.asarray(scale_m, dtype=np.float32),
                scene_name=np.asarray(scene_name),
                log_id=np.asarray(scene_to_log[scene_name]),
                track_id=np.asarray(track_id),
                category=np.asarray(category),
                hazardous=np.asarray(hazardous),
                native_input_point_count=np.asarray(len(partial), dtype=np.int32),
                native_target_point_count=np.asarray(len(target), dtype=np.int32),
            )
            temporary.replace(output_path)
            samples.append(
                {
                    "split": split,
                    "scene_name": scene_name,
                    "log_id": scene_to_log[scene_name],
                    "track_id": track_id,
                    "relative_path": str(output_path.relative_to(output_root)),
                    "native_input_point_count": len(partial),
                    "native_target_point_count": len(target),
                    "scale_m": scale_m,
                    "target_used_for_input_sampling": False,
                }
            )
            if (index + 1) % 100 == 0 or index + 1 == len(eligible):
                print(
                    json.dumps(
                        {"stage": "adapointr_export", "progress": f"{index + 1}/{len(eligible)}"}
                    ),
                    flush=True,
                )
        train_count = sum(row["split"] == "train" for row in samples)
        holdout_count = sum(row["split"] == "holdout" for row in samples)
        if train_count != int(config["expected_train_actor_count"]):
            raise RuntimeError(f"train actor count {train_count} 与冻结预期不一致")
        if holdout_count != int(config["expected_holdout_actor_count"]):
            raise RuntimeError(f"holdout actor count {holdout_count} 与冻结预期不一致")
        _write_jsonl(output_root / "SAMPLES.jsonl", samples)
        fingerprint_payload = json.dumps(
            {
                "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
                "git_commit": git_commit,
                "identities": [
                    [row["scene_name"], row["track_id"], row["split"]] for row in samples
                ],
            },
            sort_keys=True,
        ).encode()
        manifest = {
            "schema_version": "worldsim_v72.adapointr_legacy_manifest.v1",
            "status": "done",
            "task_id": config["task_id"],
            "run_id": run_id,
            "data_role": config["data_role"],
            "pretrained_holdout_exposure": True,
            "sample_count": len(samples),
            "train_actor_count": train_count,
            "holdout_actor_count": holdout_count,
            "train_log_count": len({row["log_id"] for row in samples if row["split"] == "train"}),
            "holdout_log_count": len({row["log_id"] for row in samples if row["split"] == "holdout"}),
            "input_point_count": int(config["input_point_count"]),
            "target_point_count": int(config["target_point_count"]),
            "normalization": config["normalization"],
            "target_used_for_input_sampling": False,
            "source_test_read": False,
            "external_test_read": False,
            "git_commit": git_commit,
            "fingerprint": hashlib.sha256(fingerprint_payload).hexdigest(),
            "failure_ledger_refs": list(config["failure_ledger_refs"]),
            "failure_ledger_delta": str(config["failure_ledger_delta"]),
        }
        _write_json(output_root / "manifest.json", manifest)
        summary = {
            **manifest,
            "schema_version": "worldsim_v72.adapointr_legacy_summary.v1",
            "verdict": "legacy_adapointr_adapter_ready_gpu_training_pending",
            "resources": {
                "device": "cpu",
                "peak_gpu_memory_gib": 0.0,
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                "wall_seconds": time.monotonic() - started,
            },
        }
        _write_jsonl(run_dir / "SAMPLES.jsonl", samples)
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "fingerprint.json",
            {
                "algorithm": "sha256",
                "value": manifest["fingerprint"],
                "scope": "config+git_commit+ordered_actor_identities_and_splits",
            },
        )
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"status": "done", "phase": "export", "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        )
        return summary
    except Exception as error:
        _write_json(
            run_dir / "status.json",
            {"status": "failed", "phase": "export", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

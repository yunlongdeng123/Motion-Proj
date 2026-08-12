#!/usr/bin/env python3
"""从完成的 matched AD-GS formal run 生成 baseline matrix 注册记录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


TASK_ID = "WS-V4-B0-MATCHED-BASELINES-01"
CHECKPOINT_FILES = ("point_cloud.ply", "deform.pth", "env.pth")


class ADGSRegistrationError(RuntimeError):
    """AD-GS formal run 不能安全登记。"""


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ADGSRegistrationError(f"terminal artifact 缺失：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ADGSRegistrationError(f"terminal artifact 必须为 mapping：{path}")
    return value


def _exact_checkpoint_files(run_dir: Path, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    if checkpoint.get("iteration") != 60000:
        raise ADGSRegistrationError("formal checkpoint iteration 必须为 60000")
    configured = checkpoint.get("files")
    if not isinstance(configured, Mapping) or set(configured) != set(CHECKPOINT_FILES):
        raise ADGSRegistrationError("formal checkpoint 必须且仅包含三个核心文件")
    files: dict[str, Any] = {}
    for name in CHECKPOINT_FILES:
        row = configured[name]
        if not isinstance(row, Mapping):
            raise ADGSRegistrationError(f"checkpoint row 非 mapping：{name}")
        path = Path(str(row.get("path", ""))).resolve()
        if not path.is_file() or not path.is_relative_to(run_dir):
            raise ADGSRegistrationError(f"checkpoint 不存在或不在 run 内：{path}")
        actual = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if row.get("bytes") != actual["bytes"] or row.get("sha256") != actual["sha256"]:
            raise ADGSRegistrationError(f"checkpoint 内容漂移：{name}")
        files[name] = actual
    return files


def registration_record(run_dir: Path, *, expected_scene: str) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "summary.json"
    status_path = run_dir / "status.json"
    manifest_path = run_dir / "manifest.json"
    fingerprint_path = run_dir / "fingerprint.json"
    stage_path = run_dir / "stages" / "train_formal.json"
    summary = load_json(summary_path)
    status = load_json(status_path)
    manifest = load_json(manifest_path)
    fingerprint = load_json(fingerprint_path)
    stage = load_json(stage_path)

    if summary.get("schema_version") != "worldsim_v4_adgs_summary_v1":
        raise ADGSRegistrationError("AD-GS summary schema 漂移")
    for payload in (summary, status):
        if payload.get("task_id") != TASK_ID or payload.get("scene") != expected_scene:
            raise ADGSRegistrationError("AD-GS terminal task/scene 漂移")
    if summary.get("status") != "done" or status.get("status") != "done":
        raise ADGSRegistrationError("AD-GS formal run 尚未完成")
    if summary.get("mode") != "formal" or status.get("mode") != "formal":
        raise ADGSRegistrationError("AD-GS run 不是 formal")
    if status.get("summary_sha256") != sha256_file(summary_path):
        raise ADGSRegistrationError("AD-GS summary SHA 漂移")
    if manifest.get("schema_version") != "worldsim_v4_adgs_run_manifest_v1" or manifest.get("status") != "done":
        raise ADGSRegistrationError("AD-GS manifest 尚未完成或 schema 漂移")
    if stage.get("stage") != "train_formal" or stage.get("status") != "done" or stage.get("return_code") != 0:
        raise ADGSRegistrationError("AD-GS train_formal stage 未成功")

    for field in ("development_content_read", "heldout_content_read", "test_quality_read"):
        if summary.get(field) is not False:
            raise ADGSRegistrationError(f"AD-GS formal run 未证明 {field}=false")
    if summary.get("training_started") is not True or summary.get("model_inference_started") is not False:
        raise ADGSRegistrationError("AD-GS formal execution flags 漂移")
    if summary.get("project_git", {}).get("dirty") is not False:
        raise ADGSRegistrationError("AD-GS formal run 结束时 project 非 clean")

    checkpoint = summary.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ADGSRegistrationError("AD-GS summary checkpoint 缺失")
    if fingerprint.get("checkpoint") != checkpoint or manifest.get("checkpoint") != checkpoint:
        raise ADGSRegistrationError("AD-GS checkpoint terminal evidence 不一致")
    files = _exact_checkpoint_files(run_dir, checkpoint)

    duration = stage.get("duration_seconds")
    peak_gpu = stage.get("peak_gpu_memory_mib")
    peak_memory = stage.get("peak_cgroup_memory_bytes")
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise ADGSRegistrationError("AD-GS duration 无效")
    if not isinstance(peak_gpu, int) or peak_gpu <= 0:
        raise ADGSRegistrationError("AD-GS peak GPU memory 无效")
    if not isinstance(peak_memory, int) or peak_memory <= 0:
        raise ADGSRegistrationError("AD-GS peak cgroup memory 无效")

    return {
        "state": "matched_formal_done",
        "run": str(run_dir),
        "step": 60000,
        "seed": 0,
        "duration_seconds": duration,
        "peak_gpu_memory_mib": peak_gpu,
        "peak_cgroup_memory_bytes": peak_memory,
        "development_content_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "manifest_sha256": sha256_file(manifest_path),
        "summary_sha256": sha256_file(summary_path),
        "files": files,
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = registration_record(args.run_dir, expected_scene=args.scene)
    atomic_json(args.output, record)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

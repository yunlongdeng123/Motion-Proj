#!/usr/bin/env python
"""收口 V5 M1 development raw extraction，并生成不可变 formal run 证据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RawAuditError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: dict[str, Any], omit: str | None = None) -> str:
    copy = {key: value for key, value in payload.items() if key != omit}
    encoded = json.dumps(
        copy, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_content_address(payload: dict[str, Any], path: Path) -> None:
    expected = payload.get("manifest_sha256")
    actual = sha256_payload(payload, omit="manifest_sha256")
    if expected != actual:
        raise RawAuditError(f"manifest content address 漂移: {path}: {expected} != {actual}")


def audit(raw_batch_path: Path) -> dict[str, Any]:
    batch = json.loads(raw_batch_path.read_text(encoding="utf-8"))
    validate_content_address(batch, raw_batch_path)
    if batch.get("complete") is not True:
        raise RawAuditError("raw batch 未完成")
    if batch.get("quality_read") is not False:
        raise RawAuditError("raw batch quality_read 合同漂移")
    if int(batch["scene_count"]) != 8:
        raise RawAuditError(f"scene denominator 漂移: {batch['scene_count']}")
    if len(batch.get("scenes", [])) != int(batch["scene_count"]):
        raise RawAuditError("scene manifest list denominator 漂移")
    if int(batch["required_count"]) != int(batch["present_count"]):
        raise RawAuditError("raw batch present denominator 不完整")
    raw_root = Path(batch["raw_root"])
    total_files = 0
    total_bytes = 0
    scenes = []
    for row in batch["scenes"]:
        path = Path(row["manifest"])
        scene = json.loads(path.read_text(encoding="utf-8"))
        validate_content_address(scene, path)
        if scene.get("manifest_sha256") != row["manifest_sha256"]:
            raise RawAuditError(f"scene manifest content address 漂移: {path}")
        if scene.get("complete") is not True:
            raise RawAuditError(f"scene manifest 未完成: {path}")
        if scene["scene_name"] != row["scene_name"]:
            raise RawAuditError(f"scene identity 漂移: {path}")
        if int(scene["required_count"]) != int(scene["present_count"]):
            raise RawAuditError(f"scene denominator 不完整: {path}")
        for member in scene["files"]:
            sensor = raw_root / member["filename"]
            if not sensor.is_file() or sensor.stat().st_size != int(member["bytes"]):
                raise RawAuditError(f"sensor bytes 漂移: {sensor}")
        file_count = len(scene["files"])
        byte_count = sum(int(member["bytes"]) for member in scene["files"])
        if file_count != int(row["required_count"]) or byte_count != int(row["bytes"]):
            raise RawAuditError(f"batch/scene 计数漂移: {path}")
        total_files += file_count
        total_bytes += byte_count
        scenes.append(
            {
                "scene_name": scene["scene_name"],
                "scene_index": int(scene["scene_index"]),
                "required_count": file_count,
                "bytes": byte_count,
                "manifest": str(path),
                "manifest_sha256": row["manifest_sha256"],
                "manifest_file_sha256": sha256_file(path),
            }
        )
    if total_files != int(batch["required_count"]):
        raise RawAuditError(f"file denominator 漂移: {total_files} != {batch['required_count']}")
    if total_bytes != int(batch["total_bytes"]):
        raise RawAuditError(f"byte denominator 漂移: {total_bytes} != {batch['total_bytes']}")
    shard_index = Path(batch["member_shard_index"])
    if sha256_file(shard_index) != batch["member_shard_index_sha256"]:
        raise RawAuditError("member→shard index SHA 漂移")
    mapping = json.loads(shard_index.read_text(encoding="utf-8"))
    if len(mapping) != total_files:
        raise RawAuditError(f"member→shard denominator 漂移: {len(mapping)} != {total_files}")
    return {
        "scene_count": len(scenes),
        "required_count": total_files,
        "present_count": total_files,
        "total_bytes": total_bytes,
        "raw_root": str(raw_root),
        "raw_batch_manifest": str(raw_batch_path),
        "raw_batch_manifest_sha256": sha256_file(raw_batch_path),
        "member_shard_index": str(shard_index),
        "member_shard_index_sha256": batch["member_shard_index_sha256"],
        "scenes": scenes,
    }


def finalize(run_dir: Path, raw_batch_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    raw_batch_path = raw_batch_path.resolve()
    if not run_dir.is_dir():
        raise RawAuditError(f"run 目录缺失: {run_dir}")
    if (run_dir / "status.json").exists():
        raise RawAuditError(f"run 已收口，禁止覆盖: {run_dir}")
    if not (run_dir / "stdout.log").is_file() or not (run_dir / "source_commit.txt").is_file():
        raise RawAuditError("extraction launch evidence 不完整")
    process_probe = subprocess.run(
        ["pgrep", "-f", "prepare_worldsim_v5_drivestudio_raw.py --extract"],
        text=True,
        capture_output=True,
        check=False,
    )
    if process_probe.returncode == 0 and process_probe.stdout.strip():
        raise RawAuditError("raw extraction process 仍在运行")
    evidence = audit(raw_batch_path)
    snapshot = run_dir / "source_snapshot" / Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), snapshot)
    finished_at = datetime.now(timezone.utc).isoformat()
    try:
        started_at = datetime.strptime(
            run_dir.name.split("__", 1)[0], "%Y%m%dT%H%M%SZ"
        ).replace(tzinfo=timezone.utc).isoformat()
    except ValueError as error:
        raise RawAuditError(f"run id 缺少 UTC timestamp: {run_dir.name}") from error
    summary = {
        "schema_version": "worldsim_v5_m1_raw_extraction_summary_v1",
        "task_id": "WS-V5-M1-STRUCTURED-OWNERSHIP-01",
        "stage": "development_raw_extraction",
        "status": "done",
        "run_id": run_dir.name,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        **evidence,
        "checkpoint": "N/A_data_preparation",
        "source_commit": (run_dir / "source_commit.txt").read_text().strip(),
        "audit_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "sensor_payload_decoded_for_quality": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "training_started": False,
        "model_inference_started": False,
        "parameter_search": False,
    }
    atomic_json(run_dir / "summary.json", summary)
    fingerprint = {
        "summary_sha256": sha256_file(run_dir / "summary.json"),
        "resolved_config_sha256": sha256_file(run_dir / "resolved_config.yaml"),
        "extraction_inputs_sha256": {
            filename: digest
            for digest, filename in (
                line.strip().split(maxsplit=1)
                for line in (run_dir / "input_sha256.txt").read_text().splitlines()
                if line.strip()
            )
        },
        "auditor_sha256": sha256_file(Path(__file__)),
        "raw_batch_manifest_sha256": evidence["raw_batch_manifest_sha256"],
        "member_shard_index_sha256": evidence["member_shard_index_sha256"],
    }
    atomic_json(run_dir / "fingerprint.json", fingerprint)
    with (run_dir / "events.jsonl").open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "at_utc": finished_at,
                    "event": "development_raw_extraction_audited",
                    "status": "done",
                    "scene_count": evidence["scene_count"],
                    "required_count": evidence["required_count"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": summary["task_id"],
            "stage": summary["stage"],
            "status": "done",
            "finished_at_utc": finished_at,
            "summary_sha256": fingerprint["summary_sha256"],
            "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        },
    )
    artifacts = {
        str(path.relative_to(run_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    atomic_json(
        run_dir / "manifest.json",
        {
            "schema_version": "worldsim_v5_m1_raw_extraction_run_manifest_v1",
            "task_id": summary["task_id"],
            "status": "done",
            "artifacts": artifacts,
            "sensor_payload_decoded_for_quality": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--raw-batch-manifest",
        type=Path,
        default=Path(
            "/root/autodl-tmp/data/worldsim_v5/manifests/"
            "m1_development_raw_batch_v1.json"
        ),
    )
    args = parser.parse_args()
    summary = finalize(args.run_dir, args.raw_batch_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

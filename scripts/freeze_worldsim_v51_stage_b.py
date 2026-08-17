#!/usr/bin/env python3
"""冻结 V5.1 Stage B 授权、240 张输入图与 8 个 base checkpoint 身份。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

from PIL import Image
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import (
    ProtocolError,
    V51_BRANCH,
    load_yaml,
    sha256_file,
    validate_stage_b_authorization,
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), *args], text=True
    ).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _chain_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _proposal_scenes(proposal: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    scenes = []
    for role in ("historical_diagnostic", "screening", "development_confirmation"):
        scenes.extend((role, scene) for scene in proposal["roles"][role]["scenes"])
    return scenes


def audit_images(
    authorization: dict[str, Any], proposal: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    identity = authorization["image_identity"]
    root = Path(identity["processed_root"])
    frames = sorted(
        proposal["view_contract"]["uplift_evidence_frames"]
        + proposal["view_contract"]["heldout_evaluation_frames"]
    )
    cameras = list(proposal["view_contract"]["cameras"])
    records: list[dict[str, Any]] = []
    scene_metrics: list[dict[str, Any]] = []
    for role, scene in _proposal_scenes(proposal):
        scene_records = []
        for frame in frames:
            for camera in cameras:
                filename = identity["filename_template"].format(
                    frame=int(frame), camera=int(camera)
                )
                path = root / str(scene["index"]) / "images" / filename
                if not path.is_file():
                    raise ProtocolError(f"Stage B image 缺失: {path}")
                with Image.open(path) as image:
                    width, height = image.size
                    image.verify()
                if [width, height] != [
                    int(identity["expected_width"]),
                    int(identity["expected_height"]),
                ]:
                    raise ProtocolError(f"Stage B image 尺寸漂移: {path}")
                record = {
                    "role": role,
                    "scene": scene["scene"],
                    "scene_index": int(scene["index"]),
                    "frame": int(frame),
                    "camera": int(camera),
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "width": width,
                    "height": height,
                    "sha256": sha256_file(path),
                }
                records.append(record)
                scene_records.append(record)
        scene_bytes = sum(int(record["bytes"]) for record in scene_records)
        if len(scene_records) != int(identity["expected_images_per_scene"]):
            raise ProtocolError(f"Stage B scene image count 漂移: {scene['scene']}")
        if scene_bytes != int(scene["image_bytes"]):
            raise ProtocolError(f"Stage B scene image bytes 漂移: {scene['scene']}")
        scene_metrics.append(
            {
                "metric": "stage_b_image_identity",
                "role": role,
                "scene": scene["scene"],
                "image_count": len(scene_records),
                "image_bytes": scene_bytes,
                "record_chain_sha256": _chain_sha256(scene_records),
            }
        )
    if len(records) != int(identity["expected_total_images"]):
        raise ProtocolError("Stage B total image count 漂移")
    if sum(int(record["bytes"]) for record in records) != int(
        identity["expected_total_bytes"]
    ):
        raise ProtocolError("Stage B total image bytes 漂移")
    return records, scene_metrics


def audit_checkpoints(
    authorization: dict[str, Any], proposal: dict[str, Any]
) -> list[dict[str, Any]]:
    batch_path = PROJECT / authorization["bindings"]["v5_formal_batch"]["path"]
    batch = load_yaml(batch_path)
    runs = {item["scene"]: item for item in batch["runs"]}
    records = []
    for role, scene in _proposal_scenes(proposal):
        run = runs.get(scene["scene"])
        if run is None or int(run["scene_index"]) != int(scene["index"]):
            raise ProtocolError(f"Stage B base run identity 漂移: {scene['scene']}")
        run_dir = Path(run["path"])
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            raise ProtocolError(f"Stage B base summary 缺失: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = summary.get("checkpoint", {})
        checkpoint_path = Path(str(checkpoint.get("path", "")))
        if summary.get("status") != "done" or summary.get("scene") != scene["scene"]:
            raise ProtocolError(f"Stage B base summary terminal 漂移: {scene['scene']}")
        if summary.get("validation_quality_read") is not False:
            raise ProtocolError(f"Stage B base validation quality 越界: {scene['scene']}")
        if summary.get("test_quality_read") is not False:
            raise ProtocolError(f"Stage B base test quality 越界: {scene['scene']}")
        observed_counts = checkpoint.get("gaussian_counts", {})
        expected_counts = {
            "Background": int(scene["background_gaussians"]),
            "RigidNodes": int(scene["rigid_gaussians"]),
        }
        if observed_counts != expected_counts:
            raise ProtocolError(f"Stage B Gaussian count 漂移: {scene['scene']}")
        if not checkpoint_path.is_file():
            raise ProtocolError(f"Stage B base checkpoint 缺失: {checkpoint_path}")
        observed_sha = sha256_file(checkpoint_path)
        if observed_sha != scene["checkpoint_sha256"]:
            raise ProtocolError(f"Stage B base checkpoint SHA 漂移: {scene['scene']}")
        records.append(
            {
                "role": role,
                "scene": scene["scene"],
                "scene_index": int(scene["index"]),
                "run_id": run["run_id"],
                "run_path": str(run_dir),
                "summary_sha256": sha256_file(summary_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": observed_sha,
                "gaussian_counts": observed_counts,
            }
        )
    return records


def _inventory(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "status.json"}:
            continue
        records.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def run(authorization_path: Path, run_dir: Path) -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--short")
    if branch != V51_BRANCH:
        raise ProtocolError(f"必须在 {V51_BRANCH} 执行，当前为 {branch}")
    if status:
        raise ProtocolError("Stage B formal freeze 要求 clean worktree")

    authorization = load_yaml(authorization_path)
    report = validate_stage_b_authorization(PROJECT, authorization)
    proposal_path = PROJECT / authorization["bindings"]["freeze_proposal"]["path"]
    proposal = load_yaml(proposal_path)
    _write_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(
            {"authorization": authorization, "freeze_proposal": proposal},
            allow_unicode=True,
            sort_keys=False,
        ),
    )

    image_records, scene_metrics = audit_images(authorization, proposal)
    checkpoint_records = audit_checkpoints(authorization, proposal)
    image_manifest = {
        "schema_version": "worldsim_v51_stage_b_image_manifest_v1",
        "task_id": authorization["task_id"],
        "record_count": len(image_records),
        "total_bytes": sum(int(record["bytes"]) for record in image_records),
        "record_chain_sha256": _chain_sha256(image_records),
        "records": image_records,
    }
    checkpoint_manifest = {
        "schema_version": "worldsim_v51_stage_b_checkpoint_manifest_v1",
        "task_id": authorization["task_id"],
        "record_count": len(checkpoint_records),
        "record_chain_sha256": _chain_sha256(checkpoint_records),
        "records": checkpoint_records,
    }
    _write_json(run_dir / "artifacts/image_manifest.json", image_manifest)
    _write_json(run_dir / "artifacts/checkpoint_manifest.json", checkpoint_manifest)
    metrics = scene_metrics + [
        {
            "metric": "stage_b_freeze_total",
            "image_count": len(image_records),
            "image_bytes": image_manifest["total_bytes"],
            "checkpoint_count": len(checkpoint_records),
            "image_record_chain_sha256": image_manifest["record_chain_sha256"],
            "checkpoint_record_chain_sha256": checkpoint_manifest[
                "record_chain_sha256"
            ],
        }
    ]
    _write_jsonl(run_dir / "metrics.jsonl", metrics)
    summary = {
        "schema_version": "worldsim_v51_stage_b_freeze_summary_v1",
        "task_id": authorization["task_id"],
        "status": "done",
        "conclusion": "stage_b_authorized_u2_b3_fallback_and_input_identity_frozen",
        "source_commit": head,
        "source_branch": branch,
        "worktree_clean": True,
        "authorization_config_sha256": sha256_file(authorization_path),
        "authorization_report": report,
        "image_manifest_sha256": sha256_file(
            run_dir / "artifacts/image_manifest.json"
        ),
        "image_record_chain_sha256": image_manifest["record_chain_sha256"],
        "image_count": len(image_records),
        "image_bytes": image_manifest["total_bytes"],
        "checkpoint_manifest_sha256": sha256_file(
            run_dir / "artifacts/checkpoint_manifest.json"
        ),
        "checkpoint_record_chain_sha256": checkpoint_manifest[
            "record_chain_sha256"
        ],
        "checkpoint_count": len(checkpoint_records),
        "checkpoint_download_started": False,
        "model_inference_started": False,
        "feature_extraction_started": False,
        "quality_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "m2_status": "pending",
        "m3_status": "pending",
        "failure_ledger_refs": authorization["failure_ledger_refs"],
        "failure_ledger_delta": authorization["failure_ledger_delta"],
        "created_at_utc": _utc_now(),
    }
    _write_json(run_dir / "summary.json", summary)
    fingerprint = {
        "schema_version": "worldsim_v51_stage_b_freeze_fingerprint_v1",
        "task_id": authorization["task_id"],
        "source_commit": head,
        "source_branch": branch,
        "authorization_config": {
            "path": str(authorization_path),
            "sha256": summary["authorization_config_sha256"],
        },
        "bindings": report["binding_sha256"],
        "image_manifest_sha256": summary["image_manifest_sha256"],
        "checkpoint_manifest_sha256": summary["checkpoint_manifest_sha256"],
    }
    _write_json(run_dir / "fingerprint.json", fingerprint)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authorization",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_b_authorization_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    try:
        summary = run(args.authorization.resolve(), run_dir)
        events.append({"event": "run_done", "at_utc": _utc_now()})
        _write_jsonl(run_dir / "events.jsonl", events)
        manifest = {
            "schema_version": "worldsim_v51_stage_b_freeze_manifest_v1",
            "task_id": summary["task_id"],
            "status": "done",
            "inventory": _inventory(run_dir),
        }
        _write_json(run_dir / "manifest.json", manifest)
        status = {
            "schema_version": "worldsim_v51_stage_b_freeze_status_v1",
            "task_id": summary["task_id"],
            "status": "done",
            "source_commit": summary["source_commit"],
            "summary_sha256": sha256_file(run_dir / "summary.json"),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "finished_at_utc": _utc_now(),
        }
        _write_json(run_dir / "status.json", status)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    except Exception as error:
        events.append(
            {
                "event": "run_blocked",
                "at_utc": _utc_now(),
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_stage_b_freeze_status_v1",
                "task_id": "WS-V51-M1-B-LUDVIG-UPLIFT-01",
                "status": "blocked",
                "reason": f"{type(error).__name__}: {error}",
                "finished_at_utc": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()

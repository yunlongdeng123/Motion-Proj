"""V5.2.1 基座资产与 quality-blind cohort 审计。"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import (
    ProtocolError,
    canonical_unit_key,
    partition_for_unit,
    sha256_file,
    validate_partition_invariance,
)


ASSET_STATES = frozenset(
    {
        "PRESENT_EXACT",
        "PRESENT_HASH_MISMATCH",
        "MISSING_BUT_MANIFESTED",
        "MISSING_UNRECOVERABLE",
        "PROTOCOL_MISMATCH",
    }
)


class AssetAuditError(ProtocolError):
    """资产不满足 P1 冻结合同。"""


def audit_file(spec: Mapping[str, Any], *, hash_content: bool = True) -> dict[str, Any]:
    path = Path(str(spec["path"]))
    expected_bytes = int(spec["bytes"])
    expected_sha = spec.get("sha256")
    row: dict[str, Any] = {
        "path": str(path),
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_sha,
        "present": path.is_file(),
    }
    if not path.is_file():
        row.update({"state": "MISSING_BUT_MANIFESTED", "observed_bytes": None, "observed_sha256": None})
        return row
    observed_bytes = path.stat().st_size
    observed_sha = sha256_file(path) if hash_content else None
    row.update({"observed_bytes": observed_bytes, "observed_sha256": observed_sha})
    bytes_match = observed_bytes == expected_bytes
    sha_match = expected_sha is None or observed_sha == expected_sha
    row["state"] = "PRESENT_EXACT" if bytes_match and sha_match else "PRESENT_HASH_MISMATCH"
    return row


def audit_bundle(files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = {name: audit_file(spec) for name, spec in sorted(files.items())}
    states = {row["state"] for row in rows.values()}
    if "PRESENT_HASH_MISMATCH" in states:
        state = "PRESENT_HASH_MISMATCH"
    elif states == {"PRESENT_EXACT"}:
        state = "PRESENT_EXACT"
    elif "MISSING_BUT_MANIFESTED" in states:
        state = "MISSING_BUT_MANIFESTED"
    else:
        raise AssetAuditError(f"未知 bundle 状态：{states}")
    return {"state": state, "files": rows}


def ensure_matched_asset(row: Mapping[str, Any]) -> None:
    if row.get("state") != "PRESENT_EXACT":
        raise AssetAuditError(f"matched cohort 禁止 fallback：{row.get('state')}")
    if row.get("protocol") == "stride10":
        raise AssetAuditError("old stride=10 run 不得进入 matched cohort")


_IMAGE_RE = re.compile(r"^(?P<frame>\d{3})_(?P<camera>\d+)\.jpg$")


def enumerate_quality_blind_targets(
    *,
    dataset: str,
    scene: str,
    scene_index: int,
    scene_root: str | Path,
    expected_frames: int,
    cameras: Sequence[int],
    eligible_bases: Sequence[str],
    development_remainder: int = 2,
    modulus: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(scene_root)
    images = root / "images"
    if not images.is_dir():
        raise AssetAuditError(f"images 目录缺失：{images}")
    sample_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    for frame in range(expected_frames):
        if frame % modulus != development_remainder:
            continue
        unit_key = canonical_unit_key(dataset, scene, canonical_sample_index=frame)
        partition = partition_for_unit(unit_key, modulus=modulus, confirmation_bucket=0)
        targets = []
        for camera in cameras:
            path = images / f"{frame:03d}_{int(camera)}.jpg"
            if not path.is_file() or _IMAGE_RE.match(path.name) is None:
                raise AssetAuditError(f"target 缺失或命名非法：{path}")
            target = {
                "camera": int(camera),
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            targets.append(target)
            view_rows.append(
                {
                    "dataset": dataset,
                    "scene": scene,
                    "scene_index": int(scene_index),
                    "frame": frame,
                    "sample_token": None,
                    "canonical_sample_index": frame,
                    **partition,
                    "camera": int(camera),
                    "target_path": target["path"],
                    "target_bytes": target["bytes"],
                    "target_sha256": target["sha256"],
                    "eligible_bases": sorted(eligible_bases),
                    "metric_resolution": [800, 450],
                    "quality_decoded": False,
                }
            )
        sample_rows.append(
            {
                "dataset": dataset,
                "scene": scene,
                "scene_index": int(scene_index),
                "frame": frame,
                "sample_token": None,
                "canonical_sample_index": frame,
                **partition,
                "eligible_bases": sorted(eligible_bases),
                "target_camera_count": len(targets),
                "target_manifest_sha256": sha256_file_manifest(targets),
                "quality_decoded": False,
            }
        )
    validate_partition_invariance(sample_rows)
    return sample_rows, view_rows


def sha256_file_manifest(rows: Sequence[Mapping[str, Any]]) -> str:
    import hashlib

    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze_summary(sample_rows: Sequence[Mapping[str, Any]], view_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    partition_samples = Counter(str(row["partition"]) for row in sample_rows)
    partition_views = Counter(str(row["partition"]) for row in view_rows)
    per_scene: dict[str, dict[str, Any]] = {}
    for scene in sorted({str(row["scene"]) for row in sample_rows}):
        samples = [row for row in sample_rows if row["scene"] == scene]
        views = [row for row in view_rows if row["scene"] == scene]
        per_scene[scene] = {
            "samples": len(samples),
            "views": len(views),
            "partition_samples": dict(sorted(Counter(str(row["partition"]) for row in samples).items())),
            "partition_views": dict(sorted(Counter(str(row["partition"]) for row in views).items())),
        }
    return {
        "candidate_samples": len(sample_rows),
        "candidate_views": len(view_rows),
        "partition_samples": dict(sorted(partition_samples.items())),
        "partition_views": dict(sorted(partition_views.items())),
        "per_scene": per_scene,
        "quality_bytes_decoded": 0,
    }

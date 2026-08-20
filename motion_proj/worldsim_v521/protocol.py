"""V5.2.1 全阶段共享的协议锁与确定性工具。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ProtocolError(ValueError):
    """输入违反 V5.2.1 冻结协议。"""


ALLOWED_QUALITY_ROLES = frozenset({"historical", "discovery", "confirmation"})
FORBIDDEN_QUALITY_ROLES = frozenset(
    {
        "validation",
        "test",
        "fresh_validation",
        "fresh_test",
        "kitti",
        "kitti_method_tuning",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def atomic_text(path: str | Path, value: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.partial.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, target)


def atomic_json(path: str | Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def atomic_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False)
        for row in rows
    ]
    atomic_text(path, "\n".join(lines) + ("\n" if lines else ""))


def canonical_unit_key(
    dataset: str,
    scene: str,
    *,
    sample_token: str | None = None,
    canonical_sample_index: int | None = None,
) -> str:
    dataset_value = str(dataset).strip().lower()
    scene_value = str(scene).strip()
    token_value = None if sample_token is None else str(sample_token).strip()
    if not dataset_value or not scene_value:
        raise ProtocolError("dataset/scene 不得为空")
    if token_value:
        sample_value = token_value
    elif canonical_sample_index is not None:
        index = int(canonical_sample_index)
        if index < 0:
            raise ProtocolError("canonical_sample_index 必须 >= 0")
        sample_value = str(index)
    else:
        raise ProtocolError("sample_token 与 canonical_sample_index 至少提供一个")
    return f"{dataset_value}|{scene_value}|{sample_value}"


def partition_for_unit(unit_key: str, modulus: int = 5, confirmation_bucket: int = 0) -> dict[str, Any]:
    if modulus <= 1:
        raise ProtocolError("partition modulus 必须 > 1")
    if confirmation_bucket < 0 or confirmation_bucket >= modulus:
        raise ProtocolError("confirmation_bucket 越界")
    digest = hashlib.sha256(unit_key.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % modulus
    return {
        "unit_key": unit_key,
        "split_hash": digest,
        "bucket": bucket,
        "partition": "confirmation" if bucket == confirmation_bucket else "discovery",
    }


def validate_partition_invariance(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: dict[str, tuple[str, str]] = {}
    for row in rows:
        unit_key = str(row["unit_key"])
        current = (str(row["partition"]), str(row["split_hash"]))
        previous = seen.setdefault(unit_key, current)
        if previous != current:
            raise ProtocolError(f"同一 canonical sample 跨 partition：{unit_key}")


def temporal_window_partition(member_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not member_rows:
        return {"status": "undefined_empty_window", "partition": None}
    values = {str(row["partition"]) for row in member_rows}
    if len(values) != 1:
        return {"status": "undefined_cross_partition", "partition": None}
    return {"status": "defined", "partition": next(iter(values))}


def validate_quality_read(*, split_role: str, paths: Sequence[str | Path] = ()) -> None:
    role = str(split_role).strip().lower()
    if role in FORBIDDEN_QUALITY_ROLES or role not in ALLOWED_QUALITY_ROLES:
        raise ProtocolError(f"quality-read role 被锁定：{split_role}")
    lowered = [str(path).replace("\\", "/").lower() for path in paths]
    forbidden_markers = (
        "/fresh_validation/",
        "/fresh-validation/",
        "/fresh_test/",
        "/fresh-test/",
        "/kitti/",
        "/kitti_",
    )
    hit = next((path for path in lowered if any(marker in path for marker in forbidden_markers)), None)
    if hit is not None:
        raise ProtocolError(f"quality path 命中永久锁：{hit}")


def inventory_files(root: str | Path) -> list[dict[str, Any]]:
    base = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(base).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows

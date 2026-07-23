"""V7.1 WorldState 与 RenderRequest 的规范序列化。"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

CANONICALIZATION_VERSION = "v71-canonical-json-v1"
FLOAT_PREFIX = "@f64:"


class CanonicalizationError(ValueError):
    """输入无法形成唯一规范表示。"""


def canonicalize_quaternion_wxyz(
    quaternion: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> list[float]:
    if len(quaternion) != 4:
        raise CanonicalizationError("quaternion 必须是 wxyz 四元组")
    values = [float(value) for value in quaternion]
    if not all(math.isfinite(value) for value in values):
        raise CanonicalizationError("quaternion 禁止 NaN/Inf")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= tolerance:
        raise CanonicalizationError("近零 quaternion 无法规范化")
    values = [value / norm for value in values]
    for value in values:
        if abs(value) > tolerance:
            if value < 0:
                values = [-item for item in values]
            break
    return [0.0 if value == 0.0 else value for value in values]


def _float_token(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise CanonicalizationError("浮点禁止 NaN/Inf")
    if value == 0.0:
        value = 0.0
    return FLOAT_PREFIX + struct.pack(">d", value).hex()


def canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return _float_token(value)
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("object key 必须是字符串")
            normalized = unicodedata.normalize("NFC", key)
            if normalized in seen:
                raise CanonicalizationError("Unicode NFC 后出现重复 key")
            seen.add(normalized)
            rows.append((normalized, canonicalize(item)))
        return {key: item for key, item in sorted(rows, key=lambda row: row[0])}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [canonicalize(item) for item in value]
    raise CanonicalizationError(f"不支持的规范类型: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def content_reference(*, sha256: str, shape: Sequence[int], dtype: str, semantic_version: str) -> dict:
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise CanonicalizationError("content reference 需要小写 SHA256")
    return {
        "sha256": sha256,
        "shape": [int(item) for item in shape],
        "dtype": str(dtype),
        "semantic_version": str(semantic_version),
    }

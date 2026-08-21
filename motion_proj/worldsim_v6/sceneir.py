"""WorldSim V6 SceneIR v0 的确定性封装、校验与加载。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "worldsim.sceneir.v0"
MANIFEST_VERSION = "worldsim.sceneir.package_manifest.v0"
PROVENANCE_KINDS = {"observed", "reconstructed", "generated", "unknown"}
FRAME_TYPES = {"world", "ego", "sensor", "actor"}
CHUNK_ROLES = {"static", "actor"}
REQUIRED_GAUSSIAN_ARRAYS = {
    "means_m",
    "scales_m",
    "quaternions_wxyz",
    "opacities",
    "features_dc",
    "features_rest",
    "source_indices",
}


class SceneIRError(RuntimeError):
    """SceneIR 合同、完整性或确定性校验失败。"""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SceneIRError(message)


def _unique_ids(rows: list[Mapping[str, Any]], kind: str) -> set[str]:
    identifiers = [row.get("id") for row in rows]
    _require(all(isinstance(item, str) and item for item in identifiers), f"{kind} id 非法")
    _require(len(set(identifiers)) == len(identifiers), f"{kind} id 重复")
    return set(identifiers)


def _validate_document(document: Mapping[str, Any], arrays: Mapping[str, np.ndarray] | None) -> None:
    """验证 SceneIR 文档；加载阶段同时验证数组语义。"""
    _require(document.get("schema_version") == SCHEMA_VERSION, "SceneIR schema_version 漂移")
    _require(_is_sha256(document.get("content_sha256")), "SceneIR content_sha256 非法")

    coordinate = document.get("coordinate_system")
    _require(isinstance(coordinate, Mapping), "缺少 coordinate_system")
    _require(coordinate.get("handedness") in {"right"}, "v0 只接受右手坐标系")
    _require(coordinate.get("length_unit") == "meter", "v0 长度单位必须是 meter")
    axes = coordinate.get("axes")
    _require(
        isinstance(axes, Mapping) and set(axes) == {"x", "y", "z"}
        and all(isinstance(value, str) and value for value in axes.values()),
        "coordinate_system.axes 必须显式声明 x/y/z",
    )
    _require(coordinate.get("transform_convention") == "T_dst_src", "变换命名必须为 T_dst_src")

    frames = document.get("frames")
    _require(isinstance(frames, list) and frames, "frames 不能为空")
    frame_ids = _unique_ids(frames, "frame")
    frame_types = {row.get("id"): row.get("frame_type") for row in frames}
    _require(all(kind in FRAME_TYPES for kind in frame_types.values()), "frame_type 非法")
    _require(sum(kind == "world" for kind in frame_types.values()) == 1, "必须恰有一个 world frame")

    episode = document.get("episode")
    _require(isinstance(episode, Mapping), "缺少 episode")
    _require(isinstance(episode.get("seed"), int) and episode["seed"] >= 0, "episode seed 非法")
    start_us = episode.get("start_timestamp_us")
    end_us = episode.get("end_timestamp_us")
    _require(isinstance(start_us, int) and isinstance(end_us, int) and 0 <= start_us <= end_us, "episode 时间非法")

    transforms = document.get("transforms")
    _require(isinstance(transforms, list), "transforms 必须是列表")
    transform_keys: set[tuple[str, int]] = set()
    for transform in transforms:
        src = transform.get("src_frame")
        dst = transform.get("dst_frame")
        timestamp_us = transform.get("timestamp_us")
        _require(src in frame_ids and dst in frame_ids, "transform 引用了未知 frame")
        _require(transform.get("name") == f"T_{dst}_{src}", "transform 名称与 T_dst_src 不一致")
        _require(isinstance(timestamp_us, int) and start_us <= timestamp_us <= end_us, "transform 时间越界")
        key = (transform["name"], timestamp_us)
        _require(key not in transform_keys, "同名同时间 transform 重复")
        transform_keys.add(key)
        translation = transform.get("translation_m")
        rotation = transform.get("rotation_wxyz")
        _require(isinstance(translation, list) and len(translation) == 3, "translation_m 非法")
        _require(isinstance(rotation, list) and len(rotation) == 4, "rotation_wxyz 非法")
        _require(all(isinstance(value, (int, float)) and math.isfinite(value) for value in translation + rotation), "transform 含非有限值")
        norm = math.sqrt(sum(float(value) ** 2 for value in rotation))
        _require(abs(norm - 1.0) <= 1e-5, "transform quaternion 未归一化")

    provenance = document.get("provenance")
    _require(isinstance(provenance, list) and provenance, "provenance 不能为空")
    provenance_ids = _unique_ids(provenance, "provenance")
    for row in provenance:
        _require(row.get("kind") in PROVENANCE_KINDS, "provenance kind 非法")
        _require(_is_sha256(row.get("source_sha256")), "provenance 缺少 source_sha256")
        _require(isinstance(row.get("reconstructor_version"), str) and row["reconstructor_version"], "缺少 reconstructor_version")

    supports = document.get("support")
    _require(isinstance(supports, list) and supports, "support 不能为空")
    support_ids = _unique_ids(supports, "support")
    for row in supports:
        _require(isinstance(row.get("observed_timestamp_us"), list), "support 时间必须是列表")
        _require(all(isinstance(value, int) and start_us <= value <= end_us for value in row["observed_timestamp_us"]), "support 时间越界")

    validity = document.get("validity")
    _require(isinstance(validity, Mapping), "缺少 validity")
    _require("score" not in validity and "validity_score" not in validity, "禁止单一 validity 分数")
    for key in ("q_photo", "q_geometry", "q_semantic", "q_dynamics"):
        value = validity.get(key)
        _require(isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0, f"{key} 非法")

    chunks = document.get("chunks")
    _require(isinstance(chunks, list) and chunks, "chunks 不能为空")
    chunk_ids = _unique_ids(chunks, "chunk")
    actor_rows = document.get("actors")
    _require(isinstance(actor_rows, list), "actors 必须是列表")
    actor_ids = _unique_ids(actor_rows, "actor")
    for actor in actor_rows:
        frame_id = actor.get("canonical_frame")
        _require(frame_id in frame_ids and frame_types[frame_id] == "actor", "actor canonical_frame 非法")
        actor_chunks = actor.get("chunk_ids")
        _require(isinstance(actor_chunks, list) and set(actor_chunks).issubset(chunk_ids), "actor chunk 引用非法")
        trajectory = actor.get("trajectory")
        visibility = actor.get("visibility")
        _require(isinstance(trajectory, list) and trajectory, "actor trajectory 不能为空")
        _require(isinstance(visibility, list) and len(visibility) == len(trajectory), "actor visibility 与 trajectory 不对齐")
        for key in trajectory:
            _require(isinstance(key, Mapping) and (key.get("transform_name"), key.get("timestamp_us")) in transform_keys, "actor trajectory 引用未知 transform")

    static_world = document.get("static_world")
    _require(isinstance(static_world, Mapping), "缺少 static_world")
    static_ids = static_world.get("chunk_ids")
    _require(isinstance(static_ids, list) and set(static_ids).issubset(chunk_ids), "static_world chunk 引用非法")
    _require("surfaces" in static_world and "collision_proxy" in static_world and "map_binding" in static_world, "static_world 合同不完整")

    sensors = document.get("sensors")
    _require(isinstance(sensors, list), "sensors 必须是列表")
    _unique_ids(sensors, "sensor")
    for sensor in sensors:
        frame_id = sensor.get("frame_id")
        _require(frame_id in frame_ids and frame_types[frame_id] == "sensor", "sensor frame_id 非法")
        kind = sensor.get("sensor_type")
        _require(kind in {"camera", "lidar"}, "sensor_type 非法")
        if kind == "camera":
            _require(sensor.get("camera_model") in {"pinhole", "fisheye", "equirectangular"}, "camera_model 非法")
            _require(isinstance(sensor.get("calibration"), Mapping), "camera calibration 缺失")

    for chunk in chunks:
        role = chunk.get("role")
        frame_id = chunk.get("frame_id")
        actor_id = chunk.get("actor_id")
        _require(role in CHUNK_ROLES, "chunk role 非法")
        _require(frame_id in frame_ids, "chunk frame_id 非法")
        _require(chunk.get("primitive_type") == "gaussian_splat", "v0 只接受 gaussian_splat")
        _require(isinstance(chunk.get("primitive_count"), int) and chunk["primitive_count"] >= 0, "primitive_count 非法")
        _require(chunk.get("provenance_id") in provenance_ids and chunk.get("support_id") in support_ids, "chunk 证据引用非法")
        _require(_is_sha256(chunk.get("content_sha256")), "chunk content_sha256 非法")
        if role == "static":
            _require(actor_id is None and frame_types[frame_id] == "world", "static chunk 必须位于 world frame")
            _require(chunk["id"] in static_ids, "static chunk 未被 static_world 引用")
        else:
            _require(actor_id in actor_ids and frame_types[frame_id] == "actor", "actor chunk 归属非法")
            owner = next(row for row in actor_rows if row["id"] == actor_id)
            _require(chunk["id"] in owner["chunk_ids"] and owner["canonical_frame"] == frame_id, "actor chunk 与 actor 记录不一致")
        refs = chunk.get("arrays")
        _require(isinstance(refs, Mapping) and REQUIRED_GAUSSIAN_ARRAYS.issubset(refs), "Gaussian chunk 数组不完整")
        for name, ref in refs.items():
            _require(isinstance(name, str) and isinstance(ref, Mapping), "blob ref 非法")
            _require(_is_sha256(ref.get("sha256")), "blob sha256 非法")
            _require(ref.get("path") == f"blobs/{ref['sha256']}.npy", "blob path 非规范")
            _require(isinstance(ref.get("shape"), list) and isinstance(ref.get("dtype"), str), "blob 元数据不完整")
            if arrays is not None:
                array = arrays.get(ref["sha256"])
                _require(array is not None, "blob 未加载")
                _require(list(array.shape) == ref["shape"] and array.dtype.str == ref["dtype"], "blob shape/dtype 漂移")
                _require(array.ndim >= 1 and array.shape[0] == chunk["primitive_count"], "blob 第一维与 primitive_count 不一致")
                _require(np.isfinite(array).all() if np.issubdtype(array.dtype, np.number) else True, "blob 含非有限数")
        if arrays is not None:
            quaternions = arrays[refs["quaternions_wxyz"]["sha256"]]
            _require(quaternions.shape == (chunk["primitive_count"], 4), "Gaussian quaternion shape 非法")
            _require(np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-5), "Gaussian quaternion 未归一化")
            _require((arrays[refs["scales_m"]["sha256"]] > 0).all(), "Gaussian scale 必须为正")
            opacities = arrays[refs["opacities"]["sha256"]]
            _require(((opacities >= 0) & (opacities <= 1)).all(), "Gaussian opacity 越界")


def _normalized_array(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    _require(value.dtype != object, "禁止 object dtype")
    _require(value.ndim >= 1, "SceneIR blob 必须至少一维")
    if value.dtype.byteorder == ">" or (value.dtype.byteorder == "=" and os.sys.byteorder == "big"):
        value = value.astype(value.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(value)


def _document_hash(document: Mapping[str, Any]) -> str:
    value = dict(document)
    value.pop("content_sha256", None)
    return _sha256_bytes(_canonical_json(value))


def write_sceneir(
    target: Path,
    document: Mapping[str, Any],
    chunk_arrays: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    """以内容寻址 blob 和原子目录替换写入一次不可覆盖的 SceneIR package。"""
    target = target.resolve()
    if target.exists():
        raise SceneIRError(f"SceneIR package 已存在：{target}")
    stage = target.with_name(f".{target.name}.partial-{uuid.uuid4().hex}")
    stage.mkdir(parents=True)
    (stage / "blobs").mkdir()
    try:
        result = json.loads(json.dumps(document, ensure_ascii=False))
        chunks = result.get("chunks", [])
        chunk_by_id = {row.get("id"): row for row in chunks}
        _require(set(chunk_arrays) == set(chunk_by_id), "chunk_arrays 与 chunks 不一一对应")
        blob_records: dict[str, dict[str, Any]] = {}
        normalized_blobs: dict[str, np.ndarray] = {}
        for chunk_id in sorted(chunk_arrays):
            chunk = chunk_by_id[chunk_id]
            refs: dict[str, Any] = {}
            chunk_hash_input: list[dict[str, Any]] = []
            for name in sorted(chunk_arrays[chunk_id]):
                array = _normalized_array(chunk_arrays[chunk_id][name])
                temporary = stage / "blobs" / f".{chunk_id}-{name}.partial.npy"
                with temporary.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                digest = _sha256_file(temporary)
                relative = f"blobs/{digest}.npy"
                final_blob = stage / relative
                if final_blob.exists():
                    temporary.unlink()
                else:
                    temporary.replace(final_blob)
                ref = {
                    "path": relative,
                    "sha256": digest,
                    "bytes": final_blob.stat().st_size,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                }
                refs[name] = ref
                blob_records[digest] = ref
                normalized_blobs[digest] = array
                chunk_hash_input.append({"name": name, **ref})
            chunk["arrays"] = refs
            chunk["content_sha256"] = _sha256_bytes(_canonical_json(chunk_hash_input))
        result["content_sha256"] = _document_hash(result)
        _validate_document(result, normalized_blobs)
        sceneir_bytes = _canonical_json(result)
        (stage / "sceneir.json").write_bytes(sceneir_bytes)
        files = {
            "sceneir.json": {
                "bytes": len(sceneir_bytes),
                "sha256": _sha256_bytes(sceneir_bytes),
            }
        }
        for digest, ref in sorted(blob_records.items()):
            files[ref["path"]] = {"bytes": ref["bytes"], "sha256": digest}
        manifest = {
            "schema_version": MANIFEST_VERSION,
            "sceneir_content_sha256": result["content_sha256"],
            "files": files,
        }
        (stage / "MANIFEST.json").write_bytes(_canonical_json(manifest))
        target.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(target)
        return result
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def load_sceneir(package: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fail-closed 加载并验证 package、内容哈希、blob 和语义合同。"""
    package = package.resolve()
    _require(package.is_dir(), f"SceneIR package 不存在：{package}")
    manifest = json.loads((package / "MANIFEST.json").read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == MANIFEST_VERSION, "package manifest schema 漂移")
    files = manifest.get("files")
    _require(isinstance(files, Mapping) and "sceneir.json" in files, "package manifest 文件表非法")
    for relative, record in files.items():
        _require(isinstance(relative, str) and not Path(relative).is_absolute(), "manifest 含绝对路径")
        path = (package / relative).resolve()
        try:
            path.relative_to(package)
        except ValueError as error:
            raise SceneIRError("manifest 路径穿越") from error
        _require(path.is_file(), f"manifest 文件缺失：{relative}")
        _require(path.stat().st_size == record.get("bytes") and _sha256_file(path) == record.get("sha256"), f"manifest 完整性失败：{relative}")
    document = json.loads((package / "sceneir.json").read_text(encoding="utf-8"))
    _require(document.get("content_sha256") == _document_hash(document), "SceneIR content_sha256 不匹配")
    _require(manifest.get("sceneir_content_sha256") == document["content_sha256"], "manifest 与 SceneIR content hash 不一致")
    arrays: dict[str, np.ndarray] = {}
    for chunk in document.get("chunks", []):
        for ref in chunk.get("arrays", {}).values():
            relative = ref.get("path")
            _require(isinstance(relative, str), "blob path 非法")
            path = (package / relative).resolve()
            try:
                path.relative_to(package / "blobs")
            except ValueError as error:
                raise SceneIRError("blob 路径穿越") from error
            _require(relative in files, "blob 未进入 package manifest")
            digest = ref["sha256"]
            if digest not in arrays:
                arrays[digest] = np.load(path, allow_pickle=False)
    _validate_document(document, arrays)
    return document, arrays


def verify_sceneir(package: Path) -> dict[str, Any]:
    """返回适合 fresh-process 证据记录的紧凑验证摘要。"""
    document, arrays = load_sceneir(package)
    return {
        "schema_version": SCHEMA_VERSION,
        "content_sha256": document["content_sha256"],
        "chunk_count": len(document["chunks"]),
        "actor_count": len(document["actors"]),
        "primitive_count": sum(row["primitive_count"] for row in document["chunks"]),
        "unique_blob_count": len(arrays),
    }

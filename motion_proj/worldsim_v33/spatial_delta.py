"""WorldSim V3.3 可维护空间 delta 的 schema、组合与精确回滚。"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
import zipfile

import numpy as np
import torch

from motion_proj.worldsim_v32.actor_asset_schema import SH_C0, validate_actor_asset
from motion_proj.worldsim_v33.instance_field import validate_instance_field
from motion_proj.worldsim_v33.roadpatch import validate_patch_delta


ERASE_SCHEMA_VERSION = "worldsim_v33_erase_delta_v1"
ACTOR_INSERT_SCHEMA_VERSION = "worldsim_v33_actor_insert_delta_v1"
STACK_SCHEMA_VERSION = "worldsim_v33_spatial_delta_stack_v1"
PACKAGE_SCHEMA_VERSION = "worldsim_v33_spatial_delta_package_v1"

MODEL_BACKGROUND = np.int8(0)
MODEL_RIGID = np.int8(1)
MODEL_NAMES = {int(MODEL_BACKGROUND): "Background", int(MODEL_RIGID): "RigidNodes"}
OPERATION_ORDER = ("ERASE", "INSERT_BACKGROUND", "INSERT_ACTOR", "RENDER_ONLY")
OPERATION_RANK = {name: index for index, name in enumerate(OPERATION_ORDER)}

PROVENANCE_GENERATED_ACTOR = np.uint8(3)

BACKGROUND_ATTRS = {
    "_means": "means",
    "_scales": "raw_scales",
    "_quats": "quats",
    "_features_dc": "features_dc",
    "_features_rest": "features_rest",
    "_opacities": "raw_opacities",
}
RIGID_ATTRS = (
    "_means",
    "_scales",
    "_quats",
    "_features_dc",
    "_features_rest",
    "_opacities",
)


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def atomic_save_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """写出时间戳固定、字段排序固定的可复现 NPZ。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        with zipfile.ZipFile(
            handle,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.asarray(arrays[name]), allow_pickle=False
                )
                entry = zipfile.ZipInfo(
                    f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.create_system = 3
                entry.external_attr = 0o600 << 16
                archive.writestr(
                    entry,
                    buffer.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
    os.replace(temporary, target)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def build_erase_delta(
    field: Mapping[str, np.ndarray], *, instance_id: int,
    minimum_background_instance_opacity: float | None = None,
) -> dict[str, np.ndarray]:
    """从 S1 hard assignment 生成稳定的 Background/Rigid 擦除索引。"""

    validate_instance_field(field)
    actor_ids = np.asarray(field["actor_instance_ids"], dtype=np.int32)
    positions = np.flatnonzero(actor_ids == int(instance_id))
    if positions.size != 1:
        raise ValueError(f"instance_id {instance_id} 未唯一出现在 S1 identity mapping")
    token = str(np.asarray(field["actor_tokens"])[int(positions[0])])
    hard_selected = (
        np.asarray(field["hard_instance_id"], dtype=np.int32) == int(instance_id)
    )
    base_model = np.asarray(field["base_model"], dtype=np.int8)
    instance_opacity = np.asarray(field["instance_opacity"], dtype=np.float32)
    if minimum_background_instance_opacity is None:
        selected_mask = hard_selected
        threshold = -1.0
        selection_policy = "hard_assignment_all"
    else:
        threshold = float(minimum_background_instance_opacity)
        if not 0.0 < threshold < 1.0:
            raise ValueError("Background instance opacity threshold 必须在 (0,1)")
        # Rigid core 是 registry 的物理对象行；Background 只接受 S1 学到的 MAP 正类，
        # 避免把低后验的候选背景整片硬擦除。
        selected_mask = hard_selected & (
            (base_model == MODEL_RIGID)
            | ((base_model == MODEL_BACKGROUND) & (instance_opacity >= threshold))
        )
        selection_policy = "rigid_core_plus_background_probability_ge"
    selected = np.flatnonzero(selected_mask)
    if selected.size == 0:
        raise ValueError(f"instance_id {instance_id} 没有可擦除 Gaussian")
    model_code = base_model[selected]
    source_indices = np.asarray(field["base_index"], dtype=np.int64)[selected]
    gaussian_ids = np.asarray(field["gaussian_id"], dtype=np.int64)[selected]
    order = np.lexsort((gaussian_ids, source_indices, model_code))
    selector = {
        "model_code": model_code[order],
        "source_flat_indices": source_indices[order],
        "gaussian_ids": gaussian_ids[order],
        "selection_score": instance_opacity[selected][order],
    }
    mask_hash = sha256_arrays(selector)
    delta = {
        "schema_version": np.asarray(ERASE_SCHEMA_VERSION, dtype="<U64"),
        "instance_id": np.asarray(int(instance_id), dtype=np.int32),
        "instance_token": np.asarray(token, dtype="<U64"),
        "mask_hash": np.asarray(mask_hash, dtype="<U64"),
        "selection_policy": np.asarray(selection_policy, dtype="<U64"),
        "minimum_background_instance_opacity": np.asarray(
            threshold, dtype=np.float32
        ),
        **selector,
    }
    validate_erase_delta(delta)
    return delta


def validate_erase_delta(
    delta: Mapping[str, np.ndarray], *, model_counts: Mapping[str, int] | None = None
) -> None:
    required = {
        "schema_version",
        "instance_id",
        "instance_token",
        "mask_hash",
        "selection_policy",
        "minimum_background_instance_opacity",
        "model_code",
        "source_flat_indices",
        "gaussian_ids",
        "selection_score",
    }
    missing = required - set(delta)
    if missing:
        raise ValueError(f"ERASE delta 缺字段: {sorted(missing)}")
    if str(np.asarray(delta["schema_version"]).item()) != ERASE_SCHEMA_VERSION:
        raise ValueError("ERASE schema version 漂移")
    model_code = np.asarray(delta["model_code"], dtype=np.int8)
    source_indices = np.asarray(delta["source_flat_indices"], dtype=np.int64)
    gaussian_ids = np.asarray(delta["gaussian_ids"], dtype=np.int64)
    selection_score = np.asarray(delta["selection_score"], dtype=np.float32)
    selection_policy = str(np.asarray(delta["selection_policy"]).item())
    background_threshold = float(
        np.asarray(delta["minimum_background_instance_opacity"]).item()
    )
    count = int(model_code.size)
    if (
        not count
        or source_indices.shape != (count,)
        or gaussian_ids.shape != (count,)
        or selection_score.shape != (count,)
    ):
        raise ValueError("ERASE per-Gaussian shape 不合法")
    if not np.isfinite(selection_score).all() or np.any(
        (selection_score < 0.0) | (selection_score > 1.0)
    ):
        raise ValueError("ERASE selection_score 非法")
    if selection_policy == "hard_assignment_all":
        if background_threshold != -1.0:
            raise ValueError("ERASE hard assignment policy threshold 漂移")
    elif selection_policy == "rigid_core_plus_background_probability_ge":
        if not 0.0 < background_threshold < 1.0:
            raise ValueError("ERASE Background probability threshold 非法")
        background_scores = selection_score[model_code == MODEL_BACKGROUND]
        if np.any(background_scores < background_threshold):
            raise ValueError("ERASE Background 行低于冻结 probability threshold")
    else:
        raise ValueError(f"ERASE selection policy 未知: {selection_policy}")
    if not set(int(value) for value in np.unique(model_code)) <= set(MODEL_NAMES):
        raise ValueError("ERASE model_code 含未知模型")
    if np.any(source_indices < 0) or np.unique(gaussian_ids).size != count:
        raise ValueError("ERASE index 非法或 gaussian_id 重复")
    pairs = np.rec.fromarrays([model_code, source_indices])
    if np.unique(pairs).size != count:
        raise ValueError("ERASE model/source_flat_indices 重复")
    selector = {
        "model_code": model_code,
        "source_flat_indices": source_indices,
        "gaussian_ids": gaussian_ids,
        "selection_score": selection_score,
    }
    if str(np.asarray(delta["mask_hash"]).item()) != sha256_arrays(selector):
        raise ValueError("ERASE mask_hash 不匹配")
    if model_counts is not None:
        for code, name in MODEL_NAMES.items():
            indices = source_indices[model_code == code]
            if indices.size and int(indices.max()) >= int(model_counts[name]):
                raise ValueError(f"ERASE {name} index 越界")


def atomic_save_erase_delta(path: str | Path, delta: Mapping[str, np.ndarray]) -> None:
    validate_erase_delta(delta)
    atomic_save_npz(path, delta)


def load_erase_delta(path: str | Path) -> dict[str, np.ndarray]:
    delta = load_npz(path)
    validate_erase_delta(delta)
    return delta


def build_actor_insert_delta(
    asset: Mapping[str, np.ndarray], *, instance_id: int, instance_token: str,
    rigid_model_index: int,
) -> dict[str, np.ndarray]:
    """把 S3 actor-local 资产封装为含逐行 provenance 的 INSERT_ACTOR。"""

    validate_actor_asset(dict(asset))
    count = int(np.asarray(asset["means"]).shape[0])
    delta = {name: np.asarray(value).copy() for name, value in asset.items()}
    delta.update(
        {
            "schema_version": np.asarray(
                ACTOR_INSERT_SCHEMA_VERSION, dtype="<U64"
            ),
            "instance_id": np.asarray(int(instance_id), dtype=np.int32),
            "instance_token": np.asarray(str(instance_token), dtype="<U64"),
            "rigid_model_index": np.asarray(int(rigid_model_index), dtype=np.int32),
            "provenance_code": np.full(
                count, PROVENANCE_GENERATED_ACTOR, dtype=np.uint8
            ),
            "source_asset_gaussian_index": np.arange(count, dtype=np.int64),
        }
    )
    validate_actor_insert_delta(delta)
    return delta


def validate_actor_insert_delta(delta: Mapping[str, np.ndarray]) -> None:
    required = {
        "schema_version",
        "instance_id",
        "instance_token",
        "rigid_model_index",
        "provenance_code",
        "source_asset_gaussian_index",
    }
    missing = required - set(delta)
    if missing:
        raise ValueError(f"INSERT_ACTOR 缺字段: {sorted(missing)}")
    if str(np.asarray(delta["schema_version"]).item()) != ACTOR_INSERT_SCHEMA_VERSION:
        raise ValueError("INSERT_ACTOR schema version 漂移")
    validate_actor_asset(dict(delta))
    count = int(np.asarray(delta["means"]).shape[0])
    provenance = np.asarray(delta["provenance_code"], dtype=np.uint8)
    source = np.asarray(delta["source_asset_gaussian_index"], dtype=np.int64)
    if provenance.shape != (count,) or not np.all(
        provenance == PROVENANCE_GENERATED_ACTOR
    ):
        raise ValueError("INSERT_ACTOR provenance 非法")
    if source.shape != (count,) or not np.array_equal(
        source, np.arange(count, dtype=np.int64)
    ):
        raise ValueError("INSERT_ACTOR source index 必须连续且唯一")


def atomic_save_actor_insert_delta(
    path: str | Path, delta: Mapping[str, np.ndarray]
) -> None:
    validate_actor_insert_delta(delta)
    atomic_save_npz(path, delta)


def load_actor_insert_delta(path: str | Path) -> dict[str, np.ndarray]:
    delta = load_npz(path)
    validate_actor_insert_delta(delta)
    return delta


def validate_stack_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != STACK_SCHEMA_VERSION:
        raise ValueError("spatial stack schema version 漂移")
    operations = list(manifest.get("operations", []))
    if not operations:
        raise ValueError("spatial stack 至少需要一个 operation")
    ids = [str(operation.get("operation_id", "")) for operation in operations]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("spatial stack operation_id 必须非空且唯一")
    types = [str(operation.get("type", "")) for operation in operations]
    if any(value not in OPERATION_RANK for value in types):
        raise ValueError("spatial stack 含未知 operation type")
    ranks = [OPERATION_RANK[value] for value in types]
    if ranks != sorted(ranks):
        raise ValueError(f"spatial stack 顺序必须为 {OPERATION_ORDER}")
    if len(types) != len(set(types)):
        raise ValueError("spatial stack 不允许重复 operation type")


def ordered_stack_manifest(
    *, stack_id: str, operations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(
        (dict(operation) for operation in operations),
        key=lambda operation: OPERATION_RANK[str(operation["type"])],
    )
    manifest = {
        "schema_version": STACK_SCHEMA_VERSION,
        "stack_id": str(stack_id),
        "composition_order": list(OPERATION_ORDER),
        "operations": ordered,
    }
    validate_stack_manifest(manifest)
    return manifest


def _live_counts(background: Any, rigid: Any) -> dict[str, int]:
    return {
        "Background": int(background._means.shape[0]),
        "RigidNodes": int(rigid._means.shape[0]),
    }


def _parameter(value: torch.Tensor, original: torch.nn.Parameter) -> torch.nn.Parameter:
    return torch.nn.Parameter(value, requires_grad=bool(original.requires_grad))


def _actor_tails(
    rigid: Any, delta: Mapping[str, np.ndarray]
) -> dict[str, torch.Tensor]:
    device = rigid._means.device
    dtype = rigid._means.dtype
    means = torch.as_tensor(delta["means"], device=device, dtype=dtype)
    scales = torch.as_tensor(delta["scales"], device=device, dtype=dtype)
    quats = torch.as_tensor(delta["quats"], device=device, dtype=dtype)
    rgb = torch.as_tensor(delta["rgb"], device=device, dtype=dtype).clamp(
        1e-6, 1.0 - 1e-6
    )
    opacity = torch.as_tensor(
        delta["opacity"], device=device, dtype=dtype
    ).clamp(1e-6, 1.0 - 1e-6)
    if int(rigid.sh_degree) > 0:
        features_dc = (rgb - 0.5) / SH_C0
    else:
        features_dc = torch.logit(rgb)
    features_rest = torch.zeros(
        (int(means.shape[0]),) + tuple(rigid._features_rest.shape[1:]),
        device=device,
        dtype=rigid._features_rest.dtype,
    )
    return {
        "_means": means,
        "_scales": torch.log(scales),
        "_quats": quats,
        "_features_dc": features_dc.to(dtype=rigid._features_dc.dtype),
        "_features_rest": features_rest,
        "_opacities": torch.logit(opacity)[:, None],
    }


def _reference_objects(background: Any, rigid: Any) -> dict[str, Any]:
    objects = {
        **{f"Background.{name}": getattr(background, name) for name in BACKGROUND_ATTRS},
        **{f"RigidNodes.{name}": getattr(rigid, name) for name in RIGID_ATTRS},
        "RigidNodes.point_ids": rigid.point_ids,
    }
    if hasattr(rigid, "instances_size"):
        objects["RigidNodes.instances_size"] = rigid.instances_size
    return objects


def _assert_reference_objects(
    background: Any, rigid: Any, references: Mapping[str, Any]
) -> None:
    current = _reference_objects(background, rigid)
    missing = set(references) - set(current)
    drifted = [name for name in references if current.get(name) is not references[name]]
    if missing or drifted:
        raise RuntimeError(
            f"spatial delta rollback 对象漂移: missing={sorted(missing)} "
            f"drifted={sorted(drifted)}"
        )


@contextmanager
def temporary_spatial_composition(
    models: Mapping[str, Any], *, erase_delta: Mapping[str, np.ndarray],
    background_delta: Mapping[str, np.ndarray] | None = None,
    actor_delta: Mapping[str, np.ndarray] | None = None,
) -> Iterator[dict[str, Any]]:
    """按固定次序装载 delta，退出（含异常路径）时恢复原对象。"""

    background = models["Background"]
    rigid = models["RigidNodes"]
    counts = _live_counts(background, rigid)
    validate_erase_delta(erase_delta, model_counts=counts)
    if background_delta is not None:
        validate_patch_delta(background_delta)
        source_ids = np.asarray(background_delta["source_gaussian_ids"], dtype=np.int64)
        if np.unique(source_ids).size != source_ids.size:
            raise ValueError("INSERT_BACKGROUND source_gaussian_ids 重复")
    if actor_delta is not None:
        validate_actor_insert_delta(actor_delta)
    references = _reference_objects(background, rigid)
    originals = dict(references)
    model_code = np.asarray(erase_delta["model_code"], dtype=np.int8)
    source_indices = np.asarray(
        erase_delta["source_flat_indices"], dtype=np.int64
    )
    erased_by_model = {
        name: source_indices[model_code == code]
        for code, name in MODEL_NAMES.items()
    }
    actor_tails = _actor_tails(rigid, actor_delta) if actor_delta is not None else None
    overwritten: dict[tuple[str, str], Any] = {}
    point_ids_overwritten = False
    instances_size_overwritten = False
    audit: dict[str, Any] = {
        "operation_order": list(OPERATION_ORDER),
        "base_counts": counts,
        "erase_counts": {
            name: int(indices.size) for name, indices in erased_by_model.items()
        },
        "insert_background_count": (
            int(np.asarray(background_delta["means"]).shape[0])
            if background_delta is not None
            else 0
        ),
        "insert_actor_count": (
            int(np.asarray(actor_delta["means"]).shape[0])
            if actor_delta is not None
            else 0
        ),
        "base_rows_deleted": 0,
        "duplicate_insert_indices": 0,
        "effective_erased_opacity_nonzero": 0,
    }
    try:
        for model_name, model, attributes in (
            ("Background", background, tuple(BACKGROUND_ATTRS)),
            ("RigidNodes", rigid, RIGID_ATTRS),
        ):
            erase_indices = erased_by_model[model_name]
            for attribute in attributes:
                tail: torch.Tensor | None = None
                if model_name == "Background" and background_delta is not None:
                    source_name = BACKGROUND_ATTRS[attribute]
                    original = getattr(model, attribute)
                    tail = torch.as_tensor(
                        background_delta[source_name],
                        device=original.device,
                        dtype=original.dtype,
                    )
                elif model_name == "RigidNodes" and actor_tails is not None:
                    tail = actor_tails[attribute]
                if erase_indices.size == 0 and tail is None:
                    continue
                original = getattr(model, attribute)
                base = original.detach()
                if attribute == "_opacities" and erase_indices.size:
                    base = base.clone()
                    index = torch.as_tensor(
                        erase_indices, device=base.device, dtype=torch.long
                    )
                    base[index] = torch.finfo(base.dtype).min
                combined = torch.cat([base, tail], dim=0) if tail is not None else base
                overwritten[(model_name, attribute)] = original
                setattr(model, attribute, _parameter(combined, original))

        if actor_delta is not None:
            actor_index = int(np.asarray(actor_delta["rigid_model_index"]).item())
            original_ids = rigid.point_ids
            id_shape = (int(np.asarray(actor_delta["means"]).shape[0]),) + tuple(
                original_ids.shape[1:]
            )
            new_ids = torch.full(
                id_shape,
                actor_index,
                device=original_ids.device,
                dtype=original_ids.dtype,
            )
            rigid.point_ids = torch.cat([original_ids, new_ids], dim=0)
            point_ids_overwritten = True
            if not torch.equal(rigid.point_ids[: counts["RigidNodes"]], original_ids):
                raise RuntimeError("INSERT_ACTOR 改变了 base actor index prefix")
            if hasattr(rigid, "instances_size") and "target_lwh" in actor_delta:
                original_sizes = rigid.instances_size
                updated_sizes = original_sizes.detach().clone()
                updated_sizes[actor_index] = torch.as_tensor(
                    actor_delta["target_lwh"],
                    device=updated_sizes.device,
                    dtype=updated_sizes.dtype,
                )
                rigid.instances_size = updated_sizes
                instances_size_overwritten = True

        for model_name, model in (("Background", background), ("RigidNodes", rigid)):
            indices = erased_by_model[model_name]
            if indices.size:
                values = torch.sigmoid(
                    model._opacities[
                        torch.as_tensor(indices, device=model._opacities.device)
                    ]
                )
                audit["effective_erased_opacity_nonzero"] += int(
                    torch.count_nonzero(values).item()
                )
        if audit["effective_erased_opacity_nonzero"] != 0:
            raise RuntimeError("ERASE effective opacity 未精确归零")
        audit["composed_counts"] = _live_counts(background, rigid)
        yield audit
    finally:
        for (model_name, attribute), original in overwritten.items():
            model = background if model_name == "Background" else rigid
            setattr(model, attribute, original)
        if point_ids_overwritten:
            rigid.point_ids = originals["RigidNodes.point_ids"]
        if instances_size_overwritten:
            rigid.instances_size = originals["RigidNodes.instances_size"]
        _assert_reference_objects(background, rigid, references)

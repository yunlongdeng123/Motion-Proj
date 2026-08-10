"""WorldSim V3.3 instance-opacity sidecar 的构造、schema 与校验。"""

from __future__ import annotations

from dataclasses import dataclass
import io
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from motion_proj.worldsim_v32.semantic_schema import (
    AMBIGUOUS,
    CORE_POSITIVE,
    SEMANTIC_POSITIVE,
)
from motion_proj.worldsim_v33.semantic_reassignment import (
    resolve_background_assignments,
)


NO_INSTANCE = np.int32(-1)
BASE_BACKGROUND = np.int8(0)
BASE_RIGID = np.int8(1)
PROVENANCE_NONE = np.uint8(0)
PROVENANCE_RIGID_CORE = np.uint8(1)
PROVENANCE_SEMANTIC_POSITIVE = np.uint8(2)
PROVENANCE_AMBIGUOUS_REASSIGNED = np.uint8(3)

REQUIRED_FIELDS = {
    "gaussian_id",
    "base_model",
    "base_index",
    "hard_instance_id",
    "instance_opacity_logit",
    "instance_opacity",
    "source_semantic_score",
    "num_positive_views",
    "num_negative_views",
    "visibility_mass",
    "trainable",
    "provenance",
    "actor_instance_ids",
    "actor_tokens",
}


@dataclass(frozen=True)
class ActorSemanticSource:
    role: str
    instance_id: int
    instance_token: str
    rigid_model_index: int
    arrays: Mapping[str, np.ndarray]


def probability_to_logit(value: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    probability = np.clip(np.asarray(value, dtype=np.float32), epsilon, 1.0 - epsilon)
    return (np.log(probability) - np.log1p(-probability)).astype(np.float32)


def logit_to_probability(value: np.ndarray) -> np.ndarray:
    logits = np.asarray(value, dtype=np.float32)
    positive = logits >= 0
    output = np.empty_like(logits)
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _validate_sources(sources: Sequence[ActorSemanticSource]) -> tuple[int, int]:
    if not sources:
        raise ValueError("instance field 至少需要一个 actor")
    if len({int(source.instance_id) for source in sources}) != len(sources):
        raise ValueError("actor instance_id 必须唯一")
    required = {
        "labels",
        "semantic_score",
        "num_positive_views",
        "num_negative_views",
        "visible_mass",
        "boundary_score",
        "background_count",
        "rigid_point_ids",
    }
    total: int | None = None
    background_count: int | None = None
    rigid_ids: np.ndarray | None = None
    for source in sources:
        missing = required - set(source.arrays)
        if missing:
            raise ValueError(f"{source.role} semantic sidecar 缺字段: {sorted(missing)}")
        labels = np.asarray(source.arrays["labels"])
        if labels.ndim != 1:
            raise ValueError(f"{source.role} labels 必须是一维")
        if total is None:
            total = int(labels.size)
            background_count = int(np.asarray(source.arrays["background_count"]).item())
            rigid_ids = np.asarray(source.arrays["rigid_point_ids"], dtype=np.int64)
        if labels.size != total:
            raise ValueError("actor semantic sidecar Gaussian 数量不一致")
        if int(np.asarray(source.arrays["background_count"]).item()) != background_count:
            raise ValueError("actor semantic sidecar background_count 不一致")
        np.testing.assert_array_equal(
            np.asarray(source.arrays["rigid_point_ids"], dtype=np.int64), rigid_ids
        )
        for name in required - {"background_count", "rigid_point_ids"}:
            if np.asarray(source.arrays[name]).shape != (total,):
                raise ValueError(f"{source.role}/{name} shape 不合法")
    assert total is not None and background_count is not None and rigid_ids is not None
    if not 0 < background_count < total or rigid_ids.size != total - background_count:
        raise ValueError("Background/RigidNodes Gaussian 计数合同不合法")
    return total, background_count


def build_instance_field(
    *,
    sources: Sequence[ActorSemanticSource],
    arm: str,
    allow_ambiguous_reassignment: bool,
    ambiguous_minimum_score: float,
    ambiguous_minimum_boundary_score: float,
    assignment_minimum_margin: float,
    rigid_core_opacity: float,
    unassigned_opacity: float,
) -> dict[str, np.ndarray]:
    """从冻结 V3.2 posterior 构造单一全局 instance field。"""
    total, background_count = _validate_sources(sources)
    actor_count = len(sources)
    actor_ids = np.asarray([source.instance_id for source in sources], dtype=np.int32)
    scores = np.stack(
        [np.asarray(source.arrays["semantic_score"], dtype=np.float32) for source in sources]
    )
    labels = np.stack(
        [np.asarray(source.arrays["labels"], dtype=np.int8) for source in sources]
    )
    boundaries = np.stack(
        [np.asarray(source.arrays["boundary_score"], dtype=np.float32) for source in sources]
    )
    eligible = labels == SEMANTIC_POSITIVE
    ambiguous_eligible = np.zeros_like(eligible)
    if allow_ambiguous_reassignment:
        ambiguous_eligible = (
            (labels == AMBIGUOUS)
            & (scores >= float(ambiguous_minimum_score))
            & (boundaries >= float(ambiguous_minimum_boundary_score))
        )
        eligible |= ambiguous_eligible
    eligible[:, background_count:] = False

    background_assignment, background_score, conflict = resolve_background_assignments(
        instance_ids=actor_ids,
        scores=scores[:, :background_count],
        eligible=eligible[:, :background_count],
        minimum_margin=float(assignment_minimum_margin),
    )
    hard_instance_id = np.full(total, NO_INSTANCE, dtype=np.int32)
    hard_instance_id[:background_count] = background_assignment
    source_score = np.zeros(total, dtype=np.float32)
    source_score[:background_count] = background_score
    provenance = np.full(total, PROVENANCE_NONE, dtype=np.uint8)
    positive_views = np.zeros(total, dtype=np.int32)
    negative_views = np.zeros(total, dtype=np.int32)
    visibility_mass = np.zeros(total, dtype=np.float32)

    for actor_index, source in enumerate(sources):
        selected = hard_instance_id == int(source.instance_id)
        actor_labels = labels[actor_index]
        provenance[selected & (actor_labels == SEMANTIC_POSITIVE)] = (
            PROVENANCE_SEMANTIC_POSITIVE
        )
        provenance[selected & ambiguous_eligible[actor_index]] = (
            PROVENANCE_AMBIGUOUS_REASSIGNED
        )
        positive_views[selected] = np.asarray(
            source.arrays["num_positive_views"], dtype=np.int32
        )[selected]
        negative_views[selected] = np.asarray(
            source.arrays["num_negative_views"], dtype=np.int32
        )[selected]
        visibility_mass[selected] = np.asarray(
            source.arrays["visible_mass"], dtype=np.float32
        )[selected]

        core = actor_labels == CORE_POSITIVE
        if np.any((hard_instance_id[core] != NO_INSTANCE) & (hard_instance_id[core] != source.instance_id)):
            raise ValueError(f"{source.role} rigid core 与其他 actor 冲突")
        hard_instance_id[core] = int(source.instance_id)
        source_score[core] = 1.0
        provenance[core] = PROVENANCE_RIGID_CORE
        positive_views[core] = np.asarray(
            source.arrays["num_positive_views"], dtype=np.int32
        )[core]
        negative_views[core] = np.asarray(
            source.arrays["num_negative_views"], dtype=np.int32
        )[core]
        visibility_mass[core] = np.asarray(
            source.arrays["visible_mass"], dtype=np.float32
        )[core]

    opacity = np.full(total, float(unassigned_opacity), dtype=np.float32)
    assigned = hard_instance_id != NO_INSTANCE
    opacity[assigned] = np.clip(source_score[assigned], 0.05, 0.95)
    opacity[provenance == PROVENANCE_RIGID_CORE] = float(rigid_core_opacity)
    trainable = assigned if arm != "O0_heuristic" else np.zeros(total, dtype=bool)
    base_model = np.full(total, BASE_BACKGROUND, dtype=np.int8)
    base_model[background_count:] = BASE_RIGID
    base_index = np.arange(total, dtype=np.int64)
    base_index[background_count:] -= background_count

    field = {
        "gaussian_id": np.arange(total, dtype=np.int64),
        "base_model": base_model,
        "base_index": base_index,
        "hard_instance_id": hard_instance_id,
        "instance_opacity_logit": probability_to_logit(opacity),
        "instance_opacity": opacity,
        "source_semantic_score": source_score,
        "num_positive_views": positive_views,
        "num_negative_views": negative_views,
        "visibility_mass": visibility_mass,
        "trainable": trainable.astype(bool),
        "provenance": provenance,
        "actor_instance_ids": actor_ids,
        "actor_tokens": np.asarray(
            [source.instance_token for source in sources], dtype="<U64"
        ),
        "reassignment_conflict": np.pad(
            conflict.astype(bool), (0, total - background_count)
        ),
    }
    validate_instance_field(field)
    return field


def validate_instance_field(field: Mapping[str, np.ndarray]) -> None:
    missing = REQUIRED_FIELDS - set(field)
    if missing:
        raise ValueError(f"instance field 缺字段: {sorted(missing)}")
    gaussian_id = np.asarray(field["gaussian_id"], dtype=np.int64)
    total = int(gaussian_id.size)
    if gaussian_id.ndim != 1 or not np.array_equal(gaussian_id, np.arange(total)):
        raise ValueError("gaussian_id 必须是连续全局索引")
    per_gaussian = REQUIRED_FIELDS - {"actor_instance_ids", "actor_tokens"}
    for name in per_gaussian:
        if np.asarray(field[name]).shape != (total,):
            raise ValueError(f"instance field {name} shape 不合法")
    actor_ids = np.asarray(field["actor_instance_ids"], dtype=np.int32)
    actor_tokens = np.asarray(field["actor_tokens"])
    if actor_ids.ndim != 1 or actor_tokens.shape != actor_ids.shape:
        raise ValueError("actor identity mapping shape 不合法")
    allowed = set(int(value) for value in actor_ids) | {int(NO_INSTANCE)}
    actual = set(int(value) for value in np.unique(field["hard_instance_id"]))
    if not actual <= allowed:
        raise ValueError(f"hard_instance_id 存在未知身份: {sorted(actual - allowed)}")
    trainable = np.asarray(field["trainable"], dtype=bool)
    if np.any(trainable & (np.asarray(field["hard_instance_id"]) == NO_INSTANCE)):
        raise ValueError("未分配 Gaussian 不能 trainable")
    logits = np.asarray(field["instance_opacity_logit"], dtype=np.float32)
    opacity = np.asarray(field["instance_opacity"], dtype=np.float32)
    if not np.isfinite(logits).all() or not np.isfinite(opacity).all():
        raise ValueError("instance opacity 必须有限")
    if np.any((opacity <= 0) | (opacity >= 1)):
        raise ValueError("instance opacity 必须严格位于 (0, 1)")
    np.testing.assert_allclose(logit_to_probability(logits), opacity, atol=2e-6)
    base_model = np.asarray(field["base_model"], dtype=np.int8)
    if not set(int(value) for value in np.unique(base_model)) <= {
        int(BASE_BACKGROUND), int(BASE_RIGID)
    }:
        raise ValueError("base_model 存在未知值")


def atomic_save_instance_field(path: str | Path, field: Mapping[str, np.ndarray]) -> None:
    validate_instance_field(field)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        # numpy.savez_compressed 会把当前时间写入 ZIP header，数组 exact 仍会导致
        # 文件 SHA 漂移。固定 entry 顺序、时间戳与权限，形成真正 byte-exact sidecar。
        with zipfile.ZipFile(
            handle,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(field):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.asarray(field[name]), allow_pickle=False
                )
                entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
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


def load_instance_field(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        field = {name: arrays[name] for name in arrays.files}
    validate_instance_field(field)
    return field


def field_summary(field: Mapping[str, np.ndarray]) -> dict[str, Any]:
    validate_instance_field(field)
    hard = np.asarray(field["hard_instance_id"], dtype=np.int32)
    provenance = np.asarray(field["provenance"], dtype=np.uint8)
    return {
        "gaussian_count": int(hard.size),
        "assigned_count": int((hard != NO_INSTANCE).sum()),
        "trainable_count": int(np.asarray(field["trainable"], dtype=bool).sum()),
        "reassignment_conflict_count": int(
            np.asarray(field.get("reassignment_conflict", np.zeros(hard.size))).sum()
        ),
        "actor_counts": {
            str(int(instance_id)): int((hard == int(instance_id)).sum())
            for instance_id in np.asarray(field["actor_instance_ids"])
        },
        "provenance_counts": {
            "none": int((provenance == PROVENANCE_NONE).sum()),
            "rigid_core": int((provenance == PROVENANCE_RIGID_CORE).sum()),
            "semantic_positive": int(
                (provenance == PROVENANCE_SEMANTIC_POSITIVE).sum()
            ),
            "ambiguous_reassigned": int(
                (provenance == PROVENANCE_AMBIGUOUS_REASSIGNED).sum()
            ),
        },
    }

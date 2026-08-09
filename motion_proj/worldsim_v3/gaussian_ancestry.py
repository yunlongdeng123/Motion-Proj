"""WorldSim V3 A2 的逐 Gaussian 来源与诊断账本。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import IntEnum
from typing import Any

import torch
from torch import Tensor


class InitSource(IntEnum):
    """Gaussian 出生来源的稳定编码。"""

    UNKNOWN = 0
    LIDAR = 1
    RANDOM_NEAR = 2
    RANDOM_FAR = 3
    SPLIT = 4
    CLONE = 5


INIT_SOURCE_NAMES = {
    int(InitSource.UNKNOWN): "unknown",
    int(InitSource.LIDAR): "lidar",
    int(InitSource.RANDOM_NEAR): "random-near",
    int(InitSource.RANDOM_FAR): "random-far",
    int(InitSource.SPLIT): "split",
    int(InitSource.CLONE): "clone",
}

RUNNING_METRICS = (
    "screen_grad",
    "boundary_contribution",
    "photometric_residual",
    "depth_residual",
    "normal_residual",
)

PER_GAUSSIAN_FIELDS = (
    "gaussian_id",
    "actor_id",
    "init_source",
    "parent_id",
    "lineage_root_id",
    "birth_step",
    "generation",
    "visibility_count",
    "screen_grad",
    "screen_grad_count",
    "boundary_contribution",
    "boundary_contribution_count",
    "photometric_residual",
    "photometric_residual_count",
    "depth_residual",
    "depth_residual_count",
    "normal_residual",
    "normal_residual_count",
    "nearest_lidar_distance",
)


def _one_dimensional(
    value: Tensor | int | float,
    *,
    length: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(length).clone()
    if tensor.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {tuple(tensor.shape)}")
    return tensor


class GaussianAncestryLedger:
    """保存与模型参数同索引的非梯度 provenance。

    账本不是 ``nn.Module``，避免模块关闭时改变上游 checkpoint 键。启用时由
    DriveStudio 的 Gaussian 类显式写入和恢复 ``worldsim_a2_ancestry``。
    """

    schema_version = 1

    def __init__(self, *, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.next_gaussian_id = 0
        self.initial_gaussian_count = 0
        self.reference_lidar_positions = torch.empty(
            (0, 3), device=self.device, dtype=torch.float32
        )
        self.reference_lidar_actor_ids = torch.empty(
            (0,), device=self.device, dtype=torch.long
        )
        for name in PER_GAUSSIAN_FIELDS:
            dtype = torch.float32 if (
                name in RUNNING_METRICS or name == "nearest_lidar_distance"
            ) else torch.long
            setattr(self, name, torch.empty((0,), device=self.device, dtype=dtype))

    @classmethod
    def initialize(
        cls,
        *,
        means: Tensor,
        actor_ids: Tensor | int,
        init_sources: Tensor | int,
        birth_step: int = 0,
    ) -> "GaussianAncestryLedger":
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError(f"means must have shape (N, 3), got {tuple(means.shape)}")
        ledger = cls(device=means.device)
        count = int(means.shape[0])
        actor = _one_dimensional(
            actor_ids,
            length=count,
            device=means.device,
            dtype=torch.long,
            name="actor_ids",
        )
        source = _one_dimensional(
            init_sources,
            length=count,
            device=means.device,
            dtype=torch.long,
            name="init_sources",
        )
        valid_codes = torch.tensor(
            sorted(INIT_SOURCE_NAMES), device=means.device, dtype=torch.long
        )
        if count and not torch.isin(source, valid_codes).all():
            raise ValueError("init_sources contains an unknown code")

        ids = torch.arange(count, device=means.device, dtype=torch.long)
        ledger.gaussian_id = ids
        ledger.actor_id = actor
        ledger.init_source = source
        ledger.parent_id = torch.full_like(ids, -1)
        ledger.lineage_root_id = ids.clone()
        ledger.birth_step = torch.full_like(ids, int(birth_step))
        ledger.generation = torch.zeros_like(ids)
        ledger.visibility_count = torch.zeros_like(ids)
        for metric in RUNNING_METRICS:
            setattr(
                ledger,
                metric,
                torch.full((count,), torch.nan, device=means.device),
            )
            setattr(ledger, f"{metric}_count", torch.zeros_like(ids))
        ledger.nearest_lidar_distance = torch.full(
            (count,), torch.nan, device=means.device
        )
        lidar_mask = source == int(InitSource.LIDAR)
        ledger.nearest_lidar_distance[lidar_mask] = 0.0
        ledger.reference_lidar_positions = means.detach()[lidar_mask].clone()
        ledger.reference_lidar_actor_ids = actor[lidar_mask].clone()
        ledger.next_gaussian_id = count
        ledger.initial_gaussian_count = count
        ledger.validate(expected_actor_ids=actor)
        return ledger

    def __len__(self) -> int:
        return int(self.gaussian_id.shape[0])

    def _append(self, rows: Mapping[str, Tensor]) -> None:
        row_count = None
        for name in PER_GAUSSIAN_FIELDS:
            if name not in rows:
                raise KeyError(f"missing ancestry field: {name}")
            value = rows[name].to(device=self.device, dtype=getattr(self, name).dtype)
            if value.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if row_count is None:
                row_count = int(value.shape[0])
            elif value.shape[0] != row_count:
                raise ValueError("ancestry append rows have inconsistent lengths")
            setattr(self, name, torch.cat([getattr(self, name), value], dim=0))

    def append_children(
        self,
        *,
        parent_indices: Tensor,
        repeats: int,
        source: InitSource,
        birth_step: int,
        child_means: Tensor | None = None,
    ) -> Tensor:
        if source not in (InitSource.SPLIT, InitSource.CLONE):
            raise ValueError("children must be born from split or clone")
        parents = torch.as_tensor(
            parent_indices, device=self.device, dtype=torch.long
        ).flatten()
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        if parents.numel() and (
            int(parents.min()) < 0 or int(parents.max()) >= len(self)
        ):
            raise IndexError("parent index is outside the live ledger")
        expanded = parents.repeat(int(repeats))
        count = int(expanded.numel())
        if child_means is not None and tuple(child_means.shape) != (count, 3):
            raise ValueError("child_means must align with expanded parents")
        ids = torch.arange(
            self.next_gaussian_id,
            self.next_gaussian_id + count,
            device=self.device,
            dtype=torch.long,
        )
        zeros = torch.zeros(count, device=self.device, dtype=torch.long)
        nan = torch.full((count,), torch.nan, device=self.device)
        nearest = nan.clone()
        if source == InitSource.CLONE:
            nearest = self.nearest_lidar_distance[expanded].clone()
        rows: dict[str, Tensor] = {
            "gaussian_id": ids,
            "actor_id": self.actor_id[expanded].clone(),
            "init_source": torch.full_like(ids, int(source)),
            "parent_id": self.gaussian_id[expanded].clone(),
            "lineage_root_id": self.lineage_root_id[expanded].clone(),
            "birth_step": torch.full_like(ids, int(birth_step)),
            "generation": self.generation[expanded] + 1,
            "visibility_count": zeros.clone(),
            "nearest_lidar_distance": nearest,
        }
        for metric in RUNNING_METRICS:
            rows[metric] = nan.clone()
            rows[f"{metric}_count"] = zeros.clone()
        self._append(rows)
        self.next_gaussian_id += count
        return ids

    def select(self, mask: Tensor) -> dict[str, Tensor]:
        selected = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if selected.shape != (len(self),):
            raise ValueError("selection mask must align with the live ledger")
        return {name: getattr(self, name)[selected].clone() for name in PER_GAUSSIAN_FIELDS}

    def append_external_clones(
        self,
        *,
        source_rows: Mapping[str, Tensor],
        actor_id: int,
        birth_step: int,
    ) -> Tensor:
        source_ids = source_rows["gaussian_id"].to(self.device, dtype=torch.long)
        count = int(source_ids.numel())
        ids = torch.arange(
            self.next_gaussian_id,
            self.next_gaussian_id + count,
            device=self.device,
            dtype=torch.long,
        )
        zeros = torch.zeros(count, device=self.device, dtype=torch.long)
        nan = torch.full((count,), torch.nan, device=self.device)
        rows: dict[str, Tensor] = {
            "gaussian_id": ids,
            "actor_id": torch.full_like(ids, int(actor_id)),
            "init_source": torch.full_like(ids, int(InitSource.CLONE)),
            "parent_id": source_ids,
            "lineage_root_id": source_rows["lineage_root_id"].to(
                self.device, dtype=torch.long
            ),
            "birth_step": torch.full_like(ids, int(birth_step)),
            "generation": source_rows["generation"].to(
                self.device, dtype=torch.long
            ) + 1,
            "visibility_count": zeros.clone(),
            "nearest_lidar_distance": source_rows[
                "nearest_lidar_distance"
            ].to(self.device, dtype=torch.float32),
        }
        for metric in RUNNING_METRICS:
            rows[metric] = nan.clone()
            rows[f"{metric}_count"] = zeros.clone()
        self._append(rows)
        self.next_gaussian_id += count
        return ids

    def prune(self, keep_mask: Tensor) -> None:
        keep = torch.as_tensor(keep_mask, device=self.device, dtype=torch.bool)
        if keep.shape != (len(self),):
            raise ValueError("prune mask must align with the live ledger")
        for name in PER_GAUSSIAN_FIELDS:
            setattr(self, name, getattr(self, name)[keep])

    def _record_running_mean(
        self,
        *,
        metric: str,
        indices: Tensor,
        values: Tensor,
    ) -> None:
        if metric not in RUNNING_METRICS:
            raise KeyError(metric)
        index = torch.as_tensor(indices, device=self.device, dtype=torch.long).flatten()
        value = torch.as_tensor(
            values, device=self.device, dtype=torch.float32
        ).flatten()
        if index.shape != value.shape:
            raise ValueError(f"{metric} values must align with indices")
        if index.numel() == 0:
            return
        if int(index.min()) < 0 or int(index.max()) >= len(self):
            raise IndexError(f"{metric} index is outside the live ledger")
        finite = torch.isfinite(value)
        index = index[finite]
        value = value[finite]
        if index.numel() == 0:
            return
        if torch.unique(index).numel() != index.numel():
            raise ValueError(f"{metric} indices must be unique per update")
        current = getattr(self, metric)[index]
        counts_name = f"{metric}_count"
        counts = getattr(self, counts_name)[index]
        current = torch.nan_to_num(current, nan=0.0)
        updated = (current * counts.to(current.dtype) + value) / (
            counts.to(current.dtype) + 1.0
        )
        getattr(self, metric)[index] = updated
        getattr(self, counts_name)[index] = counts + 1

    def record_screen_statistics(self, *, indices: Tensor, gradients: Tensor) -> None:
        index = torch.as_tensor(indices, device=self.device, dtype=torch.long).flatten()
        gradient = torch.as_tensor(
            gradients, device=self.device, dtype=torch.float32
        ).flatten()
        if index.shape != gradient.shape:
            raise ValueError("screen gradients must align with visible indices")
        if index.numel():
            self.visibility_count[index] += 1
        self._record_running_mean(
            metric="screen_grad", indices=index, values=gradient
        )

    def record_diagnostics(
        self,
        *,
        indices: Tensor,
        boundary_contribution: Tensor | None = None,
        photometric_residual: Tensor | None = None,
        depth_residual: Tensor | None = None,
        normal_residual: Tensor | None = None,
    ) -> None:
        values = {
            "boundary_contribution": boundary_contribution,
            "photometric_residual": photometric_residual,
            "depth_residual": depth_residual,
            "normal_residual": normal_residual,
        }
        for metric, value in values.items():
            if value is not None:
                self._record_running_mean(
                    metric=metric, indices=indices, values=value
                )

    def materialize_nearest_lidar_distance(
        self,
        *,
        means: Tensor,
        actor_ids: Iterable[int] | None = None,
        chunk_size: int = 1024,
        maximum_reference_points: int = 10_000,
    ) -> dict[str, Any]:
        if tuple(means.shape) != (len(self), 3):
            raise ValueError("means must align with the live ledger")
        if chunk_size <= 0 or maximum_reference_points <= 0:
            raise ValueError("distance materialization limits must be positive")
        requested = (
            {int(value) for value in actor_ids}
            if actor_ids is not None
            else {int(value) for value in torch.unique(self.actor_id).tolist()}
        )
        completed: dict[str, int] = {}
        skipped: dict[str, str] = {}
        for actor in sorted(requested):
            live_mask = self.actor_id == actor
            reference_mask = self.reference_lidar_actor_ids == actor
            live_count = int(live_mask.sum())
            reference_count = int(reference_mask.sum())
            if live_count == 0:
                skipped[str(actor)] = "no_live_gaussians"
                continue
            if reference_count == 0:
                skipped[str(actor)] = "no_lidar_reference"
                continue
            if reference_count > maximum_reference_points:
                skipped[str(actor)] = "reference_limit_exceeded"
                continue
            live_indices = torch.where(live_mask)[0]
            references = self.reference_lidar_positions[reference_mask]
            for start in range(0, live_count, chunk_size):
                chunk_indices = live_indices[start : start + chunk_size]
                distances = torch.cdist(
                    means.detach()[chunk_indices].to(self.device), references
                )
                self.nearest_lidar_distance[chunk_indices] = distances.min(dim=1).values
            completed[str(actor)] = live_count
        return {
            "completed": completed,
            "skipped": skipped,
            "chunk_size": int(chunk_size),
            "maximum_reference_points": int(maximum_reference_points),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "next_gaussian_id": int(self.next_gaussian_id),
            "initial_gaussian_count": int(self.initial_gaussian_count),
            "reference_lidar_positions": self.reference_lidar_positions,
            "reference_lidar_actor_ids": self.reference_lidar_actor_ids,
            "fields": {name: getattr(self, name) for name in PER_GAUSSIAN_FIELDS},
        }

    @classmethod
    def from_state_dict(
        cls, payload: Mapping[str, Any], *, device: torch.device | str
    ) -> "GaussianAncestryLedger":
        if int(payload.get("schema_version", -1)) != cls.schema_version:
            raise ValueError("unsupported Gaussian ancestry schema")
        ledger = cls(device=device)
        ledger.next_gaussian_id = int(payload["next_gaussian_id"])
        ledger.initial_gaussian_count = int(payload["initial_gaussian_count"])
        ledger.reference_lidar_positions = torch.as_tensor(
            payload["reference_lidar_positions"],
            device=ledger.device,
            dtype=torch.float32,
        ).clone()
        ledger.reference_lidar_actor_ids = torch.as_tensor(
            payload["reference_lidar_actor_ids"],
            device=ledger.device,
            dtype=torch.long,
        ).clone()
        fields = payload.get("fields")
        if not isinstance(fields, Mapping):
            raise ValueError("ancestry state is missing fields")
        for name in PER_GAUSSIAN_FIELDS:
            if name not in fields:
                raise ValueError(f"ancestry state is missing {name}")
            dtype = getattr(ledger, name).dtype
            setattr(
                ledger,
                name,
                torch.as_tensor(fields[name], device=ledger.device, dtype=dtype)
                .flatten()
                .clone(),
            )
        ledger.validate()
        return ledger

    def validate(self, *, expected_actor_ids: Tensor | None = None) -> None:
        count = len(self)
        for name in PER_GAUSSIAN_FIELDS:
            if getattr(self, name).shape != (count,):
                raise ValueError(f"ancestry field {name} is not aligned")
        if self.reference_lidar_positions.ndim != 2 or (
            self.reference_lidar_positions.shape[1:] != (3,)
        ):
            raise ValueError("reference LiDAR positions must have shape (N, 3)")
        if self.reference_lidar_actor_ids.shape != (
            self.reference_lidar_positions.shape[0],
        ):
            raise ValueError("reference LiDAR actor IDs are not aligned")
        if count:
            if torch.unique(self.gaussian_id).numel() != count:
                raise ValueError("live gaussian IDs must be unique")
            if int(self.gaussian_id.min()) < 0 or int(self.gaussian_id.max()) >= self.next_gaussian_id:
                raise ValueError("live gaussian IDs are outside the allocation range")
            roots_valid = (self.lineage_root_id >= 0) & (
                self.lineage_root_id < self.initial_gaussian_count
            )
            if not roots_valid.all():
                raise ValueError("lineage roots must reference initial Gaussian IDs")
            parents_valid = (self.parent_id == -1) | (
                (self.parent_id >= 0) & (self.parent_id < self.next_gaussian_id)
            )
            if not parents_valid.all():
                raise ValueError("parent IDs are outside the allocation range")
        if expected_actor_ids is not None:
            expected = torch.as_tensor(
                expected_actor_ids, device=self.device, dtype=torch.long
            ).flatten()
            if not torch.equal(self.actor_id, expected):
                raise ValueError("ancestry actor IDs differ from model point IDs")

    def summary(self) -> dict[str, Any]:
        source_counts = {
            INIT_SOURCE_NAMES[code]: int((self.init_source == code).sum())
            for code in sorted(INIT_SOURCE_NAMES)
        }
        return {
            "schema_version": self.schema_version,
            "live_gaussians": len(self),
            "initial_gaussians": int(self.initial_gaussian_count),
            "allocated_gaussian_ids": int(self.next_gaussian_id),
            "source_counts": source_counts,
            "visibility_observed": int((self.visibility_count > 0).sum()),
            "nearest_lidar_materialized": int(
                torch.isfinite(self.nearest_lidar_distance).sum()
            ),
        }


def validate_a2_instrumentation_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("A2 instrumentation schema_version must be 1")
    if payload.get("task_id") != "WS-V3-A2-ACTOR-DENSIFY-01":
        raise ValueError("unexpected A2 task_id")
    instrumentation = payload.get("instrumentation")
    if not isinstance(instrumentation, Mapping) or not instrumentation.get("enabled"):
        raise ValueError("A2 instrumentation must be enabled")
    configured_sources = instrumentation.get("init_source_codes")
    expected_sources = {
        name: code for code, name in sorted(INIT_SOURCE_NAMES.items())
    }
    if configured_sources != expected_sources:
        raise ValueError("A2 init_source codes changed")
    required_fields = set(PER_GAUSSIAN_FIELDS)
    if set(instrumentation.get("per_gaussian_fields", [])) != required_fields:
        raise ValueError("A2 per-Gaussian field contract changed")
    module_off = payload.get("module_off_equivalence")
    if not isinstance(module_off, Mapping):
        raise ValueError("module-off equivalence contract is missing")
    if module_off.get("enabled_value") is not False:
        raise ValueError("module-off control value must remain false")
    if not module_off.get("require_native_tensor_bitwise_equality"):
        raise ValueError("module-off bitwise equality must be required")
    if not module_off.get("require_native_checkpoint_key_equality"):
        raise ValueError("module-off checkpoint key equality must be required")
    boundary = payload.get("scope_boundary")
    if not isinstance(boundary, Mapping) or boundary.get(
        "legacy_checkpoint_without_ancestry"
    ) != "load_as_unknown_not_lidar":
        raise ValueError("legacy checkpoints must not invent LiDAR ancestry")

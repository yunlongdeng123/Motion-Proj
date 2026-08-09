from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


SCHEMA_VERSION = 1
EXPECTED_TASK_ID = "WS-V3-A2-ACTOR-DENSIFY-01"
EXPECTED_AUDIT_VERSION = "A2-D2-PROTOCOL-v1"
ORDERING_KEYS = (
    "boundary_observed_desc",
    "boundary_mean_desc",
    "photometric_residual_observed_desc",
    "photometric_residual_mean_desc",
    "screen_grad_desc",
    "gaussian_index_asc",
)


@dataclass(frozen=True)
class BoundaryResidualPolicy:
    boundary_radius_pixels: int = 3
    mask_binarization_threshold: float = 0.5
    scale_cap_threshold_multiplier: float = 1.0
    ranking: str = "boundary_residual_screen_grad_then_gaussian_index"

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "BoundaryResidualPolicy":
        policy = cls(
            boundary_radius_pixels=int(payload["boundary_radius_pixels"]),
            mask_binarization_threshold=float(
                payload["mask_binarization_threshold"]
            ),
            scale_cap_threshold_multiplier=float(
                payload["scale_cap_threshold_multiplier"]
            ),
            ranking=str(payload["ranking"]),
        )
        policy.validate()
        return policy

    @classmethod
    def from_contract(
        cls, payload: Mapping[str, Any]
    ) -> "BoundaryResidualPolicy":
        attribution = payload["training_attribution"]
        scale_cap = payload["boundary_scale_cap"]
        policy = cls(
            boundary_radius_pixels=int(
                attribution["boundary_band"]["radius_pixels"]
            ),
            mask_binarization_threshold=float(
                attribution["mask_binarization_threshold"]
            ),
            scale_cap_threshold_multiplier=float(
                scale_cap["threshold_multiplier"]
            ),
            ranking="boundary_residual_screen_grad_then_gaussian_index",
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.boundary_radius_pixels < 1:
            raise ValueError("boundary radius must be positive")
        if not math.isfinite(self.mask_binarization_threshold):
            raise ValueError("mask threshold must be finite")
        if not 0.0 < self.mask_binarization_threshold < 1.0:
            raise ValueError("mask threshold must be in the interval (0, 1)")
        if self.scale_cap_threshold_multiplier != 1.0:
            raise ValueError("D2 scale cap must reuse the native size threshold")
        if self.ranking != "boundary_residual_screen_grad_then_gaussian_index":
            raise ValueError("unsupported D2 boundary/residual ranking")


class BoundaryResidualState:
    schema_version = 1

    def __init__(self, *, policy: BoundaryResidualPolicy) -> None:
        policy.validate()
        self.policy = policy
        self.observation_events = 0
        self.boundary_observations = 0
        self.photometric_residual_observations = 0
        self.refinement_events = 0
        self.capped_gaussians = 0
        self.last_refinement: dict[str, Any] | None = None

    def record_observations(
        self, *, boundary_count: int, residual_count: int
    ) -> None:
        if boundary_count < 0 or residual_count < 0:
            raise ValueError("observation counts must be non-negative")
        self.observation_events += 1
        self.boundary_observations += int(boundary_count)
        self.photometric_residual_observations += int(residual_count)

    def record_refinement(
        self,
        *,
        step: int,
        boundary_observed_gaussians: int,
        residual_observed_gaussians: int,
        capped_gaussians: int,
        maximum_scale: float,
    ) -> None:
        counts = (
            boundary_observed_gaussians,
            residual_observed_gaussians,
            capped_gaussians,
        )
        if any(value < 0 for value in counts):
            raise ValueError("refinement counts must be non-negative")
        if not math.isfinite(maximum_scale) or maximum_scale <= 0:
            raise ValueError("maximum_scale must be finite and positive")
        self.refinement_events += 1
        self.capped_gaussians += int(capped_gaussians)
        self.last_refinement = {
            "event": self.refinement_events,
            "step": int(step),
            "boundary_observed_gaussians": int(
                boundary_observed_gaussians
            ),
            "residual_observed_gaussians": int(residual_observed_gaussians),
            "capped_gaussians": int(capped_gaussians),
            "maximum_scale": float(maximum_scale),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": asdict(self.policy),
            "counters": {
                "observation_events": self.observation_events,
                "boundary_observations": self.boundary_observations,
                "photometric_residual_observations": (
                    self.photometric_residual_observations
                ),
                "refinement_events": self.refinement_events,
                "capped_gaussians": self.capped_gaussians,
            },
            "last_refinement": self.last_refinement,
        }

    @classmethod
    def from_state_dict(
        cls, payload: Mapping[str, Any]
    ) -> "BoundaryResidualState":
        if int(payload.get("schema_version", -1)) != cls.schema_version:
            raise ValueError("unsupported boundary/residual state schema")
        state = cls(
            policy=BoundaryResidualPolicy.from_mapping(payload["policy"])
        )
        counters = payload.get("counters", {})
        for name in (
            "observation_events",
            "boundary_observations",
            "photometric_residual_observations",
            "refinement_events",
            "capped_gaussians",
        ):
            value = int(counters.get(name, 0))
            if value < 0:
                raise ValueError("boundary/residual counters must be non-negative")
            setattr(state, name, value)
        state.last_refinement = payload.get("last_refinement")
        return state

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "policy": asdict(self.policy),
            "counters": {
                "observation_events": self.observation_events,
                "boundary_observations": self.boundary_observations,
                "photometric_residual_observations": (
                    self.photometric_residual_observations
                ),
                "refinement_events": self.refinement_events,
                "capped_gaussians": self.capped_gaussians,
            },
            "last_refinement": self.last_refinement,
        }


def binary_boundary_band(
    mask: torch.Tensor, *, radius_pixels: int, threshold: float = 0.5
) -> torch.Tensor:
    """Return the two-sided binary morphological boundary band."""
    if radius_pixels < 1:
        raise ValueError("radius_pixels must be positive")
    if mask.ndim not in (2, 3, 4):
        raise ValueError("mask must have two to four dimensions")
    original_shape = mask.shape
    if mask.ndim == 2:
        values = mask[None, None]
    elif mask.ndim == 3:
        values = mask[:, None]
    else:
        if mask.shape[1] != 1:
            raise ValueError("four-dimensional masks must have one channel")
        values = mask
    binary = (values > threshold).to(torch.float32)
    kernel = radius_pixels * 2 + 1
    dilated = F.max_pool2d(
        binary, kernel_size=kernel, stride=1, padding=radius_pixels
    )
    inverse = F.pad(
        1.0 - binary,
        (radius_pixels,) * 4,
        mode="constant",
        value=1.0,
    )
    eroded = 1.0 - F.max_pool2d(
        inverse,
        kernel_size=kernel,
        stride=1,
        padding=0,
    )
    band = (dilated > 0.5) & ~(eroded > 0.5)
    if len(original_shape) == 2:
        return band[0, 0]
    if len(original_shape) == 3:
        return band[:, 0]
    return band


def photometric_residual_map(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Compute detached per-pixel channel-mean absolute RGB residual."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must have shape [H, W, 3]")
    return (prediction.detach() - target.detach()).abs().mean(dim=-1)


def sample_projected_centers(
    *,
    means2d: torch.Tensor,
    radii: torch.Tensor,
    boundary_map: torch.Tensor,
    residual_map: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Sample diagnostic maps at visible in-image Gaussian centers."""
    if means2d.ndim != 2 or means2d.shape[-1] != 2:
        raise ValueError("means2d must have shape [N, 2]")
    if radii.ndim != 1 or radii.shape[0] != means2d.shape[0]:
        raise ValueError("radii must align with means2d")
    if boundary_map.ndim != 2 or residual_map.shape != boundary_map.shape:
        raise ValueError("diagnostic maps must share shape [H, W]")
    if means2d.device != radii.device:
        raise ValueError("means2d and radii must share a device")
    if boundary_map.device != means2d.device or residual_map.device != means2d.device:
        raise ValueError("diagnostic maps and projections must share a device")

    height, width = boundary_map.shape
    x = means2d[:, 0]
    y = means2d[:, 1]
    valid = (
        (radii > 0)
        & torch.isfinite(x)
        & torch.isfinite(y)
        & (x >= 0)
        & (x <= width - 1)
        & (y >= 0)
        & (y <= height - 1)
    )
    indices = torch.where(valid)[0]
    if indices.numel() == 0:
        empty = means2d.new_empty((0,), dtype=torch.float32)
        return {
            "indices": indices,
            "boundary": empty,
            "photometric_residual": empty.clone(),
        }
    pixel_x = torch.floor(x[indices] + 0.5).to(torch.long)
    pixel_y = torch.floor(y[indices] + 0.5).to(torch.long)
    return {
        "indices": indices,
        "boundary": boundary_map[pixel_y, pixel_x].to(torch.float32),
        "photometric_residual": residual_map[pixel_y, pixel_x].to(
            torch.float32
        ),
    }


def boundary_residual_order(
    *,
    candidate_indices: torch.Tensor,
    boundary_mean: torch.Tensor,
    boundary_count: torch.Tensor,
    residual_mean: torch.Tensor,
    residual_count: torch.Tensor,
    screen_grad: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen stable lexicographic D2 candidate ordering."""
    vectors = {
        "boundary_mean": boundary_mean,
        "boundary_count": boundary_count,
        "residual_mean": residual_mean,
        "residual_count": residual_count,
        "screen_grad": screen_grad,
    }
    if candidate_indices.ndim != 1 or candidate_indices.dtype != torch.long:
        raise ValueError("candidate_indices must be a one-dimensional long tensor")
    for name, value in vectors.items():
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if value.device != candidate_indices.device:
            raise ValueError(f"{name} must share candidate_indices device")
        if candidate_indices.numel() and int(candidate_indices.max()) >= value.numel():
            raise IndexError(f"candidate index exceeds {name}")

    ranked = torch.sort(candidate_indices).values

    def stable_descending(values: torch.Tensor) -> None:
        nonlocal ranked
        order = torch.argsort(values[ranked], descending=True, stable=True)
        ranked = ranked[order]

    stable_descending(torch.nan_to_num(screen_grad, nan=-torch.inf))
    stable_descending(torch.nan_to_num(residual_mean, nan=-torch.inf))
    stable_descending((residual_count > 0).to(torch.long))
    stable_descending(torch.nan_to_num(boundary_mean, nan=-torch.inf))
    stable_descending((boundary_count > 0).to(torch.long))
    return ranked


def apply_boundary_scale_cap(
    *,
    log_scales: torch.Tensor,
    boundary_mean: torch.Tensor,
    boundary_count: torch.Tensor,
    maximum_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly cap boundary-observed Gaussian axes, preserving anisotropy."""
    if log_scales.ndim != 2 or log_scales.shape[-1] != 3:
        raise ValueError("log_scales must have shape [N, 3]")
    if boundary_mean.shape != log_scales.shape[:1]:
        raise ValueError("boundary_mean must align with log_scales")
    if boundary_count.shape != log_scales.shape[:1]:
        raise ValueError("boundary_count must align with log_scales")
    if not math.isfinite(maximum_scale) or maximum_scale <= 0:
        raise ValueError("maximum_scale must be finite and positive")
    activated = torch.exp(log_scales)
    current_max = activated.max(dim=-1).values
    boundary_observed = (
        (boundary_count > 0)
        & torch.isfinite(boundary_mean)
        & (boundary_mean > 0)
    )
    capped = boundary_observed & (current_max > maximum_scale)
    ratio = torch.ones_like(current_max)
    ratio[capped] = maximum_scale / current_max[capped]
    updated = log_scales + torch.log(ratio).unsqueeze(-1)
    return updated, capped


def validate_a2_d2_contract(payload: Mapping[str, Any]) -> None:
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported A2 D2 schema version")
    if payload.get("task_id") != EXPECTED_TASK_ID:
        raise ValueError("unexpected A2 task ID")
    if payload.get("audit_version") != EXPECTED_AUDIT_VERSION:
        raise ValueError("unexpected A2 D2 audit version")

    depends_on = payload["depends_on"]
    if depends_on.get("d1_formal_status") != "done":
        raise ValueError("D2 requires the completed D1 formal result")
    summary_sha = str(depends_on.get("d1_formal_summary_sha256", ""))
    if len(summary_sha) != 64:
        raise ValueError("D1 formal summary SHA-256 is required")

    paired = payload["paired_intervention"]
    if tuple(paired.get("order", ())) != (
        "d1-actor-quota",
        "d2-boundary-residual",
    ):
        raise ValueError("D2 paired order must be D1 then D2")
    inherited = paired["d1_inherited_exactly"]
    expected_inherited = {
        "rigid_densify_grad_threshold": 0.00025,
        "minimum_initial_multiplier": 0.5,
        "minimum_absolute_floor": 1,
        "maximum_initial_multiplier": 2.4,
        "maximum_absolute_cap": 12000,
        "below_threshold_policy": "only_to_recover_minimum",
        "split_parent_cost": "n_split_samples",
        "clone_parent_cost": 1,
        "native_cull_policy": "unchanged",
        "background_policy": "native_unchanged",
    }
    if dict(inherited) != expected_inherited:
        raise ValueError("D2 must inherit the complete D1 budget policy")

    attribution = payload["training_attribution"]
    if attribution.get("mask_source") != "image_infos.dynamic_masks":
        raise ValueError("D2 must use the frozen dynamic mask source")
    if attribution.get("mask_required") is not True:
        raise ValueError("D2 dynamic masks must be required")
    if attribution["boundary_band"].get("operator") != (
        "binary_morphological_gradient"
    ):
        raise ValueError("unexpected D2 boundary operator")
    if attribution["projection"].get("sampling") != (
        "nearest_projected_center"
    ):
        raise ValueError("unexpected D2 projection sampling")
    BoundaryResidualPolicy.from_contract(payload)

    ordering = payload["ordering"]
    if tuple(ordering.get("keys", ())) != ORDERING_KEYS:
        raise ValueError("D2 ordering key drift")
    if ordering.get("deterministic") is not True:
        raise ValueError("D2 ordering must be deterministic")
    if ordering.get("quota_accounting") != "inherited_d1_exactly":
        raise ValueError("D2 quota accounting must remain D1-exact")

    scale_cap = payload["boundary_scale_cap"]
    expected_scale_fields = {
        "geometry_classification": "use_pre_cap_scales",
        "maximum_scale": "native_densify_size_threshold_times_scene_scale",
        "transform": "uniform_axis_rescale_preserving_anisotropy",
        "optimizer_state": "zero_exp_avg_and_exp_avg_sq_for_capped_rows",
        "extra_rng_draws": False,
    }
    for key, expected in expected_scale_fields.items():
        if scale_cap.get(key) != expected:
            raise ValueError(f"D2 scale-cap contract drift: {key}")

    forbidden = set(payload["scope_boundary"].get("forbidden_in_d2", ()))
    expected_forbidden = {
        "depth_residual_ordering",
        "normal_residual_ordering",
        "lidar_distance_weighting",
        "visibility_weighting",
        "provenance_aware_pruning",
        "non_native_cull_policy",
        "background_intervention",
    }
    if forbidden != expected_forbidden:
        raise ValueError("D2 forbidden feature boundary drift")

    module_off = payload["module_off_equivalence"]
    if module_off.get("reference") != "d1-actor-quota":
        raise ValueError("D2 module-off reference must be D1")
    if module_off.get("require_native_tensor_bitwise_equality") is not True:
        raise ValueError("D2 module-off must require bitwise equality")
    if module_off.get("forbid_extra_rng_draws") is not True:
        raise ValueError("D2 module-off must not draw random numbers")

    smoke = payload["paired_smoke"]
    if int(smoke.get("num_iters", 0)) < 1000:
        raise ValueError("D2 paired smoke must exercise refinement")
    if smoke.get("formal_protocol_allowed_before_pass") is not False:
        raise ValueError("D2 formal protocol must wait for paired smoke")

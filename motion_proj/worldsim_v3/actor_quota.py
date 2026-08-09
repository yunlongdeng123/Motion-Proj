from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch


SCHEMA_VERSION = 1
EXPECTED_TASK_ID = "WS-V3-A2-ACTOR-DENSIFY-01"
EXPECTED_AUDIT_VERSION = "A2-D1-v1"


@dataclass(frozen=True)
class ActorQuotaPolicy:
    densify_grad_threshold: float
    minimum_initial_multiplier: float
    minimum_absolute_floor: int
    maximum_initial_multiplier: float
    maximum_absolute_cap: int
    ranking: str = "screen_grad_desc_then_gaussian_index"
    below_threshold_policy: str = "only_to_recover_minimum"
    budget_policy: str = "gradient_ranked_prefix"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ActorQuotaPolicy":
        policy = cls(
            densify_grad_threshold=float(payload["densify_grad_threshold"]),
            minimum_initial_multiplier=float(
                payload["minimum_initial_multiplier"]
            ),
            minimum_absolute_floor=int(payload["minimum_absolute_floor"]),
            maximum_initial_multiplier=float(
                payload["maximum_initial_multiplier"]
            ),
            maximum_absolute_cap=int(payload["maximum_absolute_cap"]),
            ranking=str(
                payload.get(
                    "ranking", "screen_grad_desc_then_gaussian_index"
                )
            ),
            below_threshold_policy=str(
                payload.get(
                    "below_threshold_policy", "only_to_recover_minimum"
                )
            ),
            budget_policy=str(
                payload.get("budget_policy", "gradient_ranked_prefix")
            ),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if not math.isfinite(self.densify_grad_threshold):
            raise ValueError("densify_grad_threshold must be finite")
        if self.densify_grad_threshold < 0:
            raise ValueError("densify_grad_threshold must be non-negative")
        if not 0 < self.minimum_initial_multiplier <= 1:
            raise ValueError(
                "minimum_initial_multiplier must be in the interval (0, 1]"
            )
        if self.minimum_absolute_floor < 1:
            raise ValueError("minimum_absolute_floor must be positive")
        if self.maximum_initial_multiplier < 1:
            raise ValueError("maximum_initial_multiplier must be at least one")
        if self.maximum_absolute_cap < self.minimum_absolute_floor:
            raise ValueError(
                "maximum_absolute_cap must not be below the minimum floor"
            )
        if self.ranking != "screen_grad_desc_then_gaussian_index":
            raise ValueError("unsupported actor quota ranking")
        if self.below_threshold_policy != "only_to_recover_minimum":
            raise ValueError("unsupported below-threshold policy")
        if self.budget_policy != "gradient_ranked_prefix":
            raise ValueError("unsupported actor quota budget policy")


class ActorQuotaController:
    def __init__(
        self,
        *,
        policy: ActorQuotaPolicy,
        initial_counts: torch.Tensor,
        minimum_counts: torch.Tensor,
        maximum_counts: torch.Tensor,
    ) -> None:
        self.policy = policy
        self.initial_counts = initial_counts.to(dtype=torch.long)
        self.minimum_counts = minimum_counts.to(dtype=torch.long)
        self.maximum_counts = maximum_counts.to(dtype=torch.long)
        self.events = 0
        self.accepted_split_parents = 0
        self.accepted_clone_parents = 0
        self.accepted_children = 0
        self.admitted_below_threshold_parents = 0
        self.rejected_by_maximum_parents = 0
        self.last_decision: dict[str, Any] | None = None
        self._validate_quota_tensors()

    @classmethod
    def initialize(
        cls, *, actor_ids: torch.Tensor, policy: ActorQuotaPolicy
    ) -> "ActorQuotaController":
        actor_ids = _validate_actor_ids(actor_ids, require_all_present=True)
        actor_count = int(actor_ids.max().item()) + 1
        initial = torch.bincount(actor_ids, minlength=actor_count)
        minimum = torch.ceil(
            initial.to(torch.float64) * policy.minimum_initial_multiplier
        ).to(torch.long)
        minimum = torch.maximum(
            minimum,
            torch.full_like(minimum, policy.minimum_absolute_floor),
        )
        maximum = torch.ceil(
            initial.to(torch.float64) * policy.maximum_initial_multiplier
        ).to(torch.long)
        maximum = torch.minimum(
            maximum,
            torch.full_like(maximum, policy.maximum_absolute_cap),
        )
        maximum = torch.maximum(maximum, minimum)
        return cls(
            policy=policy,
            initial_counts=initial,
            minimum_counts=minimum,
            maximum_counts=maximum,
        )

    @classmethod
    def from_state_dict(
        cls, payload: Mapping[str, Any], *, device: torch.device | str
    ) -> "ActorQuotaController":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported actor quota schema version")
        controller = cls(
            policy=ActorQuotaPolicy.from_mapping(payload["policy"]),
            initial_counts=payload["initial_counts"].to(device=device),
            minimum_counts=payload["minimum_counts"].to(device=device),
            maximum_counts=payload["maximum_counts"].to(device=device),
        )
        counters = payload.get("counters", {})
        for name in (
            "events",
            "accepted_split_parents",
            "accepted_clone_parents",
            "accepted_children",
            "admitted_below_threshold_parents",
            "rejected_by_maximum_parents",
        ):
            setattr(controller, name, int(counters.get(name, 0)))
        controller.last_decision = payload.get("last_decision")
        return controller

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "policy": asdict(self.policy),
            "initial_counts": self.initial_counts.detach().clone(),
            "minimum_counts": self.minimum_counts.detach().clone(),
            "maximum_counts": self.maximum_counts.detach().clone(),
            "counters": {
                "events": self.events,
                "accepted_split_parents": self.accepted_split_parents,
                "accepted_clone_parents": self.accepted_clone_parents,
                "accepted_children": self.accepted_children,
                "admitted_below_threshold_parents": (
                    self.admitted_below_threshold_parents
                ),
                "rejected_by_maximum_parents": (
                    self.rejected_by_maximum_parents
                ),
            },
            "last_decision": self.last_decision,
        }

    def select_densification(
        self,
        *,
        actor_ids: torch.Tensor,
        average_gradients: torch.Tensor,
        visibility_counts: torch.Tensor,
        split_geometry: torch.Tensor,
        clone_geometry: torch.Tensor,
        split_children: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        actor_ids = self._validate_current_actor_ids(actor_ids)
        if split_children < 1:
            raise ValueError("split_children must be positive")
        _validate_vector("average_gradients", average_gradients, actor_ids)
        _validate_vector("visibility_counts", visibility_counts, actor_ids)
        _validate_vector("split_geometry", split_geometry, actor_ids)
        _validate_vector("clone_geometry", clone_geometry, actor_ids)
        if split_geometry.dtype is not torch.bool:
            raise TypeError("split_geometry must be boolean")
        if clone_geometry.dtype is not torch.bool:
            raise TypeError("clone_geometry must be boolean")
        finite_gradients = average_gradients[torch.isfinite(average_gradients)]
        if torch.any(finite_gradients < 0):
            raise ValueError("average_gradients must be non-negative")

        current_counts = torch.bincount(
            actor_ids, minlength=self.initial_counts.numel()
        )
        selected_splits = torch.zeros_like(split_geometry)
        selected_clones = torch.zeros_like(clone_geometry)
        observed = (
            torch.isfinite(average_gradients)
            & (visibility_counts > 0)
        )
        actor_rows: list[dict[str, Any]] = []

        for actor_id in range(self.initial_counts.numel()):
            structural = (split_geometry | clone_geometry) & observed
            candidate_indices = torch.where(
                structural & (actor_ids == actor_id)
            )[0]
            current = int(current_counts[actor_id].item())
            minimum = int(self.minimum_counts[actor_id].item())
            maximum = int(self.maximum_counts[actor_id].item())
            capacity = max(maximum - current, 0)
            deficit = max(minimum - current, 0)

            if candidate_indices.numel() == 0:
                actor_rows.append(
                    {
                        "actor_id": actor_id,
                        "current": current,
                        "minimum": minimum,
                        "maximum": maximum,
                        "capacity": capacity,
                        "deficit": deficit,
                        "candidate_parents": 0,
                        "threshold_parents": 0,
                        "accepted_split_parents": 0,
                        "accepted_clone_parents": 0,
                        "accepted_children": 0,
                        "admitted_below_threshold_parents": 0,
                        "rejected_by_maximum_parents": 0,
                    }
                )
                continue

            scores = average_gradients[candidate_indices]
            order = torch.argsort(scores, descending=True, stable=True)
            ranked_indices = candidate_indices[order]
            ranked_scores = average_gradients[ranked_indices]
            ranked_costs = (
                split_geometry[ranked_indices].to(torch.long)
                * split_children
                + clone_geometry[ranked_indices].to(torch.long)
            )
            thresholded = (
                ranked_scores > self.policy.densify_grad_threshold
            )
            mandatory = torch.zeros_like(thresholded)
            if deficit > 0:
                cumulative_all = torch.cumsum(ranked_costs, dim=0)
                crossing = int(
                    torch.searchsorted(
                        cumulative_all,
                        torch.tensor(
                            deficit,
                            device=cumulative_all.device,
                            dtype=cumulative_all.dtype,
                        ),
                    ).item()
                )
                mandatory[: min(crossing + 1, mandatory.numel())] = True

            eligible = thresholded | mandatory
            eligible_indices = ranked_indices[eligible]
            eligible_costs = ranked_costs[eligible]
            accepted = torch.zeros_like(eligible_costs, dtype=torch.bool)
            if capacity > 0 and eligible_costs.numel() > 0:
                accepted = torch.cumsum(eligible_costs, dim=0) <= capacity
            accepted_indices = eligible_indices[accepted]
            selected_splits[accepted_indices] = split_geometry[accepted_indices]
            selected_clones[accepted_indices] = clone_geometry[accepted_indices]

            below_threshold = int(
                (mandatory[eligible] & ~thresholded[eligible] & accepted)
                .sum()
                .item()
            )
            accepted_split_count = int(
                selected_splits[accepted_indices].sum().item()
            )
            accepted_clone_count = int(
                selected_clones[accepted_indices].sum().item()
            )
            accepted_children = int(eligible_costs[accepted].sum().item())
            rejected_by_maximum = int(
                eligible.sum().item() - accepted.sum().item()
            )

            actor_rows.append(
                {
                    "actor_id": actor_id,
                    "current": current,
                    "minimum": minimum,
                    "maximum": maximum,
                    "capacity": capacity,
                    "deficit": deficit,
                    "candidate_parents": int(candidate_indices.numel()),
                    "threshold_parents": int(thresholded.sum().item()),
                    "accepted_split_parents": accepted_split_count,
                    "accepted_clone_parents": accepted_clone_count,
                    "accepted_children": accepted_children,
                    "admitted_below_threshold_parents": below_threshold,
                    "rejected_by_maximum_parents": rejected_by_maximum,
                }
            )

        decision = {
            "event": self.events + 1,
            "policy": asdict(self.policy),
            "current_total": int(current_counts.sum().item()),
            "minimum_total": int(self.minimum_counts.sum().item()),
            "maximum_total": int(self.maximum_counts.sum().item()),
            "accepted_split_parents": int(selected_splits.sum().item()),
            "accepted_clone_parents": int(selected_clones.sum().item()),
            "accepted_children": sum(
                int(row["accepted_children"]) for row in actor_rows
            ),
            "admitted_below_threshold_parents": sum(
                int(row["admitted_below_threshold_parents"])
                for row in actor_rows
            ),
            "rejected_by_maximum_parents": sum(
                int(row["rejected_by_maximum_parents"])
                for row in actor_rows
            ),
            "actors": actor_rows,
        }
        self.events += 1
        self.accepted_split_parents += decision["accepted_split_parents"]
        self.accepted_clone_parents += decision["accepted_clone_parents"]
        self.accepted_children += decision["accepted_children"]
        self.admitted_below_threshold_parents += decision[
            "admitted_below_threshold_parents"
        ]
        self.rejected_by_maximum_parents += decision[
            "rejected_by_maximum_parents"
        ]
        self.last_decision = decision
        return selected_splits, selected_clones, decision

    def summary(self, *, actor_ids: torch.Tensor) -> dict[str, Any]:
        actor_ids = self._validate_current_actor_ids(actor_ids)
        current = torch.bincount(
            actor_ids, minlength=self.initial_counts.numel()
        )
        actors = []
        for actor_id in range(self.initial_counts.numel()):
            actors.append(
                {
                    "actor_id": actor_id,
                    "initial": int(self.initial_counts[actor_id].item()),
                    "minimum": int(self.minimum_counts[actor_id].item()),
                    "maximum": int(self.maximum_counts[actor_id].item()),
                    "current": int(current[actor_id].item()),
                }
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "policy": asdict(self.policy),
            "actor_count": self.initial_counts.numel(),
            "initial_total": int(self.initial_counts.sum().item()),
            "minimum_total": int(self.minimum_counts.sum().item()),
            "maximum_total": int(self.maximum_counts.sum().item()),
            "current_total": int(current.sum().item()),
            "counters": self.state_dict()["counters"],
            "actors": actors,
        }

    def validate(self, *, actor_ids: torch.Tensor) -> None:
        actor_ids = self._validate_current_actor_ids(actor_ids)
        counts = torch.bincount(
            actor_ids, minlength=self.initial_counts.numel()
        )
        if torch.any(counts > self.maximum_counts):
            raise ValueError("current actor counts exceed frozen maximum quota")

    def _validate_quota_tensors(self) -> None:
        shapes = {
            tuple(self.initial_counts.shape),
            tuple(self.minimum_counts.shape),
            tuple(self.maximum_counts.shape),
        }
        if len(shapes) != 1 or self.initial_counts.ndim != 1:
            raise ValueError("actor quota tensors must share one vector shape")
        if self.initial_counts.numel() == 0:
            raise ValueError("actor quota tensors must not be empty")
        devices = {
            self.initial_counts.device,
            self.minimum_counts.device,
            self.maximum_counts.device,
        }
        if len(devices) != 1:
            raise ValueError("actor quota tensors must share one device")
        if torch.any(self.initial_counts <= 0):
            raise ValueError("initial actor counts must be positive")
        if torch.any(self.minimum_counts <= 0):
            raise ValueError("minimum actor quotas must be positive")
        if torch.any(self.minimum_counts > self.maximum_counts):
            raise ValueError("minimum actor quota exceeds maximum")

    def _validate_current_actor_ids(
        self, actor_ids: torch.Tensor
    ) -> torch.Tensor:
        actor_ids = _validate_actor_ids(actor_ids, require_all_present=False)
        if actor_ids.numel() and int(actor_ids.max().item()) >= self.initial_counts.numel():
            raise ValueError("current actor IDs exceed the frozen actor set")
        return actor_ids


def _validate_actor_ids(
    actor_ids: torch.Tensor, *, require_all_present: bool
) -> torch.Tensor:
    if actor_ids.ndim != 1:
        raise ValueError("actor_ids must be one-dimensional")
    if actor_ids.dtype != torch.long:
        raise TypeError("actor_ids must use torch.long")
    if actor_ids.numel() == 0:
        raise ValueError("actor_ids must not be empty")
    if torch.any(actor_ids < 0):
        raise ValueError("actor_ids must be non-negative")
    if require_all_present:
        unique = torch.unique(actor_ids, sorted=True)
        expected = torch.arange(
            int(unique[-1].item()) + 1,
            device=unique.device,
            dtype=unique.dtype,
        )
        if not torch.equal(unique, expected):
            raise ValueError("initial actor IDs must be contiguous")
    return actor_ids


def _validate_vector(
    name: str, value: torch.Tensor, actor_ids: torch.Tensor
) -> None:
    if value.ndim != 1 or value.shape != actor_ids.shape:
        raise ValueError(f"{name} must match actor_ids")
    if value.device != actor_ids.device:
        raise ValueError(f"{name} must share actor_ids device")


def validate_a2_d1_contract(payload: Mapping[str, Any]) -> None:
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported A2 D1 schema version")
    if payload.get("task_id") != EXPECTED_TASK_ID:
        raise ValueError("unexpected A2 task ID")
    if payload.get("audit_version") != EXPECTED_AUDIT_VERSION:
        raise ValueError("unexpected A2 D1 audit version")

    actor_densification = payload["actor_densification"]
    if actor_densification.get("enabled") is not True:
        raise ValueError("A2 D1 actor densification must be enabled")
    background = actor_densification["background"]
    if background.get("policy") != "native_unchanged":
        raise ValueError("A2 D1 must preserve native background densification")
    if float(background.get("densify_grad_threshold", -1)) != 0.0005:
        raise ValueError("unexpected native background gradient threshold")
    if background.get("quota") != "disabled":
        raise ValueError("A2 D1 must not apply quota to background Gaussians")

    rigid = actor_densification["rigid_nodes"]
    policy = ActorQuotaPolicy.from_mapping(
        {
            "densify_grad_threshold": rigid["densify_grad_threshold"],
            **rigid["quota"],
        }
    )
    expected_policy = ActorQuotaPolicy(
        densify_grad_threshold=0.00025,
        minimum_initial_multiplier=0.5,
        minimum_absolute_floor=1,
        maximum_initial_multiplier=2.4,
        maximum_absolute_cap=12000,
    )
    if policy != expected_policy:
        raise ValueError("A2 D1 actor quota policy drift")

    scope = payload["scope_boundary"]
    if scope.get("stage") != "d1_threshold_and_actor_quota_only":
        raise ValueError("unexpected A2 D1 scope stage")
    forbidden = tuple(scope.get("forbidden_in_d1", ()))
    expected_forbidden = {
        "boundary_weighting",
        "photometric_residual_weighting",
        "depth_residual_weighting",
        "normal_residual_weighting",
        "gaussian_scale_cap",
        "lidar_distance_weighting",
        "visibility_weighting",
    }
    if set(forbidden) != expected_forbidden:
        raise ValueError("A2 D1 forbidden feature boundary drift")

    module_off = payload["module_off_equivalence"]
    if module_off.get("enabled_value") is not False:
        raise ValueError("A2 D1 module-off value must be false")
    if module_off.get("require_native_tensor_bitwise_equality") is not True:
        raise ValueError("A2 D1 must require native tensor bitwise equality")
    if module_off.get("forbid_extra_rng_draws") is not True:
        raise ValueError("A2 D1 module-off path must forbid extra RNG draws")

    smoke = payload["paired_smoke"]
    if int(smoke.get("num_iters", 0)) < 1000:
        raise ValueError("A2 D1 paired smoke must exercise refinement")
    if smoke.get("formal_run_allowed_before_pass") is not False:
        raise ValueError("A2 D1 formal run must wait for paired smoke")

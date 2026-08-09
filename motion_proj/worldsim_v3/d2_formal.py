"""A2-D2 formal 的冻结合同与 D1 exact-alias 边界。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from motion_proj.worldsim_v3.a2_formal import QUALITY_DIRECTIONS


TASK_ID = "WS-V3-A2-ACTOR-DENSIFY-01"
AUDIT_VERSION = "A2-D2-FORMAL-v1"
FORMAL_ORDER = (
    "d1-actor-quota-exact-alias",
    "d2-boundary-residual",
)
CANDIDATE_STEPS = [5_000, 10_000, 15_000, 20_000, 25_000, 30_000]
D2_FORBIDDEN = {
    "depth_residual_ordering",
    "normal_residual_ordering",
    "lidar_distance_weighting",
    "visibility_weighting",
    "provenance_aware_pruning",
    "non_native_cull_policy",
    "background_intervention",
}


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def _sha256(value: Any, detail: str) -> None:
    text = str(value or "")
    _require(
        len(text) == 64
        and all(character in "0123456789abcdef" for character in text),
        detail,
    )


def validate_a2_d2_formal_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed when any preregistered D2 formal field drifts."""

    _require(contract.get("schema_version") == 1, "unsupported formal schema")
    _require(contract.get("task_id") == TASK_ID, "formal task ID drift")
    _require(
        contract.get("audit_version") == AUDIT_VERSION,
        "formal audit version drift",
    )

    depends_on = contract.get("depends_on") or {}
    for name in (
        "d2_protocol_sha256",
        "paired_smoke_summary_sha256",
        "paired_smoke_manifest_sha256",
    ):
        _sha256(depends_on.get(name), f"missing formal dependency SHA: {name}")

    reference = contract.get("d1_reference_alias") or {}
    _require(
        reference.get("mode") == "immutable_exact_alias_no_retraining",
        "D1 reference must be an immutable exact alias",
    )
    _sha256(reference.get("summary_sha256"), "D1 reference summary SHA drift")
    _sha256(
        reference.get("initialization_provenance_sha256"),
        "D1 initialization provenance SHA drift",
    )
    _sha256(
        reference.get("fixed_checkpoint_sha256"),
        "D1 fixed checkpoint SHA drift",
    )
    _require(
        reference.get("fixed_rigid_gaussians") == 105_412,
        "D1 fixed RigidNodes target drift",
    )
    _require(
        reference.get("fixed_background_gaussians") == 1_201_057,
        "D1 fixed Background count drift",
    )
    _require(
        reference.get("checkpoint_must_remain_unchanged") is True,
        "D1 alias checkpoint must remain immutable",
    )

    design = contract.get("paired_design") or {}
    _require(design.get("scene") == "scene-0230", "formal scene drift")
    _require(design.get("scene_index") == 179, "formal scene index drift")
    _require(design.get("seed") == 0, "formal seed drift")
    _require(tuple(design.get("order") or ()) == FORMAL_ORDER, "arm order drift")
    _require(
        design.get("trained_arms") == ["d2-boundary-residual"],
        "D2 formal must train exactly one new arm",
    )
    _require(
        design.get("exact_alias_arms") == ["d1-actor-quota"],
        "D1 formal arm must remain an exact alias",
    )
    _require(design.get("num_iters") == 30_000, "fixed-step budget drift")
    _require(
        design.get("checkpoint_interval") == 5_000,
        "checkpoint interval drift",
    )
    _require(design.get("test_image_stride") == 10, "held-out split drift")
    mutable = (
        design.get("paired_config_normalization") or {}
    ).get("mutable_fields")
    _require(
        mutable
        == [
            "worldsim_v3.variant",
            "model.RigidNodes.ctrl.a2_actor_quota.ranking",
            "model.RigidNodes.ctrl.a2_boundary_residual.enabled",
        ],
        "paired config normalization drift",
    )

    matched = contract.get("matched_gaussian_budget") or {}
    _require(matched.get("scope") == "RigidNodes", "matched scope drift")
    _require(
        matched.get("target") == "d1_fixed_30000_rigid_gaussians"
        and matched.get("target_count") == 105_412,
        "matched D1 target drift",
    )
    _require(
        matched.get("candidate_steps") == CANDIDATE_STEPS,
        "matched candidate grid drift",
    )
    _require(
        matched.get("selection") == "minimum_absolute_gap_then_earliest_step",
        "matched selection drift",
    )
    _require(
        float(matched.get("maximum_relative_gap", -1.0)) == 0.02,
        "matched relative-gap drift",
    )
    _require(
        matched.get("failure_status") == "ABSTAIN_BUDGET_NOT_MATCHED",
        "matched failure status drift",
    )
    _require(
        matched.get("posthoc_pruning") is False
        and matched.get("retraining") is False
        and matched.get("quota_retuning") is False
        and matched.get("checkpoint_mutation") is False,
        "matched view must remain read-only",
    )

    evaluation = contract.get("evaluation") or {}
    _require(evaluation.get("formal_full_split") is True, "formal split drift")
    _require(
        tuple(evaluation.get("actor_roles") or ())
        == ("high-support", "boundary-support"),
        "actor role drift",
    )
    _require(
        evaluation.get("non_target_definition")
        == "complement_of_selected_actor_counterfactual_union",
        "non-target definition drift",
    )
    _require(
        evaluation.get("quality_directions") == QUALITY_DIRECTIONS,
        "quality-axis contract drift",
    )
    pareto = evaluation.get("pareto") or {}
    _require(
        pareto.get("baseline_role") == "d1"
        and pareto.get("candidate_role") == "d2"
        and pareto.get("numeric_tolerance") == "none",
        "D1/D2 Pareto role drift",
    )

    resources = contract.get("resource_contract") or {}
    _require(
        resources.get("execution") == "one_new_arm_then_read_only_evaluation",
        "formal execution drift",
    )
    _require(resources.get("gpu_count") == 1, "GPU count drift")
    _require(resources.get("minimum_free_disk_gib") == 40, "disk gate drift")
    _require(resources.get("gpu_idle_max_mib") == 2048, "GPU idle gate drift")
    _require(resources.get("memory_stop_fraction") == 0.90, "memory gate drift")
    _require(
        resources.get("memory_stop_consecutive_samples") == 2,
        "memory sample gate drift",
    )

    scope = contract.get("scope_boundary") or {}
    _require(
        scope.get("stage")
        == "d2_boundary_residual_ordering_and_boundary_scale_cap_only",
        "D2 formal scope drift",
    )
    _require(set(scope.get("forbidden") or ()) == D2_FORBIDDEN, "D2 scope drift")

    gate = contract.get("formal_gate") or {}
    _require(gate.get("fixed_step_required") is True, "fixed-step gate drift")
    _require(gate.get("matched_budget_required") is True, "matched gate drift")
    _require(
        gate.get("checkpoint_immutability_required") is True,
        "checkpoint immutability gate drift",
    )
    _require(
        gate.get("d3_not_automatically_unlocked") is True,
        "D3 must remain conditional",
    )

"""Machine-readable contracts for paired counterfactual driving evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


MANIFEST_SCHEMA = "worldsim_v75_cfbench_manifest_v1"
RESULT_SCHEMA = "worldsim_v75_cfbench_result_v1"

INTERVENTION_FAMILIES = (
    "actor_speed_change",
    "actor_lateral_relocation",
    "actor_removal",
    "actor_insertion",
)

DIMENSIONS = (
    "adherence",
    "physics",
    "environment_preservation",
    "outcome",
    "trajectory_adherence",
    "object_background_preservation",
)

TARGET_ROLES = ("ego", "non_ego")
CASE_STATUSES = ("proposed_cpu_only", "qualified", "rejected")
RESULT_STATUSES = ("completed", "unsupported", "blocked", "failed")
METRIC_STATUSES = ("scored", "not_applicable", "missing", "failed")


class ContractError(ValueError):
    """Raised when a benchmark artifact violates its declared contract."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path} must be a mapping")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractError(f"{path} must be a sequence")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{path} must be a non-empty string")
    return value


def capability_key(case: Mapping[str, Any]) -> str:
    """Return the registry capability key required by a case."""

    target = _mapping(case.get("target"), "case.target")
    intervention = _mapping(case.get("intervention"), "case.intervention")
    role = _nonempty_string(target.get("role"), "case.target.role")
    family = _nonempty_string(intervention.get("family"), "case.intervention.family")
    return f"{role}:{family}"


def validate_case(case: Mapping[str, Any]) -> None:
    _nonempty_string(case.get("case_id"), "case.case_id")
    if case.get("status") not in CASE_STATUSES:
        raise ContractError(f"case.status must be one of {CASE_STATUSES}")

    dataset = _mapping(case.get("dataset"), "case.dataset")
    _nonempty_string(dataset.get("name"), "case.dataset.name")
    _nonempty_string(dataset.get("scene_id"), "case.dataset.scene_id")
    _nonempty_string(dataset.get("root"), "case.dataset.root")

    anchor = _mapping(case.get("anchor"), "case.anchor")
    event_frame = anchor.get("event_frame")
    if not isinstance(event_frame, int) or event_frame < 0:
        raise ContractError("case.anchor.event_frame must be a non-negative integer")
    if not isinstance(anchor.get("pre_frames"), int) or anchor["pre_frames"] < 1:
        raise ContractError("case.anchor.pre_frames must be a positive integer")
    if not isinstance(anchor.get("rollout_frames"), int) or anchor["rollout_frames"] < 2:
        raise ContractError("case.anchor.rollout_frames must be >= 2")
    if not isinstance(anchor.get("clip_frames"), int) or anchor["clip_frames"] < 3:
        raise ContractError("case.anchor.clip_frames must be >= 3")
    if anchor["clip_frames"] != anchor["pre_frames"] + anchor["rollout_frames"]:
        raise ContractError("clip_frames must equal pre_frames + rollout_frames")

    target = _mapping(case.get("target"), "case.target")
    if target.get("role") not in TARGET_ROLES:
        raise ContractError(f"case.target.role must be one of {TARGET_ROLES}")
    _nonempty_string(target.get("entity_id"), "case.target.entity_id")

    intervention = _mapping(case.get("intervention"), "case.intervention")
    family = intervention.get("family")
    if family not in INTERVENTION_FAMILIES:
        raise ContractError(
            f"case.intervention.family must be one of {INTERVENTION_FAMILIES}"
        )
    _nonempty_string(intervention.get("variable"), "case.intervention.variable")
    factual = _mapping(intervention.get("factual"), "case.intervention.factual")
    counterfactual = _mapping(
        intervention.get("counterfactual"), "case.intervention.counterfactual"
    )
    changed = {
        key
        for key in set(factual) | set(counterfactual)
        if factual.get(key) != counterfactual.get(key)
    }
    if changed != {intervention["variable"]}:
        raise ContractError(
            "exactly intervention.variable must differ between factual and "
            f"counterfactual controls; got {sorted(changed)}"
        )
    if family in ("actor_removal", "actor_insertion") and target["role"] != "non_ego":
        raise ContractError(f"{family} requires a non_ego target")

    expected = _mapping(case.get("expected_outcome"), "case.expected_outcome")
    _nonempty_string(expected.get("direction"), "case.expected_outcome.direction")
    affected = _sequence(expected.get("affected_entities"), "case.expected_outcome.affected_entities")
    if target["entity_id"] not in affected:
        raise ContractError("target entity must be listed in affected_entities")
    if expected.get("unchanged_scope") != "all_except_affected_entities":
        raise ContractError("unchanged_scope must enforce counterfactual invariance")

    qualification = _mapping(case.get("qualification"), "case.qualification")
    if qualification.get("state_fixation") is not True:
        raise ContractError("state_fixation must be true")
    if qualification.get("single_variable_change") is not True:
        raise ContractError("single_variable_change must be true")
    if "human_verdict" not in qualification:
        raise ContractError("qualification.human_verdict must be explicit")


def validate_manifest(manifest: Mapping[str, Any], *, require_pilot_size: bool = True) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ContractError(f"schema_version must be {MANIFEST_SCHEMA}")
    benchmark = _mapping(manifest.get("benchmark"), "benchmark")
    if benchmark.get("no_composite_score") is not True:
        raise ContractError("benchmark.no_composite_score must be true")
    dimensions = tuple(_sequence(benchmark.get("dimensions"), "benchmark.dimensions"))
    if dimensions != DIMENSIONS:
        raise ContractError(f"benchmark.dimensions must equal {DIMENSIONS}")

    cases = _sequence(manifest.get("cases"), "cases")
    if require_pilot_size and not 20 <= len(cases) <= 30:
        raise ContractError("pilot manifest must contain 20-30 paired edits")
    ids: set[str] = set()
    counts = {family: 0 for family in INTERVENTION_FAMILIES}
    for raw_case in cases:
        case = _mapping(raw_case, "cases[]")
        validate_case(case)
        case_id = str(case["case_id"])
        if case_id in ids:
            raise ContractError(f"duplicate case_id: {case_id}")
        ids.add(case_id)
        counts[str(case["intervention"]["family"])] += 1
    if require_pilot_size and any(value == 0 for value in counts.values()):
        raise ContractError(f"all intervention families must be represented: {counts}")


def validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ContractError(f"result.schema_version must be {RESULT_SCHEMA}")
    _nonempty_string(result.get("model_id"), "result.model_id")
    _nonempty_string(result.get("case_id"), "result.case_id")
    status = result.get("status")
    if status not in RESULT_STATUSES:
        raise ContractError(f"result.status must be one of {RESULT_STATUSES}")
    metrics = _mapping(result.get("metrics"), "result.metrics")
    if tuple(metrics) != DIMENSIONS:
        raise ContractError(f"result.metrics must contain dimensions in order {DIMENSIONS}")
    for dimension in DIMENSIONS:
        metric = _mapping(metrics[dimension], f"result.metrics.{dimension}")
        metric_status = metric.get("status")
        if metric_status not in METRIC_STATUSES:
            raise ContractError(
                f"result.metrics.{dimension}.status must be one of {METRIC_STATUSES}"
            )
        value = metric.get("value")
        if metric_status == "scored":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ContractError(f"scored {dimension} must have a numeric value")
            if not 0.0 <= float(value) <= 1.0:
                raise ContractError(f"scored {dimension} must be in [0, 1]")
        elif value is not None:
            raise ContractError(f"unscored {dimension} must have value null")
        _nonempty_string(metric.get("evidence"), f"result.metrics.{dimension}.evidence")
    if "composite" in result or "overall_score" in result:
        raise ContractError("pilot results must not contain a composite score")

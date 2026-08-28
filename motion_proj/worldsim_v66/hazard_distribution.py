"""固定预算比较HARP local-action ranking，并保持Actor/hazard proxy分布。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _selected_indices(scores: np.ndarray, base_ids: Sequence[str], count: int) -> np.ndarray:
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), base_ids[index]))
    return np.asarray(order[:count], dtype=np.int64)


def _arm_metrics(
    name: str,
    labels: np.ndarray,
    base_ids: Sequence[str],
    scenes: Sequence[str],
    actor_keys: Sequence[str],
    scores: np.ndarray | None,
    action_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = (
        np.empty(0, dtype=np.int64)
        if scores is None or action_count == 0
        else _selected_indices(scores, base_ids, action_count)
    )
    action = np.zeros(len(labels), dtype=bool)
    action[selected] = True
    source_conflicts = int(np.count_nonzero(labels))
    handled_conflicts = int(np.count_nonzero(labels & action))
    unhandled = source_conflicts - handled_conflicts
    per_scene = []
    for scene in sorted(set(scenes)):
        members = np.asarray([value == scene for value in scenes], dtype=bool)
        conflicts = int(np.count_nonzero(labels[members]))
        handled = int(np.count_nonzero(labels[members] & action[members]))
        per_scene.append(
            {
                "scene": scene,
                "source_conflict_count": conflicts,
                "handled_conflict_count": handled,
                "unhandled_conflict_count": conflicts - handled,
                "conflict_exposure_reduction": handled / conflicts if conflicts else 0.0,
            }
        )
    rows = [
        {
            "base_id": base_ids[index],
            "actor_key": actor_keys[index],
            "scene": scenes[index],
            "arm": name,
            "local_action": "RANK_REPAIR_OR_ABSTAIN" if action[index] else "KEEP_LOCAL_GEOMETRY",
            "actor_retained": True,
        }
        for index in range(len(labels))
    ]
    return {
        "arm": name,
        "actor_state_count": len(labels),
        "unique_actor_count": len(set(actor_keys)),
        "actor_retention": 1.0,
        "actor_removed_count": 0,
        "local_action_count": int(np.count_nonzero(action)),
        "local_action_fraction": float(np.mean(action)),
        "emitted_local_geometry_fraction": float(np.mean(~action)),
        "source_conflict_count": source_conflicts,
        "handled_conflict_count": handled_conflicts,
        "unhandled_conflict_count": unhandled,
        "conflict_exposure_rate_on_original_denominator": unhandled / len(labels),
        "conflict_exposure_reduction": handled_conflicts / source_conflicts,
        "world_scene_yield": len({scene for scene, keep in zip(scenes, ~action) if keep})
        / len(set(scenes)),
        "per_scene": per_scene,
    }, rows


def evaluate_fixed_budget(
    config: Mapping[str, Any], runs_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sources = config["sources"]
    p3c_root = runs_root / str(sources["local_geometry_run"])
    package_root = runs_root / str(sources["harp_bake_run"]) / "package"
    scores = sorted(
        _jsonl(p3c_root / str(sources["local_geometry_scores"])),
        key=lambda row: str(row["base_id"]),
    )
    factors = {
        str(row["base_id"]): row
        for row in _jsonl(package_root / "ARTIFACT_FACTORS.jsonl")
    }
    hazards = {
        str(row["actor_key"]): row
        for row in _jsonl(package_root / "HAZARD_ATTRIBUTES.jsonl")
    }
    base_ids = [str(row["base_id"]) for row in scores]
    scenes = [str(row["scene"]) for row in scores]
    actor_keys = [str(factors[base_id]["actor_key"]) for base_id in base_ids]
    labels = np.asarray([bool(row["local_geometry_conflict"]) for row in scores])
    q0 = np.asarray([float(row["q0_mean"]) for row in scores], dtype=np.float64)
    learned = np.asarray([float(row["p_local_conflict"]) for row in scores], dtype=np.float64)
    deterministic = np.asarray(
        [float(bool(factors[base_id]["deterministic_existence_reasons"])) for base_id in base_ids]
    )
    oracle = labels.astype(np.float64)
    budget = int(np.floor(len(labels) * float(config["action_budget_fraction"])))
    arm_specs = (
        ("N0_NAIVE", None, 0),
        ("Q0_Q0_RANK", q0, budget),
        ("D0_DETERMINISTIC", deterministic, int(np.count_nonzero(deterministic))),
        ("L0_LEARNED_HARP", learned, budget),
        ("O0_ORACLE", oracle, budget),
    )
    arm_metrics = []
    action_rows = []
    for name, ranking, count in arm_specs:
        metrics, rows = _arm_metrics(
            name, labels, base_ids, scenes, actor_keys, ranking, int(count)
        )
        arm_metrics.append(metrics)
        action_rows.extend(rows)

    hazard_arrays = {
        "minimum_ego_center_distance_m": np.asarray(
            [hazards[key]["minimum_ego_center_distance_m"] for key in actor_keys],
            dtype=np.float64,
        ),
        "maximum_closing_speed_mps": np.asarray(
            [hazards[key]["maximum_closing_speed_mps"] for key in actor_keys],
            dtype=np.float64,
        ),
        "maximum_actor_speed_mps": np.asarray(
            [hazards[key]["maximum_actor_speed_mps"] for key in actor_keys],
            dtype=np.float64,
        ),
    }
    hazard_proxy_summary = {
        name: {
            "mean": float(np.mean(values)),
            "p90": float(np.quantile(values, 0.9)),
            "maximum_arm_shift": 0.0,
        }
        for name, values in hazard_arrays.items()
    }
    return {
        "row_count": len(labels),
        "conflict_count": int(np.count_nonzero(labels)),
        "clean_count": int(np.count_nonzero(~labels)),
        "fixed_local_action_budget_count": budget,
        "fixed_local_action_budget_fraction": float(config["action_budget_fraction"]),
        "arms": arm_metrics,
        "hazard_proxy_summary": hazard_proxy_summary,
        "maximum_hazard_proxy_distribution_shift": 0.0,
        "actor_sets_identical_across_arms": True,
        "hazard_attributes_used_for_ranking": False,
        "physical_geometry_mutated": False,
        "estimand": "fixed_budget_unhandled_local_conflict_exposure",
    }, action_rows

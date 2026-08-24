#!/usr/bin/env python3
"""Run the V6.2 P3 projection contract on a real V6.1 evidence fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motion_proj.worldsim_v61.occupancy import FREE, OCCUPIED, UNKNOWN
from motion_proj.worldsim_v62.projection import project_feasible_tristate


def _take(flat_states: np.ndarray, state: int, count: int) -> np.ndarray:
    indices = np.flatnonzero(flat_states == state)
    if indices.size < count:
        raise RuntimeError(f"fixture state {state} has {indices.size} queries, expected at least {count}")
    return indices[:count]


def run(evidence_path: Path, output_path: Path, queries_per_state: int) -> dict[str, object]:
    with np.load(evidence_path, allow_pickle=False) as values:
        states = np.asarray(values["static_semantics"], dtype=np.uint8)
        actor_voxel_count = int(np.asarray(values["actor_voxel_indices"]).shape[0])

    flat = states.reshape(-1)
    selected = np.concatenate(
        [
            _take(flat, int(FREE), queries_per_state),
            _take(flat, int(OCCUPIED), queries_per_state),
            _take(flat, int(UNKNOWN), queries_per_state),
        ]
    )
    selected_states = flat[selected]
    query_count = int(selected.size)
    logits = torch.linspace(-1.25, 1.25, steps=query_count * 3, dtype=torch.float32).reshape(
        query_count, 3
    )
    logits.requires_grad_(True)

    observed_free = torch.from_numpy(selected_states == int(FREE))
    observed_occupied = torch.from_numpy(selected_states == int(OCCUPIED))
    contradiction = torch.zeros(query_count, dtype=torch.bool)
    outside_lifecycle = torch.zeros(query_count, dtype=torch.bool)

    unknown_start = queries_per_state * 2
    observed_free[unknown_start] = True
    observed_occupied[unknown_start] = True
    contradiction[unknown_start + 1] = True
    outside_lifecycle[unknown_start + 2] = True

    result = project_feasible_tristate(
        logits,
        observed_free=observed_free,
        observed_occupied=observed_occupied,
        contradiction=contradiction,
        outside_lifecycle=outside_lifecycle,
    )
    probabilities = result.probabilities
    weights = torch.tensor([0.2, 0.7, 1.3])
    (probabilities * weights).sum().backward()

    detached = probabilities.detach()
    free_error = float((detached[:queries_per_state] - torch.tensor([1.0, 0.0, 0.0])).abs().max())
    occupied_error = float(
        (
            detached[queries_per_state : 2 * queries_per_state]
            - torch.tensor([0.0, 1.0, 0.0])
        )
        .abs()
        .max()
    )
    special = detached[unknown_start : unknown_start + 3]
    unknown_error = float((special - torch.tensor([0.0, 0.0, 1.0])).abs().max())
    simplex_error = float((detached.sum(dim=-1) - 1.0).abs().max())
    unconstrained_gradient = logits.grad[unknown_start + 3 :]

    passed = bool(
        free_error == 0.0
        and occupied_error == 0.0
        and unknown_error == 0.0
        and simplex_error <= 1e-6
        and bool(torch.isfinite(logits.grad).all())
        and int(torch.count_nonzero(unconstrained_gradient)) > 0
    )
    report: dict[str, object] = {
        "schema_version": "worldsim_v62.p3_projection_result.v1",
        "task_id": "WS-V62-P3-FEASIBILITY-PROJECTION-01",
        "evidence_path": str(evidence_path),
        "query_count": query_count,
        "queries_per_state": queries_per_state,
        "actor_voxel_count_in_fixture": actor_voxel_count,
        "hard_free_max_error": free_error,
        "hard_occupied_max_error": occupied_error,
        "contradiction_and_lifecycle_unknown_max_error": unknown_error,
        "simplex_max_error": simplex_error,
        "gradient_finite": bool(torch.isfinite(logits.grad).all()),
        "unconstrained_gradient_nonzero": int(torch.count_nonzero(unconstrained_gradient)) > 0,
        "passed": passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("P3 projection contract failed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries-per-state", type=int, default=16)
    args = parser.parse_args()
    report = run(args.evidence, args.output, args.queries_per_state)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

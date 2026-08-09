#!/usr/bin/env python
"""Run the CPU-only A3 R0/R1 exactness smoke; this is not quality evidence."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path

import torch

from motion_proj.worldsim_v3.local_refinement import LocalRefinementGuard


def _parameters() -> OrderedDict[str, torch.nn.Parameter]:
    generator = torch.Generator().manual_seed(0)

    def parameter(*shape: int) -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.randn(*shape, generator=generator))

    return OrderedDict(
        [
            ("Background._means", parameter(4, 3)),
            ("Background._scales", parameter(4, 3)),
            ("Background._opacities", parameter(4, 1)),
            ("Background._features_dc", parameter(4, 3)),
            ("Background._features_rest", parameter(4, 3, 2)),
            ("RigidNodes._means", parameter(2, 3)),
            ("RigidNodes._opacities", parameter(2, 1)),
            ("RigidNodes.instances_quats", parameter(2, 5, 4)),
            ("RigidNodes.instances_trans", parameter(2, 5, 3)),
        ]
    )


def _optimizer(
    parameters: OrderedDict[str, torch.nn.Parameter],
) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {
                "params": [parameter],
                "name": name,
                "lr": 1.0e-2,
                "eps": 1.0e-15,
                "weight_decay": 0.0,
            }
            for name, parameter in parameters.items()
        ],
        lr=0.0,
        eps=1.0e-15,
    )


def _clone(parameters: OrderedDict[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in parameters.items()
    }


def run_smoke() -> dict[str, object]:
    torch.use_deterministic_algorithms(True)

    # R0 is deliberately an alias: zero steps and no new state/checkpoint keys.
    r0_parameters = _parameters()
    r0_before = _clone(r0_parameters)
    r0_after = _clone(r0_parameters)
    r0_exact = all(
        torch.equal(r0_before[name], r0_after[name]) for name in r0_before
    )

    parameters = _parameters()
    optimizer = _optimizer(parameters)
    # Rows 0/1 are S-A/S-B, row 2 is affected but S-C, row 3 is unaffected.
    mutable_rows = torch.tensor([True, True, False, False])
    guard = LocalRefinementGuard(
        parameters=parameters,
        optimizer=optimizer,
        mutable_rows=mutable_rows,
    )
    before = _clone(parameters)
    optimizer.zero_grad()
    # Deliberately generate gradients for every field and every row. The guard
    # must remove all forbidden gradients before Adam sees them.
    loss = sum((parameter.square().mean() + parameter.mean()) for parameter in parameters.values())
    if not torch.isfinite(loss):
        raise RuntimeError("synthetic A3 loss is not finite")
    loss.backward()
    gradient_audit = guard.before_optimizer_step()
    optimizer.step()
    exactness_audit = guard.after_optimizer_step()
    after = _clone(parameters)

    mutable_fields = {"Background._opacities", "Background._scales"}
    inside_changed = {
        name: bool(torch.any(before[name][mutable_rows] != after[name][mutable_rows]))
        for name in sorted(mutable_fields)
    }
    outside_exact = {
        name: bool(torch.equal(before[name][~mutable_rows], after[name][~mutable_rows]))
        for name in sorted(mutable_fields)
    }
    forbidden_exact = {
        name: bool(torch.equal(before[name], after[name]))
        for name in before
        if name not in mutable_fields
    }
    shape_and_order_exact = tuple(before) == tuple(after) and all(
        before[name].shape == after[name].shape for name in before
    )
    passed = bool(
        r0_exact
        and gradient_audit["pass"]
        and exactness_audit["pass"]
        and all(inside_changed.values())
        and all(outside_exact.values())
        and all(forbidden_exact.values())
        and shape_and_order_exact
    )
    if not passed:
        raise RuntimeError("A3 synthetic exactness smoke failed")
    return {
        "schema_version": 1,
        "task_id": "WS-V3-A3-LOCAL-REFINE-01",
        "audit_version": "A3-R0-R1-SYNTHETIC-SMOKE-v1",
        "evidence_tier": "synthetic_contract_only_not_quality_evidence",
        "pass": passed,
        "r0": {
            "optimizer_steps": 0,
            "new_checkpoint_created": False,
            "parameter_state_exact": r0_exact,
        },
        "r1": {
            "optimizer_steps": 1,
            "mutable_rows": mutable_rows.tolist(),
            "inside_changed": inside_changed,
            "outside_exact": outside_exact,
            "forbidden_exact": forbidden_exact,
            "shape_and_order_exact": shape_and_order_exact,
            "gradient_audit": gradient_audit,
            "exactness_audit": exactness_audit,
        },
        "formal_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = run_smoke()
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

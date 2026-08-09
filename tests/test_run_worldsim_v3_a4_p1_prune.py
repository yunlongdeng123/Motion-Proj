from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.finalize_worldsim_v3_a4_p1 import expected_removal_exact
from scripts.run_worldsim_v3_a4_p1_prune import atomic_json, directory_digest
from scripts.run_worldsim_v3_a4_p1_worker import (
    atomic_npz,
    finite_metric_tree,
    invariant_hashes,
    quality_safeguard_rows,
    row_alignment_audit,
)


def test_atomic_outputs_refuse_overwrite_and_directory_digest_is_stable(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "record.json"
    atomic_json(json_path, {"status": "done"})
    with pytest.raises(FileExistsError):
        atomic_json(json_path, {"status": "changed"})
    npz_path = tmp_path / "scores.npz"
    atomic_npz(npz_path, {"scores": np.arange(3, dtype=np.float64)})
    with pytest.raises(FileExistsError):
        atomic_npz(npz_path, {"scores": np.arange(4, dtype=np.float64)})
    first = directory_digest(tmp_path, "*.json")
    second = directory_digest(tmp_path, "*.json")
    assert first == second
    assert first["file_count"] == 1


def checkpoint_fixture() -> OrderedDict:
    def model(count: int, *, rigid: bool) -> OrderedDict:
        state = OrderedDict(
            {
                "_means": torch.zeros(count, 3),
                "_scales": torch.zeros(count, 3),
                "_quats": torch.zeros(count, 4),
                "_features_dc": torch.zeros(count, 3),
                "_features_rest": torch.zeros(count, 6),
                "_opacities": torch.zeros(count, 1),
                "worldsim_a2_ancestry": {
                    "fields": {
                        "gaussian_id": torch.arange(count),
                        "actor_id": torch.zeros(count, dtype=torch.long),
                    }
                },
            }
        )
        if rigid:
            state["points_ids"] = torch.zeros(count, 1, dtype=torch.long)
            state["instances_quats"] = torch.zeros(2, 1, 4)
            state["instances_trans"] = torch.zeros(2, 1, 3)
            state["instances_size"] = torch.ones(1, 3)
            state["instances_fv"] = torch.ones(2, 1, dtype=torch.bool)
        return state

    return OrderedDict(
        {
            "models": OrderedDict(
                {
                    "Background": model(3, rigid=False),
                    "RigidNodes": model(4, rigid=True),
                    "Sky": {"base": torch.ones(2)},
                }
            ),
            "lpips.net.weight": torch.ones(1),
            "step": torch.tensor(30_000),
        }
    )


def test_row_alignment_and_invariant_hash_inventory_are_complete() -> None:
    checkpoint = checkpoint_fixture()
    alignment = row_alignment_audit(checkpoint)
    assert alignment["exact"]
    hashes = invariant_hashes(checkpoint)
    assert "models.Sky.base" in hashes
    assert "models.RigidNodes.instances_fv" in hashes
    assert "lpips.net.weight" in hashes
    assert "step" in hashes
    checkpoint["models"]["RigidNodes"]["points_ids"] = torch.zeros(3, 1, dtype=torch.long)
    assert not row_alignment_audit(checkpoint)["exact"]


def test_quality_safeguards_cover_global_actor_boundary_and_non_target() -> None:
    metric_contracts = [
        {"name": "psnr", "direction": "higher", "maximum_regression": 0.1},
        {"name": "lpips", "direction": "lower", "maximum_regression": 0.01},
    ]
    protocol = {
        "quality_contract": {
            "global_endpoints": {"metrics": metric_contracts},
            "actor_endpoints": {
                "roles": ["high-support", "boundary-support"],
                "regions": ["actor_region", "boundary_band"],
                "metrics": metric_contracts,
            },
            "non_target_endpoints": {"metrics": metric_contracts},
        }
    }
    region = {"psnr": 20.0, "lpips": 0.1}
    baseline = {
        "global_metrics": region,
        "roles": {
            role: {name: region for name in ("actor_region", "boundary_band")}
            for role in ("high-support", "boundary-support")
        },
        "non_target": {"quality": region},
    }
    candidate = {
        **baseline,
        "global_metrics": {"psnr": 19.95, "lpips": 0.105},
    }
    rows = quality_safeguard_rows(protocol, baseline, candidate)
    assert len(rows) == 12
    assert all(row["passed"] for row in rows)
    assert finite_metric_tree(candidate)
    assert not finite_metric_tree({"bad": float("nan")})


def test_expected_removal_exact_binds_counts_to_floor_rule() -> None:
    stage = {
        "prune_fraction": 0.10,
        "candidate_grid_and_removal_counts_exact": True,
        "source_and_candidate_model_counts": {
            "source": {"Background": 10, "RigidNodes": 10},
            "candidate": {"Background": 9, "RigidNodes": 9},
        },
        "per_asset_removed_count": [
            {"source_count": 10, "removed_count": 1},
            {"source_count": 10, "removed_count": 1},
        ],
    }
    assert expected_removal_exact(stage)
    stage["per_asset_removed_count"][1]["removed_count"] = 0
    assert not expected_removal_exact(stage)

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_worldsim_v3_a4_p2_precision import atomic_json, directory_digest
from scripts.run_worldsim_v3_a4_p2_worker import (
    baseline_replay_rows,
    finite_metric_tree,
    quality_safeguard_rows,
)


def test_atomic_json_refuses_overwrite_and_directory_digest_is_stable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "record.json"
    atomic_json(path, {"status": "done"})
    with pytest.raises(FileExistsError):
        atomic_json(path, {"status": "changed"})
    assert directory_digest(tmp_path, "*.json") == directory_digest(tmp_path, "*.json")


def quality_fixture() -> dict:
    region = {
        "psnr": 20.0,
        "ssim": 0.8,
        "masked_lpips_alex_tight_crop_256px": 0.1,
        "mean_absolute_error": 0.03,
        "status": "done",
        "pixel_count": 10,
        "visible_image_count": 1,
    }
    return {
        "global_metrics": {
            "image_metrics/test/human_psnr": 20.0,
            "image_metrics/test/human_ssim": 0.8,
            "image_metrics/test/lpips": 0.1,
            "image_metrics/test/masked_psnr": 20.0,
            "image_metrics/test/masked_ssim": 0.8,
            "image_metrics/test/occupied_psnr": 20.0,
            "image_metrics/test/occupied_ssim": 0.8,
            "image_metrics/test/psnr": 20.0,
            "image_metrics/test/ssim": 0.8,
            "image_metrics/test/vehicle_psnr": 20.0,
            "image_metrics/test/vehicle_ssim": 0.8,
        },
        "roles": {
            role: {
                "actor": {"rigid_model_index": index},
                "actor_region": dict(region),
                "boundary_band": dict(region),
            }
            for index, role in enumerate(("high-support", "boundary-support"))
        },
        "non_target": {"quality": dict(region)},
    }


def protocol_fixture(source_path: Path) -> dict:
    global_metrics = [
        {"name": name, "direction": "lower" if "lpips" in name else "higher", "maximum_regression": 0.1}
        for name in quality_fixture()["global_metrics"]
    ]
    regional = [
        {"name": "psnr", "direction": "higher", "maximum_regression": 0.1},
        {"name": "ssim", "direction": "higher", "maximum_regression": 0.01},
        {
            "name": "masked_lpips_alex_tight_crop_256px",
            "direction": "lower",
            "maximum_regression": 0.01,
        },
        {
            "name": "mean_absolute_error",
            "direction": "lower",
            "maximum_regression": 0.01,
        },
    ]
    return {
        "baseline_quality": {"source_quality": {"path": str(source_path)}},
        "quality_contract": {
            "global_endpoints": {"metrics": global_metrics},
            "actor_endpoints": {
                "roles": ["high-support", "boundary-support"],
                "regions": ["actor_region", "boundary_band"],
                "metrics": regional,
            },
            "non_target_endpoints": {"metrics": regional},
            "baseline_p1_replay_tolerance": {
                "psnr_absolute": 1e-6,
                "ssim_absolute": 1e-8,
                "lpips_absolute": 1e-8,
                "mean_absolute_error_absolute": 1e-8,
            },
        },
    }


def test_quality_contract_has_exactly_31_fail_closed_rows(tmp_path: Path) -> None:
    source = quality_fixture()
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    protocol = protocol_fixture(source_path)
    replay = baseline_replay_rows(protocol, source)
    safeguards = quality_safeguard_rows(protocol, source, quality_fixture())
    assert len(replay) == 31 and all(row["passed"] for row in replay)
    assert len(safeguards) == 31 and all(row["passed"] for row in safeguards)


def test_quality_comparison_rejects_missing_or_over_budget_endpoint(
    tmp_path: Path,
) -> None:
    source = quality_fixture()
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    protocol = protocol_fixture(source_path)
    candidate = quality_fixture()
    candidate["global_metrics"]["image_metrics/test/psnr"] = 19.0
    rows = quality_safeguard_rows(protocol, source, candidate)
    assert len(rows) == 31
    assert not next(row for row in rows if row["name"] == "image_metrics/test/psnr")["passed"]
    candidate["global_metrics"].pop("image_metrics/test/psnr")
    rows = quality_safeguard_rows(protocol, source, candidate)
    assert not next(row for row in rows if row["name"] == "image_metrics/test/psnr")["passed"]


def test_finite_metric_tree_rejects_nan_and_infinity() -> None:
    assert finite_metric_tree(quality_fixture())
    assert not finite_metric_tree({"bad": float("nan")})
    assert not finite_metric_tree({"bad": float("inf")})

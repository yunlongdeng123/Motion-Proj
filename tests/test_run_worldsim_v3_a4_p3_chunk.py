from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.run_worldsim_v3_a4_p3_chunk import atomic_json, directory_digest
from scripts.run_worldsim_v3_a4_p3_worker import (
    endpoint_triples,
    exact_replay_rows,
    per_view_rgb_exact,
)


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
        "per_view_rgb_sha256": [
            {"full_image_index": index, "rgb_sha256": f"{index:064x}"}
            for index in range(57)
        ],
    }


def protocol_fixture() -> dict:
    tolerance = {
        "psnr_absolute": 1e-6,
        "ssim_absolute": 1e-8,
        "lpips_absolute": 1e-8,
        "mean_absolute_error_absolute": 1e-8,
    }
    return {
        "quality_contract": {
            "global_endpoints": {
                "metrics": list(quality_fixture()["global_metrics"])
            },
            "actor_endpoints": {
                "roles": ["high-support", "boundary-support"],
                "regions": ["actor_region", "boundary_band"],
                "metrics": [
                    "psnr",
                    "ssim",
                    "masked_lpips_alex_tight_crop_256px",
                    "mean_absolute_error",
                ],
            },
            "non_target_endpoints": {
                "metrics": [
                    "psnr",
                    "ssim",
                    "masked_lpips_alex_tight_crop_256px",
                    "mean_absolute_error",
                ]
            },
            "p2_baseline_replay_tolerance": dict(tolerance),
            "candidate_source_replay_tolerance": dict(tolerance),
        }
    }


def test_atomic_json_refuses_overwrite_and_directory_digest_is_stable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "record.json"
    atomic_json(path, {"status": "done"})
    with pytest.raises(FileExistsError):
        atomic_json(path, {"status": "changed"})
    assert directory_digest(tmp_path, "*.json") == directory_digest(tmp_path, "*.json")


def test_endpoint_contract_expands_to_exactly_31_rows() -> None:
    source = quality_fixture()
    rows = endpoint_triples(protocol_fixture(), source, quality_fixture())
    assert len(rows) == 31
    assert rows[0][0].startswith("global.")
    assert rows[-1][0] == "non_target.mean_absolute_error"


def test_exact_quality_replay_passes_equal_and_rejects_drift() -> None:
    protocol = protocol_fixture()
    source = quality_fixture()
    rows = exact_replay_rows(
        protocol,
        source,
        quality_fixture(),
        tolerance_key="candidate_source_replay_tolerance",
    )
    assert len(rows) == 31 and all(row["passed"] for row in rows)
    candidate = quality_fixture()
    candidate["global_metrics"]["image_metrics/test/psnr"] -= 0.01
    rows = exact_replay_rows(
        protocol,
        source,
        candidate,
        tolerance_key="candidate_source_replay_tolerance",
    )
    assert not next(
        row for row in rows if row["endpoint"] == "global.image_metrics/test/psnr"
    )["passed"]


def test_exact_quality_replay_rejects_missing_or_nonfinite() -> None:
    protocol = protocol_fixture()
    source = quality_fixture()
    missing = quality_fixture()
    missing["non_target"]["quality"].pop("ssim")
    rows = exact_replay_rows(
        protocol,
        source,
        missing,
        tolerance_key="candidate_source_replay_tolerance",
    )
    assert not next(row for row in rows if row["endpoint"] == "non_target.ssim")[
        "passed"
    ]
    nonfinite = quality_fixture()
    nonfinite["global_metrics"]["image_metrics/test/lpips"] = float("nan")
    rows = exact_replay_rows(
        protocol,
        source,
        nonfinite,
        tolerance_key="candidate_source_replay_tolerance",
    )
    assert not next(
        row for row in rows if row["endpoint"] == "global.image_metrics/test/lpips"
    )["passed"]


def test_per_view_rgb_sha_requires_all_57_exact() -> None:
    source = quality_fixture()
    exact, rows = per_view_rgb_exact(source, quality_fixture())
    assert exact and len(rows) == 57
    candidate = deepcopy(source)
    candidate["per_view_rgb_sha256"][7]["rgb_sha256"] = "f" * 64
    exact, rows = per_view_rgb_exact(source, candidate)
    assert not exact and not rows[7]["passed"]
    candidate = deepcopy(source)
    candidate["per_view_rgb_sha256"].pop()
    assert not per_view_rgb_exact(source, candidate)[0]

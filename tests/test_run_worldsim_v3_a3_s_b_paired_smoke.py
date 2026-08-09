from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.run_worldsim_v3_a3_s_b_paired_smoke import (
    load_and_validate_unit,
    ordered_unit_records,
    sha256_file,
)


def paired_protocol() -> dict:
    return {
        "paired_design": {
            "actor_roles": ["high-support", "boundary-support"],
            "edits": ["lateral", "delete"],
            "heldout": {"frames": [10, 20]},
            "actors": {
                "high-support": {"instance_token": "high", "rigid_model_index": 5},
                "boundary-support": {
                    "instance_token": "boundary",
                    "rigid_model_index": 21,
                },
            },
        }
    }


def unit_record(role: str, edit: str) -> dict:
    high = role == "high-support"
    return {
        "role": role,
        "edit": edit,
        "heldout": False,
        "frame": 0 if high else 31,
        "instance_token": "high" if high else "boundary",
        "rigid_model_index": 5 if high else 21,
    }


def test_unit_records_are_protocol_ordered_and_heldout_safe() -> None:
    records = [
        unit_record("boundary-support", "delete"),
        unit_record("high-support", "delete"),
        unit_record("boundary-support", "lateral"),
        unit_record("high-support", "lateral"),
    ]
    ordered = ordered_unit_records(
        paired_protocol(), {"evidence": {"units": records}}
    )
    assert [(row["role"], row["edit"]) for row in ordered] == [
        ("high-support", "lateral"),
        ("high-support", "delete"),
        ("boundary-support", "lateral"),
        ("boundary-support", "delete"),
    ]
    records[0]["frame"] = 10
    with pytest.raises(RuntimeError, match="provenance drift"):
        ordered_unit_records(paired_protocol(), {"evidence": {"units": records}})


def write_unit(path: Path, *, rgb_authorized: bool = False) -> None:
    shape = (2, 3)
    false = np.zeros(shape, dtype=np.bool_)
    source = false.copy()
    source[0, 0] = True
    affected = false.copy()
    affected[0, :2] = True
    geometry = false.copy()
    geometry[0, 1] = True
    rgb = false.copy()
    rgb[0, 1] = rgb_authorized
    depth = np.zeros(shape, dtype=np.float32)
    depth[0, 1] = 4.0
    measured_valid = depth > 0
    np.savez_compressed(
        path,
        source_actor_footprint=source,
        edited_actor_footprint=false,
        affected_pixel_mask=affected,
        rgb_loss_mask=rgb,
        geometry_loss_mask=geometry,
        depth_render_expected=depth,
        depth_render_expected_valid=measured_valid,
        depth_surface_first_hit=depth,
        depth_surface_first_hit_valid=measured_valid,
        depth_lidar_measured=depth,
        depth_lidar_measured_valid=measured_valid,
    )


def test_unit_loader_accepts_only_conservative_s_b_t0(tmp_path: Path) -> None:
    path = tmp_path / "unit.npz"
    write_unit(path)
    record = {
        "path": path.name,
        "sha256": sha256_file(path),
        "s_b_t0_geometry_pixels": 1,
    }
    arrays = load_and_validate_unit(tmp_path, record)
    assert arrays["geometry_loss_mask"].sum() == 1

    bad = tmp_path / "bad.npz"
    write_unit(bad, rgb_authorized=True)
    record["path"] = bad.name
    record["sha256"] = sha256_file(bad)
    with pytest.raises(RuntimeError, match="semantics drift"):
        load_and_validate_unit(tmp_path, record)

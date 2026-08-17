from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v5.ownership_renderer import compose_ownership_opacity
from motion_proj.worldsim_v51.evidence.cif_decoupling import (
    compose_decoupled_actor_opacity,
)


def test_constant_occupancy_is_exact_existing_renderer_noop() -> None:
    appearance = np.asarray([0.2, 0.8, 1.0], dtype=np.float32)
    identity = np.asarray([0.9, 0.4, 0.1], dtype=np.float32)
    existing = compose_ownership_opacity(appearance, identity)
    decoupled = compose_decoupled_actor_opacity(
        appearance_opacity=appearance,
        occupancy_probability=np.ones_like(appearance),
        conditional_actor_probability=identity,
    )
    assert np.array_equal(existing, decoupled)


def test_reusing_appearance_opacity_double_counts_alpha() -> None:
    appearance = np.asarray([0.2, 0.8], dtype=np.float32)
    identity = np.asarray([0.9, 0.4], dtype=np.float32)
    existing = compose_ownership_opacity(appearance, identity)
    conflated = compose_decoupled_actor_opacity(
        appearance_opacity=appearance,
        occupancy_probability=appearance,
        conditional_actor_probability=identity,
    )
    assert not np.array_equal(existing, conflated)
    assert np.all(conflated < existing)


def test_a4_config_fails_closed_without_occupancy_observable() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/worldsim_v51/m1_cif_decoupling_audit_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    candidates = config["occupancy_candidates"]
    assert candidates["appearance_opacity"]["allowed"] is False
    assert candidates["visibility_or_effective_count"]["allowed"] is False
    assert candidates["constant_one_for_instantiated_gaussian"]["expected_effect"] == "exact_noop"
    assert config["restrictions"]["evaluation_quality_read"] is False

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.ndimage import binary_erosion


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_worldsim_v5_m1b_boundary_residual_forensics.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("worldsim_v5_m1b_boundary_forensic", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _base_case() -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((32, 32), dtype=bool)
    target[8:24, 8:24] = True
    probability = np.where(target, 0.9, 0.1).astype(np.float32)
    return probability, target


def test_boundary_only_errors_are_localized_and_enriched() -> None:
    probability, target = _base_case()
    target_boundary = target & ~binary_erosion(target)
    probability[target_boundary] = 0.1
    row = MODULE.analyze_view(
        probability, target, threshold=0.5, boundary_iterations=3
    )
    aggregate = MODULE.aggregate_view_rows([row])
    assert aggregate["classification_error_count"] > 0
    assert aggregate["boundary_classification_error_share"] == 1.0
    assert aggregate["boundary_semantic_error_mass_share"] > 0.5
    assert aggregate["boundary_error_enrichment"] > 2.0


def test_far_interior_errors_do_not_look_boundary_primary() -> None:
    probability, target = _base_case()
    probability[12:20, 12:20] = 0.1
    row = MODULE.analyze_view(
        probability, target, threshold=0.5, boundary_iterations=3
    )
    aggregate = MODULE.aggregate_view_rows([row])
    assert aggregate["classification_error_count"] == 64
    assert aggregate["boundary_classification_error_share"] == 0.0
    assert aggregate["boundary_error_enrichment"] == 0.0
    assert aggregate["boundary_false_negative_mass_share"] < 0.5


def test_config_freezes_boundary_gate_without_automatic_unlock() -> None:
    config = MODULE.load_config(
        ROOT / "configs/worldsim_v5/m1b_boundary_residual_forensics_v1.yaml"
    )
    assert [row["scene"] for row in config["inputs"].values()] == [
        "scene-0471",
        "scene-1087",
        "scene-0379",
    ]
    assert config["analysis"]["probability_threshold"] == 0.5
    assert config["analysis"]["boundary_band_iterations"] == 3
    assert config["gate"]["denominator"] == "6_scene_by_unary_g0_cells"
    assert config["gate"]["minimum_boundary_primary_cells"] == 4
    assert config["gate"]["automatic_semantic_split_unlock"] is False
    assert config["restrictions"]["parameter_search_performed"] is False
    assert config["restrictions"]["semantic_split_started"] is False


def test_boundary_forensic_runner_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout

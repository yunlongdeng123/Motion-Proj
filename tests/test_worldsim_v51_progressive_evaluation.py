from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v51.progressive_evaluation import (
    METRICS,
    evaluate_progressive_h_gate,
)
from scripts.run_worldsim_v51_d0_h_evaluation import validate_config


CONFIG = ROOT / "configs/worldsim_v51/stage_d_progressive_h_evaluation_v1.yaml"


def _metrics(boundary: float, iou: float, fn: float) -> dict[str, float]:
    return {
        "boundary_f1": boundary,
        "iou_at_frozen_threshold": iou,
        "false_negative_semantic_mass": fn,
        "false_positive_semantic_mass": 0.1,
        "brier": 0.1,
        "ece": 0.1,
        "nll": 0.1,
    }


def test_progressive_h_gate_is_scene_balanced_and_fail_closed() -> None:
    scenes = []
    for name, boundary_delta in zip(("a", "b", "c"), (0.03, 0.02, -0.01)):
        baseline = _metrics(0.3, 0.4, 0.2)
        candidate = _metrics(0.3 + boundary_delta, 0.41, 0.21)
        scenes.append(
            {
                "scene": name,
                "evaluation_aggregate": {"U2_B3_G0": baseline, "D0": candidate},
            }
        )
    gate = {
        "scene_count": 3,
        "minimum_positive_boundary_f1_scenes": 2,
        "minimum_scene_balanced_boundary_f1_delta_exclusive": 0.0,
        "minimum_scene_balanced_iou_delta": 0.0,
        "maximum_scene_balanced_false_negative_semantic_mass_delta": 0.02,
    }
    report = evaluate_progressive_h_gate(scenes, gate)
    assert report["pass"] is True
    assert report["positive_boundary_f1_scene_count"] == 2
    scenes[2]["evaluation_aggregate"]["D0"]["false_negative_semantic_mass"] = 0.5
    assert evaluate_progressive_h_gate(scenes, gate)["pass"] is False


def test_d0_h_config_freezes_arms_gate_precision_and_locks() -> None:
    config, freeze = validate_config(CONFIG)
    assert freeze["status"] == "done"
    assert config["arms"]["primary_comparator"] == "U2_B3_G0"
    assert config["arms"]["strong_external_baseline"] == "U2_B3_G_V5"
    assert config["arms"]["candidate"] == "D0"
    assert config["evaluation"]["metrics"] == list(METRICS)
    assert config["evaluation"]["expected_total_view_count"] == 12
    assert config["evaluation"]["metric_source_precision"] == (
        "persisted_float16_for_all_arms"
    )
    assert config["h_gate"]["minimum_positive_boundary_f1_scenes"] == 2
    assert config["h_gate"]["maximum_scene_balanced_false_negative_semantic_mass_delta"] == 0.02
    assert config["resources"]["maximum_nvidia_at_start_mib"] == 512
    assert config["locks"]["parameter_search"] is False
    assert config["locks"]["screening_quality_read"] is False
    assert config["locks"]["confirmation_quality_read"] is False
    assert config["locks"]["validation_quality_read"] is False
    assert config["locks"]["test_quality_read"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"


def test_d0_h_runner_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_worldsim_v51_d0_h_evaluation.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout
    assert "--device" in result.stdout

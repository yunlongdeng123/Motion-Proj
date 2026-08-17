from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_e0a_superprimitive_probe import validate_config


CONFIG = ROOT / "configs/worldsim_v51/stage_e_e0a_superprimitive_probe_v2.yaml"


def test_e0a_config_freezes_route_levels_gate_and_locks() -> None:
    config, _, _ = validate_config(CONFIG)
    assert config["method"]["voxel_size_quantiles"] == [0.5, 0.75, 0.9]
    assert config["method"]["zero_length_edge_policy"] == (
        "exclude_from_scale_quantiles_preserve_all_gaussians"
    )
    assert config["method"]["selected_level"] is None
    assert config["method"]["quality_target_consumed"] is False
    assert config["method"]["propagation_executed"] is False
    assert config["e0a_gate"]["expected_scene_count"] == 3
    assert config["locks"]["e0b_propagation_execution"] is False
    assert config["locks"]["e1_panogs_execution"] is False
    assert config["locks"]["e2_ag2aussian_execution"] is False
    assert config["locks"]["validation_quality_read"] is False
    assert config["locks"]["test_quality_read"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"
    assert config["recovery"]["blocked_run"]["status"] == "blocked"


def test_e0a_runner_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_worldsim_v51_e0a_superprimitive_probe.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run-dir" in result.stdout

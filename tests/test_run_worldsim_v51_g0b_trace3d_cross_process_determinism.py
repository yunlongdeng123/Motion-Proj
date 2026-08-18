from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_g0b_trace3d_cross_process_determinism import _validate_config


CONFIG = ROOT / "configs/worldsim_v51/stage_g_g0b_trace3d_cross_process_determinism_v1.yaml"


def test_g0b_freezes_fresh_process_determinism_and_failover_without_patch_or_quality():
    config = _validate_config(CONFIG)
    assert config["runtime"]["fresh_process_count"] == 8
    assert config["gates"]["foreground_alpha_unique_vector_count_maximum"] == 1
    assert config["decision"]["source_patch_allowed"] is False
    assert config["decision"]["fail_next_task"] == "WS-V51-M1-H-GRAPHFREE-01"
    assert config["locks"]["real_checkpoint_read"] is False
    assert config["locks"]["quality_metrics_read"] is False

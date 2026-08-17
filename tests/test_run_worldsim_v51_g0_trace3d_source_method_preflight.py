from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_g0_trace3d_source_method_preflight import _validate_config


CONFIG = ROOT / "configs/worldsim_v51/stage_g_g0_trace3d_source_method_preflight_v1.yaml"


def test_trace3d_source_preflight_binds_official_commit_and_immutable_boundary():
    config = _validate_config(CONFIG)
    assert config["official_source"]["repository"]["commit"] == "7465ad94d8e7e988513c1326bbc015e8b59cc442"
    assert config["official_source"]["repository"]["initialize_submodules"] is False
    boundary = config["normative_adapter_boundary"]
    assert boundary["immutable_base"] is True
    assert "no_quality_disagreement_diagnostic" in boundary["initially_allowed"]
    assert "gaussian_split" in boundary["initially_forbidden"]
    assert config["locks"]["source_code_execution"] is False
    assert config["locks"]["quality_metrics_read"] is False

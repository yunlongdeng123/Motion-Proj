from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_g0a_trace3d_reverse_tracing_capability import _validate_config


CONFIG = ROOT / "configs/worldsim_v51/stage_g_g0a_trace3d_reverse_tracing_capability_v1.yaml"


def test_trace3d_g0a_binds_exact_source_isolated_build_and_synthetic_only_scope():
    config = _validate_config(CONFIG)
    assert config["official_source"]["commit"] == "7465ad94d8e7e988513c1326bbc015e8b59cc442"
    assert config["official_source"]["source_patch_allowed"] is False
    assert config["build"]["expected_torch"] == "2.1.2+cu118"
    assert config["build"]["expected_torch_cuda"] == "11.8"
    assert config["build"]["torch_cuda_arch_list"] == "8.6"
    assert config["synthetic_probe"]["forbidden_claim"] == "real_worldsim_adapter_or_quality_supported"
    assert config["locks"]["real_checkpoint_read"] is False
    assert config["locks"]["quality_metrics_read"] is False

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_g0_trace3d_source_method_preflight_v2 import _pdf_page_marker_count, _validate_config


CONFIG = ROOT / "configs/worldsim_v51/stage_g_g0_trace3d_source_method_preflight_v2.yaml"


def test_trace3d_recovery_reuses_exact_assets_and_preserves_locks():
    config = _validate_config(CONFIG)
    assert config["recovery"]["reuse_exact_published_assets"] is True
    assert config["recovery"]["redownload_assets"] is False
    assert config["published_assets"]["repository"]["commit"] == "7465ad94d8e7e988513c1326bbc015e8b59cc442"
    assert config["locks"]["network_access"] is False
    assert config["locks"]["source_code_execution"] is False
    assert config["locks"]["quality_metrics_read"] is False


def test_standard_library_pdf_page_marker_count(tmp_path):
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.7\n1 0 obj<</Type /Page>>endobj\n2 0 obj<</Type\n/Page>>endobj\n")
    assert _pdf_page_marker_count(paper) == 2

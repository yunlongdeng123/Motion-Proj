from pathlib import Path

from scripts.replay_n1_cutin_k4_evidence import replay


def test_frozen_k4_evidence_regression_rejects_all_human_false_positives(tmp_path):
    root = Path(__file__).resolve().parents[1]
    report = replay(
        root / "tests" / "fixtures" / "n1_cutin_k4",
        tmp_path / "evidence-replay",
    )
    rows = {row["audit_id"]: row for row in report["cases"]}
    assert report["passed"] is True
    assert report["human_false_positive_pass_count"] == 0
    assert report["blocking_tp_passed"] is True
    assert rows["K4-010"]["status"] == "PASS"
    assert rows["K4-011"]["status"] == "PASS"
    assert rows["K4-012"]["primary_reason"] == "RECEIVER_IDENTITY_SWITCH"
    assert rows["K4-015"]["status"] == "FAIL"

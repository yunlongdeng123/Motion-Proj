import json
from pathlib import Path

import yaml

from scripts.replay_n1_cutin_k4_evidence import strict_from_frozen_evidence
from scripts.replay_n1_cutin_k4_scenes import _case_report


def test_scene_replay_comparison_requires_raw_semantic_match():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests" / "fixtures" / "n1_cutin_k4"
    evidence = json.loads((fixture / "evidence" / "K4-010.json").read_text(encoding="utf-8"))
    expected = yaml.safe_load((fixture / "expected_strict_status.yaml").read_text(encoding="utf-8"))["cases"]["K4-010"]
    strict = strict_from_frozen_evidence(evidence)
    record = {
        "event_id": "scene-0400:11:F1:110:111",
        "event_record_sha256": "fresh-v2-record",
        "subject_instance_token": evidence["roles"]["SUBJECT"],
        "source_run": evidence["source_run"],
        "target_run": evidence["target_run"],
        "crossing_frame": evidence["crossing_frame"],
        "strict": strict,
    }
    result = _case_report("K4-010", evidence, expected, record)
    assert result["passed"] is True
    assert result["comparison"]["raw_only"] is True
    missing = _case_report("K4-010", evidence, expected, None)
    assert missing["passed"] is False
    assert missing["failure"] == "event_not_reconstructed_from_raw_annotation_and_map"

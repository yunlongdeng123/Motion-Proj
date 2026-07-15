from __future__ import annotations

import json
from pathlib import Path

from motion_proj.diagnostics.physics_dpo_pa0_review import aggregate_pa0_reviews


P0_CHECKS = (
    "nonidentity_primary_correction",
    "correction_above_uncertainty",
    "direction_preserved",
    "dynamic_degree_preserved",
    "net_displacement_preserved",
    "turn_preserved",
    "frame0_exact",
    "visibility_preserved",
    "support_preserved",
    "synthetic_clean_preserved",
    "synthetic_high_snr_outlier_improved",
    "synthetic_subuncertainty_jitter_not_amplified",
)
E0_CHECKS = (
    "all_real_clips_have_valid_tracks",
    "identical_video_rerun",
    "occlusion_low_texture_invalidity_recognized",
    "perturbation_rank_correlation",
    "synthetic_acceleration_and_jerk_order",
    "threshold_sweep_rank_correlation",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _make_source(root: Path, *, role: str, valid_count: int = 12) -> None:
    root.mkdir()
    run_id = "p0-source" if role == "p0" else "e0-source"
    _write_json(root / "manifest.json", {"run_id": run_id, "status": "awaiting_reviews"})
    (root / "resolved.yaml").write_text("thresholds: {}\n", encoding="utf-8")
    _write_json(root / "summary.json", {"status": "awaiting_reviews"})
    rows = [
        {"case_id": f"{role}-{index:02d}", "verdict": "valid" if index < valid_count else "invalid", "reviewer": "human", "notes": "ok"}
        for index in range(12)
    ]
    _write_jsonl(root / "reviews.template.jsonl", [
        {"case_id": row["case_id"], "verdict": "pending", "reviewer": "human", "notes": ""}
        for row in rows
    ])
    _write_jsonl(root / "reviews.jsonl", rows)
    if role == "p0":
        _write_json(root / "machine_summary.json", {
            "uses_future_gt": False,
            "base_generation_adapter_loaded": False,
            "machine_checks": {"P-UNC": {"machine_pass": True, "checks": {name: True for name in P0_CHECKS}}},
            "generated_track_audit": {"P-UNC": {
                "frame0_correction_px": {"max": 0.0},
                "visibility_expansion_count": 0,
                "valid_time_index_changed_count": 0,
                "support_violation_count": 0,
            }},
        })
    else:
        _write_json(root / "machine_summary.json", {
            "uses_future_gt": False,
            "fallback_used": False,
            "machine_pass": True,
            "decision": {"machine_pass": True, "checks": {name: True for name in E0_CHECKS}},
        })


def _config(p0: Path, e0: Path, output: Path) -> dict:
    return {
        "run_id": "pa0-test",
        "work_dir": str(output),
        "task_id": "PA0-REVIEW-00",
        "protocol": "sap-dpo-pa0-review-v1",
        "sources": {
            "p0": {
                "path": str(p0), "run_id": "p0-source", "required_reviews": 12,
                "minimum_valid_reviews": 11, "candidate": "P-UNC",
                "required_machine_checks": list(P0_CHECKS),
                "required_zero_audit_fields": [
                    "frame0_correction_px.max", "visibility_expansion_count",
                    "valid_time_index_changed_count", "support_violation_count",
                ],
            },
            "e0": {
                "path": str(e0), "run_id": "e0-source", "required_reviews": 12,
                "minimum_valid_reviews": 10, "required_machine_checks": list(E0_CHECKS),
            },
        },
    }


def test_pa0_aggregation_copies_reviews_and_preserves_source_hashes(tmp_path: Path) -> None:
    p0, e0, output = tmp_path / "p0", tmp_path / "e0", tmp_path / "pa0"
    _make_source(p0, role="p0")
    _make_source(e0, role="e0")
    p0_before = (p0 / "reviews.jsonl").read_bytes()
    e0_before = (e0 / "reviews.jsonl").read_bytes()

    run_dir, summary = aggregate_pa0_reviews(_config(p0, e0, output), command=["pytest"])

    assert run_dir == output
    assert summary["status"] == "done"
    assert summary["p0_valid_reviews"] == 12
    assert summary["e0_valid_reviews"] == 12
    assert (output / "COMPLETE").is_file()
    assert (output / "reviews.p0.jsonl").read_bytes() == p0_before
    assert (output / "reviews.e0.jsonl").read_bytes() == e0_before
    assert (p0 / "reviews.jsonl").read_bytes() == p0_before
    assert (e0 / "reviews.jsonl").read_bytes() == e0_before
    decision = json.loads((output / "review_decision.json").read_text(encoding="utf-8"))
    assert decision["p0"]["review"]["case_ids_match_template"]
    assert decision["e0"]["review"]["case_ids_match_template"]


def test_pa0_below_threshold_is_recorded_as_blocked_not_promoted(tmp_path: Path) -> None:
    p0, e0, output = tmp_path / "p0", tmp_path / "e0", tmp_path / "pa0"
    _make_source(p0, role="p0", valid_count=10)
    _make_source(e0, role="e0")

    _, summary = aggregate_pa0_reviews(_config(p0, e0, output), command=["pytest"])

    assert summary["status"] == "blocked"
    assert not summary["p0_pass"]
    assert summary["e0_pass"]
    assert (output / "COMPLETE").is_file()

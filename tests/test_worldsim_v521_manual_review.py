from __future__ import annotations

from motion_proj.worldsim_v521.review_package import select_representative_cases


def _row(base: str, split: str, axis: str, scene: str, frame: int, camera: int, psnr: float) -> dict:
    case = f"BC-{base.upper()}-{split}-{axis}-{scene}-{frame}-{camera}"
    metrics = {
        name: {"status": "done", "psnr": psnr, "ssim": 0.5, "lpips_alex": 0.2, "pixel_count": 100}
        for name in ("global", "actor", "boundary")
    }
    return {
        "case_id": case, "event_ids": {axis: "BCE-" + case}, "base": base,
        "split_role": split, "evidence_tier": "D" if split == "discovery" else "C",
        "scene": scene, "canonical_sample_index": frame, "camera": camera,
        "entity_kind": "view", "classification_status": "labeled", "selected_for_panel": True,
        "panel_path": "/tmp/panel.png", "failure_axes": [axis], "failure_class": ["B-X"],
        "confirmation_verdict": "direction_confirmed", "metrics": metrics,
    }


def test_representative_review_has_all_frozen_slots_and_is_deterministic() -> None:
    rows = []
    for axis_index, axis in enumerate(("GLOBAL_RGB", "ACTOR_RGB", "BOUNDARY")):
        for base in ("adgs", "streetgs"):
            offset = axis_index * 10
            rows.extend(
                [
                    _row(base, "discovery", axis, "scene-a", offset + 1, 0, 10.0),
                    _row(base, "discovery", axis, "scene-b", offset + 2, 1, 11.0),
                    _row(base, "confirmation", axis, "scene-c", offset + 3, 2, 12.0),
                ]
            )
    panels = [
        {"case_id": row["case_id"], "panel_path": row["panel_path"], "panel_sha256": "a" * 64, "metric_row_sha256": "b" * 64}
        for row in rows
    ]
    first, summary = select_representative_cases(rows, panels)
    second, _ = select_representative_cases(list(reversed(rows)), list(reversed(panels)))
    assert first == second
    assert len(first) == 18
    assert summary["unique_case_ids"] == 18
    assert set(summary["coverage"].values()) == {1, 2}

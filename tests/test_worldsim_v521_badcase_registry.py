from __future__ import annotations

from motion_proj.worldsim_v521.badcase_registry import (
    build_leaderboards,
    build_registry,
    case_id,
    event_id,
    freeze_thresholds,
    panel_union,
)


def row(base: str, scene: str, frame: int, psnr: float) -> dict:
    metric = lambda value, pixels: {
        "status": "done", "pixel_count": pixels, "psnr": value,
        "ssim": 0.8, "lpips_alex": 0.2,
    }
    return {
        "base": base, "scene": scene, "frame": frame, "sample_token": None,
        "canonical_sample_index": frame, "camera": 0, "partition": "discovery",
        "prediction_sha256": "p", "target_sha256": "t", "dynamic_mask_sha256": "m",
        "metrics": {
            "global": metric(psnr, 1000),
            "static": metric(psnr + 5.0, 800),
            "actor": metric(psnr - 1.0, 100),
            "boundary": metric(psnr - 2.0, 80),
            "geometry": {"status": "undefined_no_comparable_base_depth"},
            "temporal": {"status": "deferred_to_window_table"},
        },
        "actor_context": {},
    }


def test_case_and_event_ids_are_axis_separated() -> None:
    sample = row("adgs", "scene-a", 2, 10.0)
    identifier = case_id(sample)
    assert identifier == case_id(sample)
    assert identifier.startswith("BC-ADGS-") and len(identifier.split("-")[-1]) == 12
    assert event_id(identifier, "ACTOR_RGB") != event_id(identifier, "BOUNDARY")


def test_thresholds_are_scene_balanced_and_registry_keeps_multi_axis() -> None:
    rows = [
        row("adgs", "scene-a", 2, 10.0), row("adgs", "scene-a", 7, 20.0),
        row("adgs", "scene-b", 2, 11.0), row("adgs", "scene-b", 7, 21.0),
    ]
    minimums = {"global": 1, "static": 1, "actor": 64, "boundary": 64}
    thresholds = freeze_thresholds(rows, minimums)
    assert thresholds["adgs"]["global"]["valid_scenes"] == 2
    leaderboards = build_leaderboards(rows, [], minimums)
    selected = panel_union(leaderboards, 120)
    registry = build_registry(rows, [], thresholds, minimums, selected)
    worst = next(item for item in registry if item["frame"] == 2 and item["scene"] == "scene-a")
    assert set(worst["failure_axes"]) == {"GLOBAL_RGB", "ACTOR_RGB", "BOUNDARY"}
    assert "B-MIXED" in worst["failure_class"]
    assert set(worst["event_ids"]) == set(worst["failure_axes"])


def test_leaderboard_has_severity_and_scene_coverage_tables() -> None:
    rows = [row("streetgs", "scene-a", frame, 10.0 + frame) for frame in range(12)]
    rows += [row("streetgs", "scene-b", 20 + frame, 30.0 + frame) for frame in range(4)]
    minimums = {"global": 1, "static": 1, "actor": 64, "boundary": 64}
    boards = build_leaderboards(rows, [], minimums)["streetgs"]["GLOBAL_RGB"]
    assert boards["k"] == 12
    assert len(boards["severity_topk"]) == 12
    assert len(boards["scene_coverage_topk"]) == 4
    assert sum(item["scene"] == "scene-a" for item in boards["scene_coverage_topk"]) == 2


def test_confirmation_registry_reuses_thresholds_without_refit() -> None:
    rows = [row("adgs", "scene-a", 2, 10.0), row("adgs", "scene-b", 2, 11.0)]
    minimums = {"global": 1, "static": 1, "actor": 64, "boundary": 64}
    thresholds = freeze_thresholds(rows, minimums)
    boards = build_leaderboards(rows, [], minimums)
    registry = build_registry(
        rows, [], thresholds, minimums, panel_union(boards, 120),
        split_role="confirmation", evidence_tier="C",
    )
    assert registry
    assert all(item["split_role"] == "confirmation" for item in registry)
    assert all(item["evidence_tier"] == "C" for item in registry)
    assert all(item["confirmation_verdict"] == "pending_class_verdict" for item in registry)

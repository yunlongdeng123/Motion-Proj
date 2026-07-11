import json

import torch
from omegaconf import OmegaConf

from motion_proj.auditor.state import Track
from motion_proj.eval.projection_inspection import run_experiment, summarize_reviews
from motion_proj.eval.synthetic_corrupt_nuscenes import corrupt_track, select_track_index


def _case(case_id: str, index: int, corruption: str = "center_impulse"):
    return {
        "case_id": case_id,
        "case_index": index,
        "source": "synthetic",
        "corruption": corruption,
        "energy_decreased": True,
        "eligible_fraction": 0.75,
    }


def _settings():
    return {"minimum_reasonable_rate": 0.70}


def test_summarize_reviews_accepts_when_threshold_met():
    cases = [_case(f"synthetic-{index:03d}", index) for index in range(10)]
    reviews = [
        {"case_id": row["case_id"], "verdict": "reasonable", "reviewer": "test"}
        for row in cases
    ]
    summary = summarize_reviews(cases, reviews, _settings())
    assert summary["reasonable_rate"] == 1.0
    assert summary["acceptance"]["accepted"]
    assert summary["acceptance"]["pending_case_ids"] == []


def test_summarize_reviews_excludes_borderline_from_rate():
    cases = [_case("synthetic-000", 0), _case("synthetic-001", 1)]
    reviews = [
        {"case_id": "synthetic-000", "verdict": "reasonable", "reviewer": "test"},
        {"case_id": "synthetic-001", "verdict": "borderline", "reviewer": "test"},
    ]
    summary = summarize_reviews(cases, reviews, _settings())
    assert summary["reasonable_rate"] == 1.0
    assert summary["borderline_cases"] == 1
    assert summary["acceptance"]["all_cases_reviewed"]
    assert summary["acceptance"]["accepted"]


def test_aggregate_only_preserves_export_provenance(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "work_dir": str(tmp_path),
            "seed": 20260711,
            "experiment": {
                "task_id": "P1-PROJECTION-01",
                "num_cases": 1,
                "minimum_reasonable_rate": 0.7,
            },
        }
    )
    run_id = "p1-test"
    run_dir = tmp_path / run_id
    (run_dir / "cases").mkdir(parents=True)
    (run_dir / "resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    (run_dir / "cases" / "synthetic-000.json").write_text(
        json.dumps(_case("synthetic-000", 0)),
        encoding="utf-8",
    )
    (run_dir / "reviews.jsonl").write_text(
        json.dumps(
            {
                "case_id": "synthetic-000",
                "verdict": "reasonable",
                "reviewer": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "command": ["motionproj-inspect"],
        "config_fingerprint": "export-fingerprint",
        "cache_fingerprint": "not-applicable:projection-target-manual-v1",
        "seed": 20260711,
        "git": {"commit": "export-commit"},
        "environment": {},
        "data_split": "test",
        "parent_run_id": None,
        "started_at": "2026-07-11T00:00:00+00:00",
        "ended_at": "2026-07-11T00:01:00+00:00",
        "exit_reason": "awaiting_reviews",
        "status": "completed",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "motion_proj.eval.projection_inspection.git_state",
        lambda _: {"commit": "aggregate-commit", "dirty": False},
    )

    _, summary = run_experiment(cfg, run_id=run_id, aggregate_only=True)

    assert summary["acceptance"]["accepted"]
    assert summary["git_commit"] == "export-commit"
    assert summary["experiment_fingerprint"] == "export-fingerprint"
    assert summary["review_aggregation"]["git_commit"] == "aggregate-commit"
    assert summary["review_fingerprint"]
    updated_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert updated_manifest["git"]["commit"] == "export-commit"
    assert updated_manifest["exit_reason"] == "acceptance_passed"


def test_corrupt_track_changes_geometry():
    track = Track(
        instance_token="t0",
        category="vehicle.car",
        xyxy=torch.tensor([[10.0, 10.0, 30.0, 30.0]] * 8),
        depth=torch.full((8,), 12.0),
        present=torch.ones(8, dtype=torch.bool),
    )
    before = track.center.clone()
    generator = torch.Generator().manual_seed(0)
    corrupt_track(track, "center_jitter", generator)
    assert not torch.allclose(track.center, before)


def test_select_track_index_prefers_most_visible():
    tracks = [
        Track("a", "vehicle.car", torch.zeros(8, 4), torch.ones(8), torch.tensor([True] * 3 + [False] * 5)),
        Track("b", "vehicle.car", torch.zeros(8, 4), torch.ones(8), torch.tensor([True] * 6 + [False] * 2)),
    ]
    assert select_track_index(tracks, case_index=0, seed=1234) == 1

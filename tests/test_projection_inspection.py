import torch

from motion_proj.auditor.state import Track
from motion_proj.eval.projection_inspection import summarize_reviews
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

from __future__ import annotations

from scripts.run_worldsim_v5_m2_actor_masks import denominator


def test_actor_mask_denominator_keeps_absent_views_and_rejections() -> None:
    views = [
        {"box_count": 3},
        {"box_count": 5},
        {"box_count": 0},
        {"box_count": 6},
        {"box_count": 9},
        {"box_count": 0},
    ]
    masks = [{"accepted": True} for _ in range(22)] + [{"accepted": False}]
    assert denominator(masks, views) == {
        "view_count": 6,
        "available_view_count": 4,
        "unavailable_view_count": 2,
        "actor_mask_count": 23,
        "accepted_actor_mask_count": 22,
        "rejected_actor_mask_count": 1,
    }

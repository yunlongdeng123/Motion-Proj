from __future__ import annotations

from motion_proj.preference.selective_order import build_condition_partial_order


def _strict(a: str, b: str) -> dict:
    return {
        "relation": "strict", "candidate_a": a, "candidate_b": b,
        "winner_candidate_id": a, "loser_candidate_id": b,
    }


def test_cycle_invalidates_entire_condition_instead_of_dropping_weak_edge() -> None:
    graph = build_condition_partial_order(
        "condition", [_strict("a", "b"), _strict("b", "c"), _strict("c", "a")]
    )
    assert graph["status"] == "invalid_cycle"
    assert set(graph["cycle_nodes"]) == {"a", "b", "c"}
    assert graph["reduced_edges"] == []


def test_transitive_reduction_is_only_applied_to_dag() -> None:
    graph = build_condition_partial_order(
        "condition", [_strict("a", "b"), _strict("b", "c"), _strict("a", "c")]
    )
    assert graph["status"] == "strict"
    assert ["a", "c"] not in graph["reduced_edges"]

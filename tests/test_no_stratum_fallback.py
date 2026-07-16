from __future__ import annotations

import torch

from motion_proj.preference.paired_tracks import select_common_queries


def test_insufficient_dynamic_stratum_is_invalid_without_fallback() -> None:
    frame = torch.zeros(3, 8, 8)
    flow = torch.zeros(2, 8, 8, 2)
    confidence = torch.ones(2, 8, 8)
    settings = {
        "min_confidence": 0.5,
        "background_queries": 8,
        "dynamic_queries": 8,
        "minimum_background_queries": 4,
        "minimum_dynamic_queries": 4,
        "min_distance": 2.0,
        "background_residual_quantile": 0.50,
        "dynamic_residual_quantile": 0.75,
        "texture_quantile": 0.0,
    }

    result = select_common_queries(frame, flow, confidence, torch.zeros_like(flow), settings)

    assert not result.valid
    assert result.diagnostics["fallback_used"] is False
    assert result.diagnostics["dynamic_query_count"] == 0
    assert "insufficient_dynamic_queries" in result.diagnostics["invalid_reasons"]

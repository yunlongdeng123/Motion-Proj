"""Aggregation helpers that deliberately keep all six pilot dimensions separate."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .schema import DIMENSIONS, validate_result


def aggregate_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    status_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    result_count = 0
    for result in results:
        validate_result(result)
        result_count += 1
        model_id = str(result["model_id"])
        status_counts[model_id][str(result["status"])] += 1
        for dimension in DIMENSIONS:
            metric = result["metrics"][dimension]
            if metric["status"] == "scored":
                buckets[(model_id, dimension)].append(float(metric["value"]))

    model_ids = sorted({model_id for model_id, _ in buckets} | set(status_counts))
    models: dict[str, Any] = {}
    for model_id in model_ids:
        dimension_rows: dict[str, Any] = {}
        for dimension in DIMENSIONS:
            values = buckets.get((model_id, dimension), [])
            dimension_rows[dimension] = {
                "count": len(values),
                "mean": (sum(values) / len(values)) if values else None,
            }
        models[model_id] = {
            "status_counts": dict(sorted(status_counts[model_id].items())),
            "dimensions": dimension_rows,
        }
    return {
        "schema_version": "worldsim_v75_cfbench_aggregate_v1",
        "result_count": result_count,
        "no_composite_score": True,
        "models": models,
    }

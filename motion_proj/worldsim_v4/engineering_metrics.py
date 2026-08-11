"""WorldSim V4 生产工程指标的可审计派生。"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np


PHASES = ("prepare", "train", "semantic", "asset", "repair", "compile", "render", "eval")
TERMINAL_STATES = {"done", "failed", "blocked", "abstain"}


def _nonnegative(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} 必须为有限非负数")
    return number


def derive_engineering_record(record: Mapping[str, Any]) -> dict[str, Any]:
    status = record.get("status")
    if status not in TERMINAL_STATES:
        raise ValueError(f"非法 terminal 状态：{status}")
    timings = record.get("timings_seconds", {})
    if not isinstance(timings, Mapping):
        raise ValueError("timings_seconds 必须是 mapping")
    unknown = sorted(set(timings) - set(PHASES))
    if unknown:
        raise ValueError(f"未知 timing phase：{unknown}")
    phase = {name: _nonnegative(timings.get(name, 0.0), name) for name in PHASES}
    total = float(sum(phase.values()))
    requested = int(record.get("requested_edits", 0))
    accepted = int(record.get("quality_accepted_edits", 0))
    source_clips = int(record.get("source_clips", 0))
    valid_clips = int(record.get("valid_edited_clips", 0))
    for label, value in (("requested_edits", requested), ("quality_accepted_edits", accepted), ("source_clips", source_clips), ("valid_edited_clips", valid_clips)):
        if value < 0:
            raise ValueError(f"{label} 必须非负")
    if accepted > requested or (source_clips == 0 and valid_clips > 0):
        raise ValueError("production count 不一致")
    actual = _nonnegative(record.get("actual_runtime_seconds", total), "actual_runtime_seconds")
    ideal = _nonnegative(record.get("ideal_single_pass_seconds", total), "ideal_single_pass_seconds")
    rerun = _nonnegative(record.get("rerun_seconds", 0.0), "rerun_seconds")
    full = _nonnegative(record.get("full_rerun_seconds", total), "full_rerun_seconds")
    gpu_hours = _nonnegative(record.get("gpu_hours", total / 3600.0), "gpu_hours")
    output = dict(record)
    output["timings_seconds"] = phase
    output["total_seconds"] = total
    output["timing_ratios"] = {name: (value / total if total else None) for name, value in phase.items()}
    output["valid_edit_yield"] = accepted / requested if requested else None
    output["counterfactual_expansion_ratio"] = valid_clips / source_clips if source_clips else None
    output["retry_amplification"] = actual / ideal if ideal else None
    output["resume_efficiency"] = 1.0 - rerun / full if full else None
    output["gpu_hours_per_accepted_clip"] = gpu_hours / accepted if accepted else None
    return output


def summarize_engineering(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [derive_engineering_record(record) for record in records]
    attempted = len(rows)
    completed = sum(row["status"] == "done" for row in rows)
    requested = sum(int(row.get("requested_edits", 0)) for row in rows)
    accepted = sum(int(row.get("quality_accepted_edits", 0)) for row in rows)
    source_clips = sum(int(row.get("source_clips", 0)) for row in rows)
    valid_clips = sum(int(row.get("valid_edited_clips", 0)) for row in rows)
    total_gpu_hours = sum(float(row.get("gpu_hours", row["total_seconds"] / 3600.0)) for row in rows)
    total_seconds = sum(float(row["total_seconds"]) for row in rows)
    frame_times = [float(value) for row in rows for value in row.get("frame_times_seconds", [])]
    return {
        "attempted": attempted,
        "states": {state: sum(row["status"] == state for row in rows) for state in TERMINAL_STATES},
        "pipeline_success_rate": completed / attempted if attempted else None,
        "valid_edit_yield": accepted / requested if requested else None,
        "counterfactual_expansion_ratio": valid_clips / source_clips if source_clips else None,
        "scene_throughput_per_gpu_day": completed / (total_gpu_hours / 24.0) if total_gpu_hours else None,
        "edit_throughput_per_gpu_hour": accepted / total_gpu_hours if total_gpu_hours else None,
        "gpu_hours_per_accepted_clip": total_gpu_hours / accepted if accepted else None,
        "total_seconds": total_seconds,
        "frame_time_p50_seconds": float(np.percentile(frame_times, 50.0)) if frame_times else None,
        "frame_time_p95_seconds": float(np.percentile(frame_times, 95.0)) if frame_times else None,
        "fps": len(frame_times) / sum(frame_times) if frame_times and sum(frame_times) else None,
    }

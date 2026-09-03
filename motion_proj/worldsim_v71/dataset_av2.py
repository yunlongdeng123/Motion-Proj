"""AV2 V7.1 冻结 zero-shot 数据入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from motion_proj.worldsim_v7.av2_four_action_compiler import compile_log


RIGID_AV2_CATEGORIES = (
    "REGULAR_VEHICLE",
    "BUS",
    "BOX_TRUCK",
    "TRUCK",
    "VEHICULAR_TRAILER",
    "SCHOOL_BUS",
    "ARTICULATED_BUS",
    "LARGE_VEHICLE",
)


def load_frozen_av2_cohort(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    logs = payload.get("logs", [])
    if not logs:
        raise ValueError("AV2 cohort 为空")
    return payload


def compile_av2_log_v71(
    log_dir: Path,
    compiler_config: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    compiled = compile_log(log_dir, compiler_config, device, include_diagnostics=True)
    diagnostics = compiled["compiled"]["diagnostics"]
    return [
        {
            "log_id": log_dir.name,
            "row": row,
            "diagnostics": diagnostics[row["track_id"]],
        }
        for row in compiled["actor_rows"]
    ]

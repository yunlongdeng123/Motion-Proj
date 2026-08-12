"""冻结并校验 S1 evaluation-only 分区。"""

from __future__ import annotations

from typing import Any, Mapping


SUPPORTED_EVALUATION_PARTITIONS = ("development", "heldout")


def resolve_evaluation_frames(
    config: Mapping[str, Any], partition: str
) -> list[int]:
    if partition not in SUPPORTED_EVALUATION_PARTITIONS:
        raise ValueError(f"不支持的 evaluation partition: {partition}")
    split = config["split"]
    frames = [int(value) for value in split[f"{partition}_frames"]]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError(f"{partition} evaluation frames 为空或重复")
    other = "heldout" if partition == "development" else "development"
    other_frames = {int(value) for value in split[f"{other}_frames"]}
    overlap = set(frames) & other_frames
    if overlap:
        raise ValueError(f"development/heldout frames 重叠: {sorted(overlap)}")
    return frames


def manifest_evaluation_partition(manifest: Mapping[str, Any]) -> str:
    """旧版 V3.3 manifest 没有该字段，唯一合法解释是 heldout。"""

    partition = str(manifest.get("evaluation_partition", "heldout"))
    if partition not in SUPPORTED_EVALUATION_PARTITIONS:
        raise ValueError(f"manifest evaluation partition 非法: {partition}")
    return partition


def resolve_forbidden_optimization_frames(
    config: Mapping[str, Any], *, phase: str, evaluation_partition: str
) -> set[int]:
    if phase not in {"smoke", "formal"}:
        raise ValueError(f"不支持的 S1 phase: {phase}")
    heldout = set(resolve_evaluation_frames(config, "heldout"))
    if phase == "formal" and evaluation_partition == "development":
        return heldout | set(resolve_evaluation_frames(config, "development"))
    return heldout

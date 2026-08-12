"""V4 语义优化分区合同。"""

from __future__ import annotations

from typing import Any, Mapping


class SemanticSplitError(RuntimeError):
    """语义优化读取了 development/heldout 或分区声明漂移。"""


def resolve_semantic_split(config: Mapping[str, Any]) -> dict[str, list[int]]:
    split = config["split"]
    development = sorted({int(value) for value in split.get("development_frames", [])})
    heldout = sorted({int(value) for value in split["heldout_frames"]})
    overlap = set(development) & set(heldout)
    if overlap:
        raise SemanticSplitError(f"development/heldout frames 重叠: {sorted(overlap)}")
    return {"development": development, "heldout": heldout}


def validate_prompt_optimization_split(
    config: Mapping[str, Any], prompt_manifest: Mapping[str, Any]
) -> set[int]:
    split = resolve_semantic_split(config)
    manifest_development = sorted(
        {int(value) for value in prompt_manifest.get("development_frames", [])}
    )
    manifest_heldout = sorted(
        {int(value) for value in prompt_manifest.get("heldout_frames", [])}
    )
    if manifest_development != split["development"]:
        raise SemanticSplitError("prompt development frame 集漂移")
    if manifest_heldout != split["heldout"]:
        raise SemanticSplitError("prompt heldout frame 集漂移")
    if split["development"] and not prompt_manifest.get("development_excluded"):
        raise SemanticSplitError("prompt 未声明 development_excluded")
    if not prompt_manifest.get("heldout_excluded"):
        raise SemanticSplitError("prompt 未声明 heldout_excluded")
    forbidden = set(split["development"]) | set(split["heldout"])
    train = [int(value) for value in prompt_manifest.get("train_frames", [])]
    leaked = sorted(set(train) & forbidden)
    if leaked:
        raise SemanticSplitError(f"prompt train frames 命中冻结分区: {leaked}")
    return forbidden


def forbidden_frames_from_mask_manifest(
    config: Mapping[str, Any], mask_manifest: Mapping[str, Any]
) -> set[int]:
    split = resolve_semantic_split(config)
    if sorted(int(value) for value in mask_manifest.get("development_frames", [])) != split[
        "development"
    ]:
        raise SemanticSplitError("mask development frame 集漂移")
    if sorted(int(value) for value in mask_manifest.get("heldout_frames", [])) != split[
        "heldout"
    ]:
        raise SemanticSplitError("mask heldout frame 集漂移")
    if split["development"] and not mask_manifest.get("development_excluded"):
        raise SemanticSplitError("mask 未声明 development_excluded")
    if not mask_manifest.get("heldout_excluded"):
        raise SemanticSplitError("mask 未声明 heldout_excluded")
    forbidden = set(split["development"]) | set(split["heldout"])
    leaked = sorted(
        {int(row["frame"]) for row in mask_manifest.get("masks", [])} & forbidden
    )
    if leaked:
        raise SemanticSplitError(f"mask rows 命中冻结分区: {leaked}")
    return forbidden

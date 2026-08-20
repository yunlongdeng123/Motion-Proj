"""V5.2.1 frozen-selection badcase panel builder。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from motion_proj.worldsim_v4.region_masks import RegionMaskProtocol, build_baseline_region_masks

from .census import CensusError, sha256_file


TILE_SIZE = (400, 225)
TITLE_HEIGHT = 28


def _rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        value = image.convert("RGB")
        if value.size != (800, 450):
            value = value.resize((800, 450), Image.Resampling.LANCZOS)
        return np.asarray(value, dtype=np.uint8)


def _mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        value = image.convert("L")
        if value.size != (800, 450):
            value = value.resize((800, 450), Image.Resampling.NEAREST)
        return np.asarray(value) > 0


def _residual(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    residual = np.mean(np.abs(target.astype(np.float32) - prediction.astype(np.float32)), axis=-1)
    normalized = np.clip(residual * 4.0, 0.0, 255.0).astype(np.uint8)
    return np.stack([normalized, np.zeros_like(normalized), 255 - normalized], axis=-1)


def _overlay(target: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    value = target.astype(np.float32).copy()
    tint = np.asarray(color, dtype=np.float32)
    value[mask] = value[mask] * 0.35 + tint * 0.65
    return np.clip(value, 0, 255).astype(np.uint8)


def _canvas(tiles: Sequence[tuple[str, np.ndarray]], output: str | Path) -> None:
    if not tiles:
        raise CensusError("panel tiles 为空")
    canvas = Image.new("RGB", (TILE_SIZE[0] * len(tiles), TILE_SIZE[1] + TITLE_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(tiles):
        tile = Image.fromarray(array, mode="RGB").resize(TILE_SIZE, Image.Resampling.LANCZOS)
        x = index * TILE_SIZE[0]
        canvas.paste(tile, (x, TITLE_HEIGHT))
        draw.text((x + 8, 7), label, fill="black")
    canvas.save(output)


def build_view_panel(
    *,
    target_path: str | Path,
    prediction_paths: Mapping[str, str | Path],
    dynamic_mask_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    if set(prediction_paths) != {"adgs", "streetgs"}:
        raise CensusError("matched panel 必须同时有 adgs/streetgs")
    target = _rgb(target_path)
    predictions = {base: _rgb(path) for base, path in prediction_paths.items()}
    dynamic = _mask(dynamic_mask_path)
    boundary = build_baseline_region_masks(
        dynamic, np.zeros_like(dynamic), protocol=RegionMaskProtocol(boundary_radius_pixels=3)
    )["boundary"]
    tiles = [
        ("GT", target),
        ("AD-GS", predictions["adgs"]),
        ("StreetGS", predictions["streetgs"]),
        ("AD-GS residual x4", _residual(target, predictions["adgs"])),
        ("StreetGS residual x4", _residual(target, predictions["streetgs"])),
        ("dynamic union", _overlay(target, dynamic, (255, 0, 0))),
        ("boundary L1 r=3", _overlay(target, boundary, (255, 255, 0))),
    ]
    _canvas(tiles, output)
    return {
        "panel_path": str(Path(output).resolve()),
        "panel_sha256": sha256_file(output),
        "layout": [label for label, _ in tiles],
        "residual_visual_scale": 4.0,
        "geometry_tile_status": "omitted_undefined_no_comparable_base_depth",
    }


def build_temporal_panel(
    *,
    target_paths: Sequence[str | Path],
    prediction_paths: Sequence[str | Path],
    frames: Sequence[int],
    base: str,
    output: str | Path,
) -> dict[str, Any]:
    if len(target_paths) != 2 or len(prediction_paths) != 2 or len(frames) != 2:
        raise CensusError("temporal panel 必须正好两个 member")
    targets = [_rgb(path) for path in target_paths]
    predictions = [_rgb(path) for path in prediction_paths]
    tiles = []
    for frame, target, prediction in zip(frames, targets, predictions):
        tiles.extend(
            [
                (f"GT f{frame:03d}", target),
                (f"{base} f{frame:03d}", prediction),
                (f"residual f{frame:03d} x4", _residual(target, prediction)),
            ]
        )
    _canvas(tiles, output)
    return {
        "panel_path": str(Path(output).resolve()),
        "panel_sha256": sha256_file(output),
        "layout": [label for label, _ in tiles],
        "classification_caveat": "unwarped_temporal_proxy_only",
    }

"""Explicit foreground masks for the paper reproduction; RGB is not a mask."""
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch


@lru_cache(maxsize=512)
def _read_mask(filename, width, height):
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Paper profile requires an explicit binary foreground mask: {path}")
    image = Image.open(path).convert("L")
    values = np.unique(np.asarray(image))
    if not set(values.tolist()).issubset({0, 1, 255}):
        raise ValueError(f"Expected binary mask, got values {values[:12]} in {path}")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return torch.from_numpy((np.asarray(image).copy() > 0).astype(np.float32))[None]


def load_dynamic_mask(source_path, camera):
    """Load appearance mask, optionally unioned with a separately predicted shadow.

    Camera indexing follows the converter (0=FRONT). Geometry initialization must
    continue to use actor-only masks: a shadow is not an actor LiDAR surface.
    Missing actor masks are errors; an explicitly saved all-zero mask is valid.
    """
    path = Path(camera.image_path)
    match = re.fullmatch(r"image_(\d+)", path.parent.name)
    if not match:
        raise ValueError(f"Cannot infer Waymo camera index from {path}")
    index = int(match.group(1))
    root = Path(source_path) / "dynamic_masks"
    filename = root / f"dynamic_mask_{index}" / f"{path.stem}.png"
    mask = _read_mask(str(filename), camera.image_width, camera.image_height)
    shadow = root / f"shadow_mask_{index}" / f"{path.stem}.png"
    if shadow.exists():
        mask = torch.maximum(mask, _read_mask(str(shadow), camera.image_width, camera.image_height))
    return mask.to(device=camera.world_view_transform.device)

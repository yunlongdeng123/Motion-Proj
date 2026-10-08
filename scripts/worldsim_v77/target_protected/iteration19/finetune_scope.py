"""Select native DriveEditor main U-Net parameters and census a checkpoint header.

This module is deliberately independent of torch so its CLI never loads model
weights or touches a GPU. ``selected`` can replace r7 train_control.selected.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


MAIN_UNET_PREFIX = "model.diffusion_model."
OFFICIAL_FROZEN_NAME_PARTS = ("time_embed.", "label_emb.")


def selected(name: str) -> bool:
    """Match the official main U-Net trainable names in the full engine state.

    DriveEditor freezes names containing ``_3d``, ``time_embed.`` or
    ``label_emb.`` in ``DiffusionEngine.configure_model``. The prefix restricts
    selection to the main U-Net, excluding first-stage VAE and conditioners.
    """
    return (
        name.startswith(MAIN_UNET_PREFIX)
        and "_3d" not in name
        and not any(part in name for part in OFFICIAL_FROZEN_NAME_PARTS)
    )


def read_safetensors_header(path: Path) -> dict:
    """Read only the safetensors JSON header, never tensor data."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors header length: {path}")
        (length,) = struct.unpack("<Q", prefix)
        if length == 0 or length > 100_000_000:
            raise ValueError(f"Invalid safetensors header length: {length}")
        payload = stream.read(length)
        if len(payload) != length:
            raise ValueError(f"Truncated safetensors header: {path}")
    header = json.loads(payload)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be a JSON object")
    return header


def census(path: Path) -> dict:
    tensors = []
    for name, entry in read_safetensors_header(path).items():
        if not selected(name):
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("shape"), list):
            raise ValueError(f"Missing tensor shape: {name}")
        shape = entry["shape"]
        if any(not isinstance(dim, int) or dim < 0 for dim in shape):
            raise ValueError(f"Invalid tensor shape: {name}")
        tensors.append(
            {
                "name": name,
                "shape": shape,
                "dtype": entry["dtype"],
                "parameters": math.prod(shape),
            }
        )
    tensors.sort(key=lambda item: item["name"])
    return {
        "checkpoint": str(path),
        "method": "safetensors header only; no tensor data or model loaded",
        "scope": "official native main SVD U-Net, excluding _3d/time_embed./label_emb.; VAE and CLIP excluded",
        "tensor_count": len(tensors),
        "parameter_count": sum(item["parameters"] for item in tensors),
        "tensors": tensors,
        "gpu_memory_fit_32gb": "untested",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Official model.safetensors")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    result = census(args.checkpoint)
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()

"""完整、原子且可验证的训练 checkpoint。"""
from __future__ import annotations

import glob
import os
import random
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from ..config import save_resolved_config
from .atomic import atomic_directory, atomic_write_json
from .fingerprint import git_state

COMPLETE = "COMPLETE"


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(root: str, step: int, backbone, optimizer, sampler, cfg,
                    config_fingerprint: str, cache_fingerprint: str, *, final: bool = False) -> str:
    name = f"step_{step:09d}" + ("_final" if final else "")
    target = os.path.join(root, name)
    if os.path.isdir(target) and is_complete_checkpoint(target, config_fingerprint, cache_fingerprint):
        return target
    if os.path.exists(target):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        os.replace(target, f"{target}.incomplete-{stamp}")
    with atomic_directory(target) as tmp:
        backbone.save_adapter(os.path.join(tmp, "adapter.safetensors"))
        torch.save(
            {"global_step": int(step), "optimizer": optimizer.state_dict(),
             "sampler": sampler.state_dict(), "rng": capture_rng_state()},
            os.path.join(tmp, "training_state.pt"),
        )
        save_resolved_config(cfg, os.path.join(tmp, "resolved.yaml"))
        atomic_write_json(
            os.path.join(tmp, "metadata.json"),
            {"schema_version": 1, "global_step": int(step), "config_fingerprint": config_fingerprint,
             "cache_fingerprint": cache_fingerprint, "git": git_state(), "final": final},
        )
        with open(os.path.join(tmp, COMPLETE), "w", encoding="utf-8") as handle:
            handle.write("ok\n")
    return target


def _metadata(path: str) -> dict | None:
    import json

    try:
        with open(os.path.join(path, "metadata.json"), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def is_complete_checkpoint(path: str, config_fingerprint: str | None = None,
                           cache_fingerprint: str | None = None) -> bool:
    required = (COMPLETE, "adapter.safetensors", "training_state.pt", "resolved.yaml", "metadata.json")
    if not all(os.path.isfile(os.path.join(path, item)) for item in required):
        return False
    meta = _metadata(path)
    if not meta:
        return False
    return ((config_fingerprint is None or meta.get("config_fingerprint") == config_fingerprint)
            and (cache_fingerprint is None or meta.get("cache_fingerprint") == cache_fingerprint))


def find_latest_checkpoint(root: str, config_fingerprint: str, cache_fingerprint: str) -> str | None:
    candidates = []
    for path in glob.glob(os.path.join(root, "step_*")):
        if is_complete_checkpoint(path, config_fingerprint, cache_fingerprint):
            meta = _metadata(path) or {}
            candidates.append((int(meta.get("global_step", -1)), path))
    return max(candidates)[1] if candidates else None


def load_checkpoint(path: str, backbone, optimizer, sampler, *, expected_config: str,
                    expected_cache: str, map_location: str = "cpu") -> int:
    if not is_complete_checkpoint(path, expected_config, expected_cache):
        raise ValueError(f"checkpoint 不完整或 fingerprint 不匹配: {path}")
    backbone.load_adapter(os.path.join(path, "adapter.safetensors"))
    try:
        state = torch.load(os.path.join(path, "training_state.pt"), map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0
        state = torch.load(os.path.join(path, "training_state.pt"), map_location=map_location)
    optimizer.load_state_dict(state["optimizer"])
    sampler.load_state_dict(state["sampler"])
    restore_rng_state(state["rng"])
    return int(state["global_step"])

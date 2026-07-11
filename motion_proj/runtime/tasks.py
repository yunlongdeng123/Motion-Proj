"""生成评估等细粒度任务的可恢复状态。"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from .atomic import atomic_write_json


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


class TaskStore:
    def __init__(self, root: str):
        self.root = root

    def path(self, checkpoint: str, seed: int, clip: str, camera: str = "CAM_FRONT") -> str:
        digest = hashlib.sha256(f"{checkpoint}\0{seed}\0{camera}\0{clip}".encode()).hexdigest()[:12]
        return os.path.join(self.root, _safe(checkpoint), str(seed), _safe(camera),
                            f"{_safe(clip)}-{digest}.json")

    def read(self, checkpoint: str, seed: int, clip: str,
             camera: str = "CAM_FRONT") -> dict[str, Any] | None:
        try:
            with open(self.path(checkpoint, seed, clip, camera), encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None

    def completed_result(self, checkpoint: str, seed: int, clip: str,
                         camera: str = "CAM_FRONT") -> dict | None:
        value = self.read(checkpoint, seed, clip, camera)
        return value.get("result") if value and value.get("status") == "completed" else None

    def mark(self, checkpoint: str, seed: int, clip: str, status: str,
             camera: str = "CAM_FRONT", **extra) -> None:
        value = {"checkpoint": checkpoint, "seed": int(seed), "clip": clip,
                 "camera": camera, "status": status,
                 "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
        atomic_write_json(self.path(checkpoint, seed, clip, camera), value)

"""cache/train/eval 共用的幂等 stage manifest。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .atomic import atomic_write_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageManifest:
    directory: str
    name: str
    fingerprint: str

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.directory, "manifest.json")

    @property
    def complete_path(self) -> str:
        return os.path.join(self.directory, "COMPLETE")

    def is_complete(self) -> bool:
        if not os.path.isfile(self.complete_path) or not os.path.isfile(self.manifest_path):
            return False
        import json

        try:
            with open(self.manifest_path, encoding="utf-8") as handle:
                data = json.load(handle)
            return data.get("status") == "completed" and data.get("fingerprint") == self.fingerprint
        except (OSError, ValueError):
            return False

    def begin(self, extra: dict[str, Any] | None = None) -> None:
        os.makedirs(self.directory, exist_ok=True)
        data = {"schema_version": 1, "stage": self.name, "fingerprint": self.fingerprint,
                "status": "running", "started_at": _now()}
        data.update(extra or {})
        atomic_write_json(self.manifest_path, data)
        if os.path.exists(self.complete_path):
            os.unlink(self.complete_path)

    def complete(self, extra: dict[str, Any] | None = None) -> None:
        import json

        with open(self.manifest_path, encoding="utf-8") as handle:
            data = json.load(handle)
        data.update(extra or {})
        data.update({"status": "completed", "completed_at": _now()})
        atomic_write_json(self.manifest_path, data)
        with open(self.complete_path, "w", encoding="utf-8") as handle:
            handle.write(self.fingerprint + "\n")

    def fail(self, reason: str) -> None:
        import json

        try:
            with open(self.manifest_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            data = {"stage": self.name, "fingerprint": self.fingerprint}
        data.update({"status": "failed", "ended_at": _now(), "reason": reason})
        atomic_write_json(self.manifest_path, data)

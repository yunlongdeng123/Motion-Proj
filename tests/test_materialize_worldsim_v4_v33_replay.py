from __future__ import annotations

import json
from pathlib import Path

from scripts.materialize_worldsim_v4_v33_replay import atomic_json


def test_atomic_json_is_canonical_and_replaces_partial(tmp_path: Path) -> None:
    output = tmp_path / "resolved.json"

    atomic_json(output, {"z": 1, "a": "中文"})

    assert output.read_text(encoding="utf-8") == '{"a":"中文","z":1}\n'
    assert list(tmp_path.glob("*.partial.*")) == []

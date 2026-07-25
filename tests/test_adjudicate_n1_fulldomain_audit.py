import json
from pathlib import Path

import pytest

from scripts.adjudicate_n1_fulldomain_audit import (
    _read_jsonl,
    _without_review_fields,
)


def test_without_review_fields_preserves_evidence() -> None:
    row = {
        "event_id": "e1",
        "front_gap_m": 12.0,
        "verdict": "FALSE_POSITIVE",
        "reviewer": "reviewer",
        "notes": "reason",
    }
    assert _without_review_fields(row) == {
        "event_id": "e1",
        "front_gap_m": 12.0,
    }


def test_read_jsonl_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(["not", "object"]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不是 JSON object"):
        _read_jsonl(path)


def test_read_jsonl_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('\n{"event_id": "e1"}\n\n', encoding="utf-8")
    assert _read_jsonl(path) == [{"event_id": "e1"}]

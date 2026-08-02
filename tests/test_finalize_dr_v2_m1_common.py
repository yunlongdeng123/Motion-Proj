import json
from pathlib import Path

import pytest

from scripts import finalize_dr_v2_m1_common as finalizer


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_native_source_accepts_complete_native_results(tmp_path: Path):
    write_json(tmp_path / "native_summary.json", {"status": "native_done"})
    rows = []
    for index in range(18):
        stage = f"native_{index:03d}"
        rows.append({"pseudo_scene": f"{index:03d}", "stage": stage})
        write_json(tmp_path / "stages" / f"{stage}.json", {"status": "done"})
    write_json(
        tmp_path / "metrics.json",
        {"native_1view_rows": rows, "native_3view_rows": rows},
    )

    summary, metrics = finalizer.validate_native_source(tmp_path)

    assert summary["status"] == "native_done"
    assert len(metrics["native_3view_rows"]) == 18


def test_validate_native_source_fails_closed_on_partial_coverage(tmp_path: Path):
    write_json(tmp_path / "native_summary.json", {"status": "native_done"})
    write_json(
        tmp_path / "metrics.json",
        {"native_1view_rows": [], "native_3view_rows": []},
    )
    with pytest.raises(RuntimeError, match="native_1view_rows=0/18"):
        finalizer.validate_native_source(tmp_path)


def test_finalize_requires_separate_result_run(tmp_path: Path):
    with pytest.raises(RuntimeError, match="result-run"):
        finalizer.finalize(tmp_path, tmp_path)

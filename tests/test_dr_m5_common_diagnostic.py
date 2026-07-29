import json

import pytest

from scripts.finalize_dr_m5_common_diagnostic import (
    mean_metrics,
    split_and_render_index,
)
from scripts import run_dr_m6_stress


@pytest.mark.parametrize(
    "processed_frame,view,expected",
    [
        (0, 0, ("train", 0)),
        (3, 2, ("train", 11)),
        (4, 0, ("test", 0)),
        (24, 0, ("test", 15)),
        (25, 0, ("train", 57)),
        (56, 0, ("test", 39)),
        (59, 2, ("train", 137)),
    ],
)
def test_split_and_render_index(processed_frame, view, expected):
    assert split_and_render_index(processed_frame, view) == expected


def test_split_and_render_index_fails_closed():
    with pytest.raises(ValueError):
        split_and_render_index(60, 0)
    with pytest.raises(ValueError):
        split_and_render_index(0, 3)


def test_mean_metrics():
    rows = [
        {"metrics": {"PSNR": 10.0, "SSIM": 0.5, "LPIPS(ALEX)": 0.4}},
        {"metrics": {"PSNR": 20.0, "SSIM": 0.9, "LPIPS(ALEX)": 0.2}},
    ]
    assert mean_metrics(rows) == {
        "PSNR": 15.0,
        "SSIM": 0.7,
        "LPIPS(ALEX)": pytest.approx(0.3),
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _patch_m4_gate(monkeypatch, tmp_path):
    m4 = tmp_path / "m4"
    _write_json(
        m4 / "summary.json",
        {
            "status": "done",
            "all_gates_passed": True,
            "official_test_mean": {"PSNR": 31.0},
        },
    )
    monkeypatch.setattr(run_dr_m6_stress, "M4_AGGREGATE", m4)


def test_m6_gate_requires_common_diagnostic_after_native_done(
    monkeypatch, tmp_path
):
    _patch_m4_gate(monkeypatch, tmp_path)
    m5 = tmp_path / "m5"
    _write_json(m5 / "terminal.json", {"status": "done", "failure": None})
    with pytest.raises(RuntimeError, match="common-observation"):
        run_dr_m6_stress.verify_upstream_gates(m5, None)


def test_m6_gate_accepts_complete_common_diagnostic(monkeypatch, tmp_path):
    _patch_m4_gate(monkeypatch, tmp_path)
    m5 = tmp_path / "m5"
    common = tmp_path / "common"
    _write_json(m5 / "terminal.json", {"status": "done", "failure": None})
    _write_json(common / "terminal.json", {"status": "done", "failure": None})
    _write_json(
        common / "summary.json",
        {"status": "done", "target_mapping_coverage": 1.0},
    )
    gate = run_dr_m6_stress.verify_upstream_gates(m5, common)
    assert gate["m5_common_status"] == "done"


def test_m6_gate_allows_common_not_run_when_native_is_blocked(
    monkeypatch, tmp_path
):
    _patch_m4_gate(monkeypatch, tmp_path)
    m5 = tmp_path / "m5"
    _write_json(
        m5 / "terminal.json",
        {"status": "blocked", "failure": {"type": "RuntimeError"}},
    )
    gate = run_dr_m6_stress.verify_upstream_gates(m5, None)
    assert gate["m5_common_status"] == "not_run_upstream_blocked"

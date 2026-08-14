from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_worldsim_v5_m1_graph_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("worldsim_v5_m1_graph_runner", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_graph_config_freezes_two_unaries_and_four_graph_arms() -> None:
    path = ROOT / "configs/worldsim_v5/m1_graph_diagnostic_scene0471_v1.yaml"
    config = MODULE.load_config(path)
    assert config["graph"]["unary_inputs"] == ["B1", "B3"]
    assert config["graph"]["arms"] == ["G0", "G1", "G2", "G3"]
    assert config["graph"]["base_model_consumed_by_graph"] is False
    assert config["graph"]["candidate_k"] == 6
    assert config["graph"]["diffusion_iterations"] == 2
    assert config["evaluation"]["automatic_validation_unlock"] is False
    assert config["evaluation"]["automatic_semantic_split_unlock"] is False
    assert config["inputs"]["unary_summary"]["sha256"] == (
        "dd8b2a9e5f09f130f948c9de2b6b8eaa5bea9ab714278bed7fa56a633dd7a22d"
    )


def test_graph_config_rejects_automatic_validation_unlock(tmp_path: Path) -> None:
    source = yaml.safe_load(
        (ROOT / "configs/worldsim_v5/m1_graph_diagnostic_scene0471_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["evaluation"]["automatic_validation_unlock"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(MODULE.GraphDiagnosticError, match="自动解锁"):
        MODULE.load_config(path)


def test_graph_config_rejects_automatic_semantic_split_unlock(tmp_path: Path) -> None:
    source = yaml.safe_load(
        (ROOT / "configs/worldsim_v5/m1_graph_diagnostic_scene0471_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["evaluation"]["automatic_semantic_split_unlock"] = True
    path = tmp_path / "bad-split.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(MODULE.GraphDiagnosticError, match="semantic split"):
        MODULE.load_config(path)


def test_metric_row_uses_frozen_threshold_and_boundary_tolerance() -> None:
    target = np.zeros((9, 9), dtype=bool)
    target[2:7, 2:7] = True
    probability = target.astype(np.float32) * 0.8 + 0.1
    metrics = MODULE._metric_row(
        probability,
        target,
        threshold=0.5,
        boundary_tolerance=1,
        ece_bins=5,
    )
    assert metrics["iou_at_frozen_threshold"] == 1.0
    assert metrics["boundary_f1"] == 1.0
    assert metrics["brier"] < 0.05


@pytest.mark.parametrize(("accepted", "abstained"), [(8, 7), (1, 14), (3, 12)])
def test_evaluation_denominator_accepts_bound_unary_counts(
    accepted: int, abstained: int
) -> None:
    rows = [
        {"frame": index, "camera_id": index % 3, "path": f"B1/{index}.npz"}
        for index in range(accepted)
    ]
    b3_rows = [dict(row) for row in rows]
    summary = {
        "accepted_evaluation_view_count": accepted,
        "abstained_evaluation_view_count": abstained,
        "evaluation_view_count": accepted + abstained,
    }
    diagnostics = {
        "accepted_evaluation_view_count": accepted,
        "abstained_evaluation_view_count": abstained,
        "evaluation_view_count": accepted + abstained,
        "evaluation_rows": {"B1": rows, "B3": b3_rows},
    }
    verified_b1, verified_b3 = MODULE._verified_evaluation_rows(
        summary, diagnostics
    )
    assert verified_b1 == rows
    assert verified_b3 == b3_rows


def test_evaluation_denominator_rejects_zero_or_mismatched_rows() -> None:
    summary = {
        "accepted_evaluation_view_count": 0,
        "abstained_evaluation_view_count": 15,
        "evaluation_view_count": 15,
    }
    diagnostics = {
        **summary,
        "evaluation_rows": {"B1": [], "B3": []},
    }
    with pytest.raises(MODULE.GraphDiagnosticError, match="denominator"):
        MODULE._verified_evaluation_rows(summary, diagnostics)

    summary["accepted_evaluation_view_count"] = 1
    summary["abstained_evaluation_view_count"] = 14
    diagnostics.update(summary)
    diagnostics["evaluation_rows"] = {
        "B1": [{"frame": 2, "camera_id": 0}],
        "B3": [{"frame": 2, "camera_id": 1}],
    }
    with pytest.raises(MODULE.GraphDiagnosticError, match="denominator"):
        MODULE._verified_evaluation_rows(summary, diagnostics)


def test_graph_runner_help_works_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.finalize_worldsim_v4_v33_scene_chain import (
    build_records,
    load_terminal_stage,
    sha256_file,
)


def test_terminal_stage_requires_exact_summary_and_no_sealed_read(tmp_path: Path) -> None:
    run = tmp_path / "semantic"
    run.mkdir()
    summary = {
        "status": "done",
        "scene": "scene-0255",
        "stage": "semantic_lift",
        "heldout_content_read": False,
        "test_quality_read": False,
    }
    (run / "stage_summary.json").write_text(json.dumps(summary))
    (run / "status.json").write_text(
        json.dumps(
            {
                "status": "done",
                "stage_summary_sha256": sha256_file(run / "stage_summary.json"),
            }
        )
    )

    result = load_terminal_stage(
        run, expected_scene="scene-0255", expected_stage="semantic_lift"
    )

    assert result["summary"]["test_quality_read"] is False


def test_build_records_uses_only_dev_base_render_and_real_target(tmp_path: Path) -> None:
    spatial = tmp_path / "spatial"
    render = spatial / "evaluation" / "artifacts" / "renders" / "f002_c0"
    render.mkdir(parents=True)
    Image.fromarray(np.zeros((45, 80, 3), dtype=np.uint8)).save(render / "base_only.png")
    summary = {
        "state": "completed",
        "decision": {"accepted": True},
        "rows": [{"frame": 2, "camera_id": 0}],
    }
    (spatial / "evaluation" / "summary.json").write_text(json.dumps(summary))
    processed = tmp_path / "processed"
    (processed / "images").mkdir(parents=True)
    (processed / "dynamic_masks" / "all").mkdir(parents=True)
    Image.fromarray(np.zeros((90, 160, 3), dtype=np.uint8)).save(
        processed / "images" / "002_0.jpg"
    )
    Image.fromarray(np.zeros((90, 160), dtype=np.uint8)).save(
        processed / "dynamic_masks" / "all" / "002_0.png"
    )
    config = {
        "scene": {
            "processed_root": str(processed),
            "model_native_width": 80,
            "model_native_height": 45,
            "development_frames": [2, 7],
        }
    }

    rows = build_records(
        spatial_run=spatial, spatial_config=config, output_dir=tmp_path / "output"
    )

    assert len(rows) == 1
    assert rows[0]["partition"] == "development"
    assert Path(rows[0]["target"]).is_file()
    assert Image.open(rows[0]["target"]).size == (80, 45)
    assert Path(rows[0]["egocar_mask"]).is_file()

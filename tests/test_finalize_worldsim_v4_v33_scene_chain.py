from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from motion_proj.worldsim_v4.v33_replay import V33ReplayError
from scripts.finalize_worldsim_v4_v33_scene_chain import (
    build_records,
    load_development_actor_masks,
    load_terminal_stage,
    sha256_file,
)


def test_terminal_stage_requires_exact_summary_and_no_sealed_read(tmp_path: Path) -> None:
    run = tmp_path / "semantic"
    run.mkdir()
    summary = {
        "task_id": "WS-V4-M1-EVIDENCE-FIELD-01",
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
                "task_id": "WS-V4-M1-EVIDENCE-FIELD-01",
                "status": "done",
                "stage_summary_sha256": sha256_file(run / "stage_summary.json"),
            }
        )
    )

    result = load_terminal_stage(
        run,
        expected_scene="scene-0255",
        expected_stage="semantic_lift",
        expected_task_id="WS-V4-M1-EVIDENCE-FIELD-01",
    )

    assert result["summary"]["test_quality_read"] is False


def test_terminal_stage_rejects_task_id_drift(tmp_path: Path) -> None:
    run = tmp_path / "semantic"
    run.mkdir()
    summary = {
        "task_id": "WS-V4-B0-MATCHED-BASELINES-01",
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
                "task_id": "WS-V4-B0-MATCHED-BASELINES-01",
                "status": "done",
                "stage_summary_sha256": sha256_file(run / "stage_summary.json"),
            }
        )
    )

    with pytest.raises(V33ReplayError, match="task_id"):
        load_terminal_stage(
            run,
            expected_scene="scene-0255",
            expected_stage="semantic_lift",
            expected_task_id="WS-V4-M1-EVIDENCE-FIELD-01",
        )


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
    Image.fromarray(np.zeros((90, 160, 3), dtype=np.uint8)).save(
        processed / "images" / "002_0.jpg"
    )
    actor_npz = tmp_path / "actor.npz"
    actor_mask = np.zeros((45, 80), dtype=bool)
    actor_mask[10:20, 30:50] = True
    np.savez_compressed(actor_npz, binary=actor_mask)
    config = {
        "scene": {
            "processed_root": str(processed),
            "model_native_width": 80,
            "model_native_height": 45,
            "development_frames": [2, 7],
        }
    }

    rows = build_records(
        spatial_run=spatial,
        spatial_config=config,
        actor_masks={
            (2, 0): {
                "mask": str(actor_npz),
                "mask_sha256": sha256_file(actor_npz),
                "positive_pixels": 200,
            }
        },
        output_dir=tmp_path / "output",
    )

    assert len(rows) == 1
    assert rows[0]["partition"] == "development"
    assert Path(rows[0]["target"]).is_file()
    assert Image.open(rows[0]["target"]).size == (80, 45)
    assert Path(rows[0]["egocar_mask"]).is_file()
    assert rows[0]["dynamic_mask_source"] == (
        "accepted_high_support_sam2_development_mask"
    )
    assert int(np.asarray(Image.open(rows[0]["dynamic_mask"]), dtype=bool).sum()) == 200


def test_load_development_actor_masks_verifies_frozen_manifest(tmp_path: Path) -> None:
    run = tmp_path / "instance"
    mask_dir = run / "eval_targets" / "artifacts" / "masks"
    mask_dir.mkdir(parents=True)
    source = mask_dir / "actor.npz"
    np.savez_compressed(source, binary=np.ones((2, 3), dtype=bool))
    manifest = {
        "evaluation_partition": "development",
        "evaluation_frames": [2],
        "optimization_forbidden": True,
        "masks": [
            {
                "role": "high_support",
                "frame": 2,
                "camera_id": 0,
                "accepted": True,
                "positive_pixels": 6,
                "mask": str(source),
                "mask_sha256": sha256_file(source),
            }
        ],
    }
    manifest_path = mask_dir / "mask_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    eval_summary = {
        "status": "done",
        "evaluation_partition": "development",
        "optimization_forbidden": True,
        "mask_manifest": str(manifest_path),
        "mask_manifest_sha256": sha256_file(manifest_path),
        "accepted_mask_count": 1,
    }
    eval_summary_path = run / "eval_targets" / "summary.json"
    eval_summary_path.write_text(json.dumps(eval_summary), encoding="utf-8")

    masks = load_development_actor_masks(
        run, instance_stage={"eval_summary_sha256": sha256_file(eval_summary_path)}
    )

    assert set(masks) == {(2, 0)}

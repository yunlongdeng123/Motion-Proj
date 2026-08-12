from __future__ import annotations

import json
from pathlib import Path

from scripts.build_worldsim_v4_v33_registration import registration_record
from motion_proj.worldsim_v4.v33_replay import sha256_file


def test_registration_record_binds_terminal_and_core_files(tmp_path: Path) -> None:
    run = tmp_path / "chain"
    run.mkdir()
    scene = "scene-0255"
    for name, payload in (
        (
            "scene_chain.json",
            {
                "schema_version": "worldsim_v4_v33_scene_chain_v1",
                "scene": scene,
                "algorithm_commit": "e6663e1",
                "base_checkpoint_sha256": "base-sha",
                "test_quality_read": False,
            },
        ),
        ("render_manifest.json", {"rows": [1]}),
        ("metrics.json", {"rows": [1]}),
        ("summary.json", {"scene": scene, "status": "done"}),
        ("manifest.json", {"files": []}),
    ):
        (run / name).write_text(json.dumps(payload))
    (run / "status.json").write_text(
        json.dumps(
            {
                "scene": scene,
                "status": "done",
                "summary_sha256": sha256_file(run / "summary.json"),
                "manifest_sha256": sha256_file(run / "manifest.json"),
                "test_quality_read": False,
            }
        )
    )

    record = registration_record(run, expected_scene=scene)

    assert record["algorithm_commit"] == "e6663e1"
    assert record["base_checkpoint_sha256"] == "base-sha"
    assert set(record["files"]) == {
        "scene_chain.json",
        "render_manifest.json",
        "metrics.json",
    }
    assert all(row["bytes"] > 0 for row in record["files"].values())

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.run_worldsim_v4_adgs_scene import (
    ADGSRunError,
    build_preprocess_commands,
    build_train_command,
    load_config,
    validate_processed,
)


def minimal_config(tmp_path: Path) -> dict:
    return {
        "implementation": {"environment": "/env/adgs", "model_config": "configs/adgs.py"},
        "training": {"modes": {"profile100": 100, "formal": 60000}, "data_device": "cuda:0"},
        "data": {"processed_root": str(tmp_path)},
    }


def test_train_command_disables_internal_test_evaluation(tmp_path: Path) -> None:
    command, model_root, iterations = build_train_command(
        minimal_config(tmp_path), tmp_path, "scene-0230", "profile100", tmp_path / "run"
    )
    assert command[0].replace("\\", "/") == "/env/adgs/bin/python"
    assert "--disable_test_evaluation" in command
    assert command[command.index("--iterations") + 1] == "100"
    assert model_root == tmp_path / "run/model"
    assert iterations == 100


def test_preprocess_flow_disables_diagnostic_visualization(tmp_path: Path) -> None:
    config = minimal_config(tmp_path)
    config.update(
        {
            "implementation": {**config["implementation"], "root": "/third/AD-GS"},
            "dependencies": {"depth_anything": {"checkpoint": "/weights/dpt.pth"}},
            "preprocess": {"depth_encoder": "vitl", "flow_step": 4, "seed": 0},
            "data": {**config["data"], "source_root": "/data/source"},
            "scenes": {"scene-0230": 179},
        }
    )
    flow_command = build_preprocess_commands(config, tmp_path, "scene-0230")[-1][1]
    assert "--disable-visualization" in flow_command


def test_validate_processed_accepts_train_only_native_scene(tmp_path: Path) -> None:
    scene = tmp_path / "scene-0230"
    for folder, suffix in (("image", ".png"), ("semantic", ".npy"), ("sky", ".npy"), ("depth", ".npy")):
        target = scene / folder
        target.mkdir(parents=True)
        for index in range(354):
            (target / f"{index:06d}{suffix}").write_bytes(b"x")
    flow = scene / "flow"
    flow.mkdir()
    (flow / "000000.npz").write_bytes(b"x")
    (scene / "points3d.ply").write_bytes(b"ply")
    (scene / "partition.json").write_text("{}", encoding="utf-8")
    np.savez(scene / "meta.npz", is_val_list=np.zeros(354, dtype=np.bool_))
    (scene / "adapter_manifest.json").write_text(
        json.dumps({"included_partitions": ["train"], "partition_image_counts": {"train": 354, "development": 0, "heldout": 0}, "image_count": 354}),
        encoding="utf-8",
    )
    audit = validate_processed(minimal_config(tmp_path), "scene-0230")
    assert audit["counts"]["image"] == 354
    assert audit["heldout_content_read"] is False


def test_validate_processed_rejects_validation_flags(tmp_path: Path) -> None:
    scene = tmp_path / "scene-0230"
    scene.mkdir()
    (scene / "adapter_manifest.json").write_text(
        json.dumps({"included_partitions": ["train"], "partition_image_counts": {"train": 354, "development": 0, "heldout": 0}, "image_count": 354}),
        encoding="utf-8",
    )
    flags = np.zeros(354, dtype=np.bool_)
    flags[-1] = True
    np.savez(scene / "meta.npz", is_val_list=flags)
    with pytest.raises(ADGSRunError, match="validation flag"):
        validate_processed(minimal_config(tmp_path), "scene-0230")


def test_load_config_requires_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(["not", "mapping"]), encoding="utf-8")
    with pytest.raises(ADGSRunError, match="mapping"):
        load_config(path)

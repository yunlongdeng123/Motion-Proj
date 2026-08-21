import json
from pathlib import Path

import numpy as np
import pytest

from motion_proj.worldsim_v6.sceneir import SceneIRError, load_sceneir, verify_sceneir, write_sceneir
from motion_proj.worldsim_v6.sceneir_adapters import (
    native_recondrive_view,
    native_streetgs_views,
    recondrive_to_sceneir,
    streetgs_to_sceneir,
)
from motion_proj.worldsim_v6.sceneir_render import compare_views, render_view


def _unit_quaternions(count: int, rng: np.random.Generator) -> np.ndarray:
    values = rng.normal(size=(count, 4)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _recondrive_fixture() -> tuple[dict, np.ndarray, dict, list[int], dict]:
    rng = np.random.default_rng(7)
    count = 19
    outputs = {
        "xyz": rng.normal(size=(count, 3)).astype(np.float32),
        "rot_maps": _unit_quaternions(count, rng),
        "scale_maps": (rng.random((count, 3)) + 0.01).astype(np.float32),
        "opacity_maps": rng.random((count, 1)).astype(np.float32),
        "sh_maps": rng.normal(size=(count, 3, 4)).astype(np.float32),
        "forward_flow": rng.normal(size=(count, 3)).astype(np.float32),
    }
    assignments = np.asarray([-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1])
    timestamps = [0, 100_000]
    poses = {
        0: {
            "class": "vehicle",
            "translation_m": np.asarray([[1.0, 2.0, 0.0], [1.1, 2.0, 0.0]], dtype=np.float32),
            "rotation_wxyz": np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "visibility": np.asarray([True, True]),
        },
        1: {
            "class": "vehicle",
            "translation_m": np.asarray([[-1.0, 0.0, 0.0], [-1.0, 0.2, 0.0]], dtype=np.float32),
            "rotation_wxyz": np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "visibility": np.asarray([True, False]),
        },
    }
    camera = {
        "camera_model": "pinhole",
        "resolution_px": [518, 280],
        "intrinsics_3x3": [[400.0, 0.0, 259.0], [0.0, 400.0, 140.0], [0.0, 0.0, 1.0]],
    }
    return outputs, assignments, poses, timestamps, camera


def _write_recondrive(path: Path) -> tuple[dict, np.ndarray]:
    outputs, assignments, poses, timestamps, camera = _recondrive_fixture()
    document, arrays = recondrive_to_sceneir(
        outputs,
        assignments,
        poses,
        timestamps_us=timestamps,
        source_sha256="a" * 64,
        source_uri="third_party://recondrive",
        reconstructor_version="unit-test",
        camera=camera,
    )
    write_sceneir(path, document, arrays)
    return outputs, assignments


def test_recondrive_round_trip_and_fresh_load(tmp_path: Path) -> None:
    outputs, assignments = _write_recondrive(tmp_path / "sceneir")
    summary = verify_sceneir(tmp_path / "sceneir")
    assert summary["actor_count"] == 2
    view = render_view(tmp_path / "sceneir", 0)
    native = native_recondrive_view(outputs)
    for group, mask in (("static", assignments == -1), ("dynamic", assignments >= 0)):
        expected = {name: value[mask] for name, value in native.items()}
        expected = {name: value[np.argsort(expected["source_indices"])] for name, value in expected.items()}
        assert compare_views(expected, view[group], atol=2e-6, rtol=2e-6)["passed"]


def test_sceneir_serialization_is_byte_deterministic(tmp_path: Path) -> None:
    _write_recondrive(tmp_path / "first")
    _write_recondrive(tmp_path / "second")
    first_files = sorted(path.relative_to(tmp_path / "first") for path in (tmp_path / "first").rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(tmp_path / "second") for path in (tmp_path / "second").rglob("*") if path.is_file())
    assert first_files == second_files
    assert all((tmp_path / "first" / relative).read_bytes() == (tmp_path / "second" / relative).read_bytes() for relative in first_files)


def test_sceneir_rejects_tamper_and_overwrite(tmp_path: Path) -> None:
    _write_recondrive(tmp_path / "sceneir")
    with pytest.raises(SceneIRError, match="已存在"):
        _write_recondrive(tmp_path / "sceneir")
    blob = next((tmp_path / "sceneir/blobs").glob("*.npy"))
    data = bytearray(blob.read_bytes())
    data[-1] ^= 1
    blob.write_bytes(data)
    with pytest.raises(SceneIRError, match="完整性"):
        load_sceneir(tmp_path / "sceneir")


def test_sceneir_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    _write_recondrive(tmp_path / "sceneir")
    manifest_path = tmp_path / "sceneir/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["../escape.npy"] = {"bytes": 0, "sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SceneIRError, match="路径穿越"):
        load_sceneir(tmp_path / "sceneir")


def test_recondrive_rejects_nonfinite_values(tmp_path: Path) -> None:
    outputs, assignments, poses, timestamps, camera = _recondrive_fixture()
    outputs["xyz"][0, 0] = np.nan
    document, arrays = recondrive_to_sceneir(
        outputs,
        assignments,
        poses,
        timestamps_us=timestamps,
        source_sha256="b" * 64,
        source_uri="third_party://recondrive",
        reconstructor_version="unit-test",
        camera=camera,
    )
    with pytest.raises(SceneIRError, match="非有限"):
        write_sceneir(tmp_path / "bad", document, arrays)


def _streetgs_fixture() -> dict:
    rng = np.random.default_rng(9)

    def model(count: int) -> dict:
        return {
            "_means": rng.normal(size=(count, 3)).astype(np.float32),
            "_scales": rng.normal(size=(count, 3)).astype(np.float32),
            "_quats": _unit_quaternions(count, rng),
            "_opacities": rng.normal(size=(count, 1)).astype(np.float32),
            "_features_dc": rng.normal(size=(count, 3)).astype(np.float32),
            "_features_rest": rng.normal(size=(count, 3, 3)).astype(np.float32),
        }

    rigid = model(7)
    rigid.update(
        {
            "points_ids": np.asarray([[0], [0], [0], [1], [1], [1], [1]], dtype=np.int64),
            "instances_trans": np.asarray(
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.1, 0.0, 0.0], [1.0, 0.2, 0.0]]],
                dtype=np.float32,
            ),
            "instances_quats": np.asarray(
                [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0], [0.9238795, 0.0, 0.0, 0.3826834]]],
                dtype=np.float32,
            ),
            "instances_fv": np.asarray([[True, True], [True, False]]),
        }
    )
    return {"models": {"Background": model(5), "RigidNodes": rigid}, "step": 30_000}


def test_streetgs_static_actor_split_and_round_trip(tmp_path: Path) -> None:
    checkpoint = _streetgs_fixture()
    document, arrays = streetgs_to_sceneir(
        checkpoint,
        source_sha256="c" * 64,
        source_uri="checkpoint://streetgs_reference",
        reconstructor_version="unit-test",
    )
    write_sceneir(tmp_path / "sceneir", document, arrays)
    actual = render_view(tmp_path / "sceneir", 100_000)
    expected = native_streetgs_views(checkpoint, 1)
    assert compare_views(expected["static"], actual["static"], atol=2e-6, rtol=2e-6)["passed"]
    assert compare_views(expected["dynamic"], actual["dynamic"], atol=2e-6, rtol=2e-6)["passed"]

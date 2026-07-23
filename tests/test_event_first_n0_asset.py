import hashlib
from pathlib import Path

import numpy as np
import pytest

from resim.event_first_n0_asset import _rotation_error, _verify_file


def test_verify_file_checks_size_and_hash(tmp_path: Path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"official-map-asset")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    record = _verify_file(path, expected, len(path.read_bytes()))

    assert record["sha256"] == expected
    assert record["size_bytes"] == 18


def test_verify_file_rejects_hash_mismatch(tmp_path: Path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA256"):
        _verify_file(path, "0" * 64)


def test_rotation_error_is_zero_for_identical_transform():
    transform = np.eye(4)

    assert _rotation_error(transform, transform) == pytest.approx(0.0)


def test_rotation_error_detects_quarter_turn():
    left = np.eye(4)
    right = np.eye(4)
    right[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]

    assert _rotation_error(left, right) == pytest.approx(np.pi / 2)

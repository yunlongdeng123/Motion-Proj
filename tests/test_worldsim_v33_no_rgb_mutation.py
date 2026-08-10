import hashlib

import numpy as np

from motion_proj.worldsim_v33.instance_field import (
    atomic_save_instance_field,
    probability_to_logit,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_instance_field_write_does_not_mutate_base_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"immutable-rgb-gaussian-checkpoint")
    before = sha256(checkpoint)
    opacity = np.asarray([0.0001, 0.98], dtype=np.float32)
    field = {
        "gaussian_id": np.asarray([0, 1], dtype=np.int64),
        "base_model": np.asarray([0, 1], dtype=np.int8),
        "base_index": np.asarray([0, 0], dtype=np.int64),
        "hard_instance_id": np.asarray([-1, 13], dtype=np.int32),
        "instance_opacity_logit": probability_to_logit(opacity),
        "instance_opacity": opacity,
        "source_semantic_score": np.asarray([0.0, 1.0], dtype=np.float32),
        "num_positive_views": np.asarray([0, 3], dtype=np.int32),
        "num_negative_views": np.asarray([0, 1], dtype=np.int32),
        "visibility_mass": np.asarray([0.0, 1.0], dtype=np.float32),
        "trainable": np.asarray([False, True]),
        "provenance": np.asarray([0, 1], dtype=np.uint8),
        "actor_instance_ids": np.asarray([13], dtype=np.int32),
        "actor_tokens": np.asarray(["token-13"], dtype="<U64"),
    }
    atomic_save_instance_field(tmp_path / "instance_field.npz", field)
    assert sha256(checkpoint) == before
    assert checkpoint.read_bytes() == b"immutable-rgb-gaussian-checkpoint"

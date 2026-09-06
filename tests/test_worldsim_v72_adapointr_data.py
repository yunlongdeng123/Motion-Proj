from __future__ import annotations

from pathlib import Path

import numpy as np

from motion_proj.worldsim_v72.baselines.adapointr_data import (
    denormalize_adapointr_output,
    load_adapointr_input,
)


def test_adapointr_inference_loader_excludes_target(tmp_path: Path) -> None:
    path = tmp_path / "actor.npz"
    np.savez_compressed(
        path,
        partial_normalized=np.ones((4, 3), dtype=np.float32),
        target_normalized=np.full((8, 3), 9.0, dtype=np.float32),
        scene_name=np.asarray("scene-a"),
        log_id=np.asarray("log-a"),
        track_id=np.asarray("actor-a"),
        scale_m=np.asarray(2.0, dtype=np.float32),
        native_input_point_count=np.asarray(3, dtype=np.int32),
    )
    partial, metadata = load_adapointr_input(path)
    assert partial.shape == (4, 3)
    assert "target" not in metadata
    assert np.allclose(denormalize_adapointr_output(partial, metadata["scale_m"]), 2.0)

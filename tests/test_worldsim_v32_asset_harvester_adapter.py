from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import Parameter

from motion_proj.worldsim_v32.asset_harvester_adapter import inject_actor_asset
from motion_proj.worldsim_v32.asset_harvester_adapter import load_asset_harvester_ply


def test_inject_actor_asset_replaces_only_selected_rigid_actor() -> None:
    rigid = SimpleNamespace(
        point_ids=torch.tensor([[0], [1], [1]], dtype=torch.long),
        _means=Parameter(torch.zeros((3, 3))),
        _scales=Parameter(torch.zeros((3, 3))),
        _quats=Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)),
        _features_dc=Parameter(torch.zeros((3, 3))),
        _features_rest=Parameter(torch.zeros((3, 2, 3))),
        _opacities=Parameter(torch.zeros((3, 1))),
        instances_size=torch.zeros((2, 3)),
        sh_degree=1,
    )
    asset = {
        "means": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        "scales": np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
        "quats": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "rgb": np.asarray([[0.2, 0.4, 0.6]], dtype=np.float32),
        "opacity": np.asarray([0.8], dtype=np.float32),
        "target_lwh": np.asarray([5.0, 2.0, 1.8], dtype=np.float32),
    }
    counts = inject_actor_asset(rigid, actor_index=1, asset=asset)
    assert counts == {"removed_gaussians": 2, "inserted_gaussians": 1}
    assert rigid.point_ids.tolist() == [[0], [1]]
    torch.testing.assert_close(rigid._means[-1], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(rigid.instances_size[1], torch.tensor([5.0, 2.0, 1.8]))


def test_load_asset_harvester_ascii_ply(tmp_path) -> None:
    path = tmp_path / "gaussians.ply"
    properties = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    header = ["ply", "format ascii 1.0", "element vertex 1"]
    header.extend(f"property float {name}" for name in properties)
    header.append("end_header")
    values = [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    path.write_text(
        "\n".join(header) + "\n" + " ".join(str(value) for value in values) + "\n",
        encoding="ascii",
    )
    asset = load_asset_harvester_ply(path)
    np.testing.assert_allclose(asset["means"], [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(asset["scales"], [[1.0, 1.0, 1.0]])
    np.testing.assert_allclose(asset["quats"], [[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(asset["rgb"], [[0.5, 0.5, 0.5]])
    np.testing.assert_allclose(asset["opacity"], [0.5])

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_dr_v2_drivestudio_edit_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_dr_v2_drivestudio_edit_smoke", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fake_rigid():
    return SimpleNamespace(
        _means=torch.tensor([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]),
        _scales=torch.zeros(3, 3),
        _quats=torch.tensor([[1.0, 0, 0, 0]] * 3),
        _features_dc=torch.zeros(3, 3),
        _features_rest=torch.zeros(3, 2, 3),
        _opacities=torch.zeros(3, 1),
        point_ids=torch.tensor([[0], [1], [0]]),
        instances_size=torch.ones(2, 3),
        instances_fv=torch.tensor([[True, True], [True, False]]),
        instances_trans=torch.zeros(2, 2, 3),
        instances_quats=torch.tensor([[[1.0, 0, 0, 0], [1.0, 0, 0, 0]]] * 2),
    )


def test_local_y_edit_is_exact_and_non_target_is_unchanged() -> None:
    rigid = fake_rigid()
    before = MODULE.non_target_hash(rigid, 0)
    edited = MODULE.move_actor_local_y(rigid, 0, 1.0)
    assert edited == 2
    assert torch.equal(rigid.instances_trans[:, 0], torch.tensor([[0.0, 1.0, 0.0]] * 2))
    assert torch.equal(rigid.instances_trans[:, 1], torch.zeros(2, 3))
    assert MODULE.non_target_hash(rigid, 0) == before


def test_quaternion_rotation_maps_local_y() -> None:
    # +90 degrees around z maps local +y to world -x.
    q = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])
    rotation = MODULE.quaternion_wxyz_to_matrix(q)[0]
    assert torch.allclose(rotation[:, 1], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)

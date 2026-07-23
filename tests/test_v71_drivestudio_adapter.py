import torch

from motion_proj.resim.drivestudio_adapter import global_actor_gaussian_mask


def test_global_actor_mask_preserves_model_identity():
    labels = torch.tensor([0, 0, 1, 1, 1, 2])
    point_ids = torch.tensor([[0], [1], [0]])
    mask = global_actor_gaussian_mask(labels, 1, point_ids, 0)
    assert mask.tolist() == [False, False, True, False, True, False]

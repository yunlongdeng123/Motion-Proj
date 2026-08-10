import numpy as np
import pytest

from motion_proj.worldsim_v32.semantic_schema import (
    AMBIGUOUS,
    CORE_POSITIVE,
    NEGATIVE,
    SEMANTIC_POSITIVE,
    binary_inner_boundary,
    classify_gaussians,
    contiguous_blocks,
    project_box_prompt,
    semantic_posterior,
    validate_actor_identity_contract,
    validate_disjoint_split,
)


def test_binary_inner_boundary_keeps_only_foreground_edge() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    expected = mask.copy()
    expected[2, 2] = False
    np.testing.assert_array_equal(binary_inner_boundary(mask), expected)


def test_binary_inner_boundary_rejects_non_image_array() -> None:
    with pytest.raises(ValueError, match="二维"):
        binary_inner_boundary(np.zeros(3, dtype=bool))


def test_contiguous_blocks_preserve_gaps() -> None:
    assert contiguous_blocks([0, 1, 2, 4, 5, 8]) == [[0, 1, 2], [4, 5], [8]]


def test_heldout_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="泄漏"):
        validate_disjoint_split([0, 1, 10], [10, 20])


def test_actor_identity_contract_accepts_exact_mapping() -> None:
    validate_actor_identity_contract(
        role="high_support",
        actor_config={
            "instance_token": "token-13",
            "rigid_model_index": 5,
            "class_name": "vehicle.car",
        },
        dataset_instance={"id": "token-13", "class_name": "vehicle.car"},
        registry_actor={
            "instance_token": "token-13",
            "rigid_model_index": 5,
            "class_name": "vehicle.car",
        },
    )


def test_actor_identity_contract_rejects_dataset_id_token_mismatch() -> None:
    with pytest.raises(ValueError, match="dataset_instance_id/token 错配"):
        validate_actor_identity_contract(
            role="high_support",
            actor_config={
                "instance_token": "token-13",
                "rigid_model_index": 5,
                "class_name": "vehicle.car",
            },
            dataset_instance={"id": "token-5", "class_name": "vehicle.car"},
            registry_actor={
                "instance_token": "token-13",
                "rigid_model_index": 5,
                "class_name": "vehicle.car",
            },
        )


def test_project_box_prompt_identity_camera() -> None:
    obj_to_world = np.eye(4)
    obj_to_world[2, 3] = 10.0
    intrinsics = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    result = project_box_prompt(
        obj_to_world=obj_to_world,
        box_size=np.asarray([2.0, 2.0, 2.0]),
        camera_to_world=np.eye(4),
        intrinsics=intrinsics,
        image_width=100,
        image_height=100,
        padding_fraction=0.0,
    )
    assert result is not None
    np.testing.assert_allclose(result, [38.88889, 38.88889, 61.11111, 61.11111], rtol=1e-5)


def test_classification_keeps_core_identity_hard() -> None:
    semantic_mass = np.asarray([0.0, 0.7, 0.4, 0.05])
    visible_mass = np.asarray([1.0, 1.0, 1.0, 1.0])
    posterior = semantic_posterior(semantic_mass, visible_mass)
    labels = classify_gaussians(
        posterior=posterior,
        semantic_mass=semantic_mass,
        positive_view_count=np.asarray([0, 3, 3, 1]),
        core_mask=np.asarray([True, False, False, False]),
        semantic_threshold=0.6,
        ambiguous_threshold=0.2,
        minimum_semantic_mass=0.1,
        minimum_positive_views=2,
    )
    np.testing.assert_array_equal(
        labels,
        np.asarray([CORE_POSITIVE, SEMANTIC_POSITIVE, AMBIGUOUS, NEGATIVE]),
    )

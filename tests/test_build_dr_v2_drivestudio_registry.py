import importlib.util
import json
import sys
import types
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_dr_v2_drivestudio_registry.py"
)
SPEC = importlib.util.spec_from_file_location("build_dr_v2_drivestudio_registry", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
if importlib.util.find_spec("omegaconf") is None:
    OMEGACONF = types.ModuleType("omegaconf")
    OMEGACONF.OmegaConf = object
    sys.modules["omegaconf"] = OMEGACONF
SPEC.loader.exec_module(MODULE)


def test_raw_chains_uses_frozen_standard_json_table(tmp_path: Path) -> None:
    rows = [
        {
            "token": "keep",
            "first_annotation_token": "a0",
            "last_annotation_token": "a1",
            "nbr_annotations": 2,
        },
        {
            "token": "skip",
            "first_annotation_token": "b0",
            "last_annotation_token": "b1",
            "nbr_annotations": 3,
        },
    ]
    (tmp_path / "instance.json").write_text(json.dumps(rows), encoding="utf-8")
    assert MODULE.raw_chains(tmp_path, {"keep"}) == {
        "keep": {
            "first_annotation_token": "a0",
            "last_annotation_token": "a1",
            "nbr_annotations": 2,
        }
    }


def test_requested_filtered_actor_is_retained_as_explicit_unavailable() -> None:
    base = {
        "schema_version": "dr-v2-drivestudio-actor-registry-v2",
        "actors": [],
        "actor_count": 0,
        "available_actor_count": 0,
        "empty_checkpoint_actor_count": 0,
    }
    base["actor_registry_sha256"] = MODULE.canonical_sha256(base)
    processed = {
        "8": {
            "id": "high-token",
            "class_name": "vehicle.car",
            "frame_annotations": {"frame_idx": [0, 1]},
        }
    }
    chains = {
        "high-token": {
            "first_annotation_token": "a0",
            "last_annotation_token": "a1",
            "nbr_annotations": 2,
        }
    }

    result = MODULE.add_requested_unavailable_actors(
        base,
        requested_tokens=["high-token"],
        processed_instances=processed,
        raw_instance_chains=chains,
        dataset_true_ids=[8],
        dataset_model_types=[1],
        ordered_init_columns=[],
        rigid_model_type=1,
    )

    actor = result["actors"][0]
    assert actor["availability"] == "unavailable_initialization_filter"
    assert actor["dataset_instance_column"] == 0
    assert actor["rigid_model_index"] is None
    assert actor["checkpoint_tensor_slice"]["gaussian_count"] == 0
    assert result["requested_unavailable_actor_count"] == 1

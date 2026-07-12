import json
from pathlib import Path

import pytest
import torch
from diffusers import UNetSpatioTemporalConditionModel
from omegaconf import OmegaConf
from safetensors.torch import load_file

from motion_proj.backbones.svd_backbone import (
    SVDBackbone,
    classify_attention_module_path,
    select_lora_module_names,
)
from motion_proj.config import ConfigError, load_config, validate_config
from motion_proj.train.trainer import Trainer


def _tiny_unet():
    return UNetSpatioTemporalConditionModel(
        sample_size=8,
        in_channels=8,
        out_channels=4,
        down_block_types=("CrossAttnDownBlockSpatioTemporal",),
        up_block_types=("CrossAttnUpBlockSpatioTemporal",),
        block_out_channels=(32,),
        addition_time_embed_dim=4,
        projection_class_embeddings_input_dim=12,
        layers_per_block=1,
        cross_attention_dim=16,
        transformer_layers_per_block=1,
        num_attention_heads=(4,),
        num_frames=2,
    )


def _cfg(scope="temporal_only"):
    return OmegaConf.create(
        {
            "sigma_floor": 1.0e-3,
            "num_frames": 2,
            "lora": {
                "enable": True,
                "rank": 16,
                "alpha": 16,
                "dropout": 0.0,
                "scope": scope,
                "projections": ["to_q", "to_k", "to_v", "to_out.0"],
            },
        }
    )


def test_temporal_and_spatial_scopes_use_full_module_paths():
    unet = _tiny_unet()
    temporal, temporal_kinds = select_lora_module_names(unet, "temporal_only")
    spatial, spatial_kinds = select_lora_module_names(unet, "spatial_only")
    all_attention, all_kinds = select_lora_module_names(unet, "all_attention")

    assert temporal
    assert spatial
    assert set(temporal).isdisjoint(spatial)
    assert all(kind == "temporal" for kind in temporal_kinds.values())
    assert all(kind == "spatial" for kind in spatial_kinds.values())
    assert all_attention == sorted(temporal + spatial)
    assert all_kinds == {**temporal_kinds, **spatial_kinds}
    assert temporal == select_lora_module_names(unet, "temporal_only")[0]
    assert all(classify_attention_module_path(name) == "temporal" for name in temporal)


def test_temporal_only_injection_and_adapter_counts_are_fail_closed(tmp_path: Path):
    backbone = SVDBackbone(_cfg())
    backbone.unet = _tiny_unet()
    backbone.unet.requires_grad_(False)
    backbone._inject_lora()

    metadata = backbone.adapter_metadata()
    backbone._set_lora_enabled(False)
    assert backbone._lora_enabled is False
    backbone._set_lora_enabled(True)
    assert backbone._lora_enabled is True
    adapter_path = tmp_path / "adapter.safetensors"
    backbone.save_adapter(str(adapter_path))
    saved = load_file(str(adapter_path))

    assert metadata["scope"] == "temporal_only"
    assert metadata["selected_module_count"] > 0
    assert metadata["temporal_module_count"] == metadata["selected_module_count"]
    assert metadata["spatial_module_count"] == 0
    assert metadata["trainable_tensor_count"] == metadata["adapter_tensor_count"]
    assert metadata["adapter_tensor_count"] == len(saved)
    assert metadata["trainable_parameter_count"] == sum(
        parameter.numel() for parameter in backbone.trainable_parameters()
    )
    assert all(
        classify_attention_module_path(name) == "temporal"
        for name in metadata["selected_module_names"]
    )


class _AmbiguousAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(4, 4)


def test_unclassified_attention_path_is_rejected():
    with torch.no_grad():
        unet = _AmbiguousAttention()
    try:
        select_lora_module_names(unet, "all_attention", ["to_q"])
    except RuntimeError as exc:
        assert "无法按完整路径分类" in str(exc)
    else:
        raise AssertionError("未分类 attention 不得静默进入 LoRA")


def test_trainer_persists_reproducible_module_manifest(tmp_path: Path):
    class _Backbone:
        def adapter_metadata(self):
            return {
                "scope": "temporal_only",
                "rank": 16,
                "selected_module_names": ["down.temporal_transformer_blocks.0.attn1.to_q"],
                "selected_module_count": 1,
                "temporal_module_count": 1,
                "spatial_module_count": 0,
                "trainable_tensor_count": 2,
                "trainable_parameter_count": 128,
                "adapter_tensor_count": 2,
            }

    trainer = Trainer.__new__(Trainer)
    trainer.work_dir = str(tmp_path)
    trainer.cfg = OmegaConf.create({"model": {"name": "svd", "lora": {"enable": True}}})
    trainer.backbone = _Backbone()
    (tmp_path / "manifest.json").write_text('{"status": "running"}', encoding="utf-8")

    trainer._record_backbone_manifest()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    selected = (tmp_path / "selected_modules.txt").read_text(encoding="utf-8").splitlines()
    assert selected == manifest["model"]["adapter"]["selected_module_names"]
    assert manifest["model"]["adapter"]["trainable_parameter_count"] == 128


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (("scope", "unknown", "lora.scope"), ("rank", 0, "lora.rank")),
)
def test_invalid_lora_config_is_rejected(field, value, message):
    cfg = load_config("configs/train/motionproj_v1.yaml")
    mutable = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    mutable.model.lora[field] = value
    with pytest.raises(ConfigError, match=message):
        validate_config(mutable)

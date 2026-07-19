import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from motion_proj.diagnostics.resim_c1_smoke import (
    build_exact_manifest,
    build_resolved_config,
    disk_budget,
    is_cuda_oom,
    new_output_directories,
    resolve_checkpoint,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_exact_manifest_rewrites_root_and_records_scene(tmp_path):
    data_root = tmp_path / "nuscenes"
    filename = "samples/CAM_FRONT/example.jpg"
    frame_paths = [filename] + [f"sweeps/CAM_FRONT/{index:02d}.jpg" for index in range(32)]
    for relative in frame_paths:
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"rgb")
    metadata = data_root / "v1.0-trainval"
    _write_json(
        metadata / "sample_data.json",
        [{"filename": filename, "token": "sd", "sample_token": "sample"}],
    )
    _write_json(
        metadata / "sample.json",
        [{"token": "sample", "scene_token": "scene"}],
    )
    _write_json(
        metadata / "scene.json",
        [{"token": "scene", "name": "scene-0001"}],
    )
    source = tmp_path / "source.json"
    _write_json(
        source,
        {
            "meta": {"data_root": "/remote", "num_clips": 9},
            "clips": [
                {
                    "img_seq": frame_paths,
                    "traj_fut": [[float(i), 0.0, 0.0] for i in range(8)],
                    "cmd": "Moving_Forward",
                    "token": 7,
                }
            ],
        },
    )

    manifest, provenance = build_exact_manifest(source, data_root, 0)

    assert manifest["meta"]["data_root"] == str(data_root.resolve())
    assert manifest["meta"]["num_clips"] == 1
    assert len(manifest["clips"]) == 1
    assert provenance["scene_name"] == "scene-0001"
    assert provenance["trajectory_shape"] == [8, 3]
    assert provenance["consumed_rgb_frame_count"] == 33


def test_resolve_checkpoint_uses_ema_tag(tmp_path):
    root = tmp_path / "checkpoint"
    state = root / "30000-ema" / "mp_rank_00_model_states.pt"
    state.parent.mkdir(parents=True)
    state.write_bytes(b"weights")
    (root / "latest").write_text("30000", encoding="utf-8")

    result = resolve_checkpoint(root, use_ema=True)

    assert result["latest"] == 30000
    assert result["resolved_tag"] == "30000-ema"
    assert result["resolved_state"] == str(state.resolve())


def test_build_resolved_config_removes_shard_selection_and_freezes_shape(tmp_path):
    template = tmp_path / "infer.yaml"
    template.write_text(
        """
args:
  load: old
  valid_data: [old.json]
  sampling_num_frames: 13
  sampling_video_size: [512, 896]
  apply_traj: false
  save_gt: true
  concat_gt_for_demo: true
data:
  target: data_waymo.WaymoDataset
  params:
    video_size: [512, 896]
    max_num_frames: 49
    fps: 10
    n_subset: 20
    ind_subset: 2
model:
  network_config:
    params:
      latent_width: 112
      latent_height: 64
      modules:
        pos_embed_config:
          params:
            height_interpolation: 2.0
            width_interpolation: 2.3333
  conditioner_config:
    params:
      emb_models:
        - params: {model_dir: old-t5}
  first_stage_config:
    params: {ckpt_path: old-vae}
""",
        encoding="utf-8",
    )
    manifest = tmp_path / "one.json"
    manifest.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    t5 = tmp_path / "t5"
    vae = tmp_path / "vae.pt"

    cfg = build_resolved_config(
        template,
        data_manifest_path=manifest,
        checkpoint_root=checkpoint,
        t5_root=t5,
        vae_path=vae,
        seed=11,
        source_rgb_frames=49,
        height=256,
        width=448,
        latent_height=32,
        latent_width=56,
        height_interpolation=1.0,
        width_interpolation=1.16665,
    )

    params = OmegaConf.to_container(cfg.data.params, resolve=True)
    assert cfg.args.load == str(checkpoint.resolve())
    assert cfg.args.use_ema is True
    assert cfg.args.sampling_num_frames == 9
    assert cfg.args.save_gt is False
    assert cfg.data.target == "data_nus.nuScenesDataset"
    assert "n_subset" not in params and "ind_subset" not in params
    assert cfg.data.params.max_num_frames == 49
    assert cfg.model.network_config.params.latent_height == 32
    assert cfg.model.network_config.params.latent_width == 56


def test_new_output_directories_only_returns_new_children(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    (root / "old").mkdir()
    before = {"old"}
    (root / "new-b").mkdir()
    (root / "new-a").mkdir()

    assert [path.name for path in new_output_directories(root, before)] == ["new-a", "new-b"]


@pytest.mark.parametrize("message", ["CUDA out of memory", "torch.OutOfMemoryError: CUDA"])
def test_is_cuda_oom(message):
    assert is_cuda_oom(message)
    assert not is_cuda_oom("ordinary failure")


def test_disk_budget_is_fail_closed(tmp_path):
    free = disk_budget(tmp_path, estimated_peak_bytes=10**18, minimum_free_bytes=1)
    assert not free["passed"]
    assert free["projected_free_bytes"] < 0

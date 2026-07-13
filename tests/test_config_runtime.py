import json
import os

import pytest
import torch
from omegaconf import OmegaConf

from motion_proj.cache.writer import CACHE_SCHEMA_VERSION, ProjectionCacheWriter
from motion_proj.config import ConfigError, cache_config_fingerprint, cache_stage_fingerprint, config_fingerprint, load_config, validate_config
from motion_proj.runtime.experiment import ExperimentRegistry, JsonlMetrics
from motion_proj.runtime.sampler import ResumableRandomSampler
from motion_proj.runtime.stage import StageManifest
from motion_proj.runtime.tasks import TaskStore


def test_config_schema_and_resume_fingerprint():
    cfg = load_config("configs/train/motionproj_v1.yaml")
    assert cfg.schema_version == 2
    base = config_fingerprint(cfg, resume_compatible=True)
    changed = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    changed.train.max_steps = int(cfg.train.max_steps) + 1
    assert config_fingerprint(changed, resume_compatible=True) == base
    assert cache_config_fingerprint(changed) == cache_config_fingerprint(cfg)
    changed.cache.max_samples = 10
    assert cache_stage_fingerprint(changed) != cache_stage_fingerprint(cfg)
    changed.cache.source = "clean"
    assert cache_config_fingerprint(changed) != cache_config_fingerprint(cfg)
    changed.train.lr = 9e-5
    assert config_fingerprint(changed, resume_compatible=True) != base
    nondeterministic_loader = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    nondeterministic_loader.train.num_workers = 1
    with pytest.raises(ConfigError, match="num_workers=0"):
        validate_config(nondeterministic_loader)
    bad = OmegaConf.create({"schema_version": 1})
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_p2_front_config_isolates_cache_and_runtime_paths():
    cfg = load_config(
        "configs/train/motionproj_front_p2.yaml",
        ["work_dir=/tmp/p2-run"],
    )

    assert cfg.data.version == "v1.0-trainval"
    assert cfg.data.split == "train"
    assert cfg.paths.cache_dir == "/root/autodl-tmp/cache/p2-front/train"
    assert cfg.paths.ckpt_dir == "/tmp/p2-run/ckpts"
    assert cfg.paths.log_dir == "/tmp/p2-run/logs"


def test_replay_v2_config_requires_base_parent_without_adapter():
    cfg = load_config("configs/replay/p2_v2_base.yaml")
    assert cfg.cache.source == "replay_v2"
    assert cfg.auditor.generated_geometry_mode == "estimated_background_motion"
    bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_readonly(bad, False)
    bad.model.lora.enable = True
    with pytest.raises(ConfigError, match="lora.enable=false"):
        validate_config(bad)


def test_v2_capacity_pilot_is_the_only_replay_v2_training_exception():
    cfg = load_config("configs/replay/p2_v2_base.yaml")
    pilot = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_readonly(pilot, False)
    pilot.train.experiment_type = "v2_capacity_pilot"
    pilot.model.lora.enable = True
    validate_config(pilot)
    pilot.model.lora.enable = False
    with pytest.raises(ConfigError, match="v2_capacity_pilot"):
        validate_config(pilot)


def test_resumable_sampler_continues_exact_position():
    data = list(range(10))
    sampler = ResumableRandomSampler(data, seed=7)
    iterator = iter(sampler)
    prefix = [next(iterator) for _ in range(4)]
    state = sampler.state_dict()
    expected_suffix = list(iterator)
    resumed = ResumableRandomSampler(data, seed=7)
    resumed.load_state_dict(state)
    assert list(resumed) == expected_suffix
    assert sorted(prefix + expected_suffix) == list(range(10))


def test_atomic_cache_rejects_partial_and_preserves_stale(tmp_path):
    writer = ProjectionCacheWriter(str(tmp_path), store="latent", fingerprint="abc")
    partial = tmp_path / "clip"
    partial.mkdir()
    (partial / "metadata.json").write_text("{}")
    assert not writer.exists("clip")
    y = torch.zeros(2, 4, 3, 5)
    mask = torch.ones(2, 1, 3, 5)
    context = {"image_embeds": torch.zeros(1)}
    writer.write("clip", y, y, mask, {"energies": {"obj_before": 1.0}}, context)
    assert writer.exists("clip")
    assert list(tmp_path.glob("clip.stale-*"))
    with open(partial / "metadata.json", encoding="utf-8") as handle:
        assert json.load(handle)["cache_schema_version"] == CACHE_SCHEMA_VERSION


def test_stage_registry_metrics_and_tasks(tmp_path):
    stage = StageManifest(str(tmp_path / "stage"), "eval", "fp")
    stage.begin()
    assert not stage.is_complete()
    stage.complete({"tasks": 1})
    assert stage.is_complete()

    registry = ExperimentRegistry(str(tmp_path / "runs.sqlite3"))
    registry.register("r1", "queued", "fp", "/tmp/r1")
    registry.update("r1", "completed", summary={"score": 1.0})
    assert registry.list()[0]["summary"]["score"] == 1.0

    metrics = JsonlMetrics(str(tmp_path / "metrics.jsonl"))
    metrics.append(3, {"loss": 0.5})
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["step"] == 3

    tasks = TaskStore(str(tmp_path / "tasks"))
    tasks.mark("ckpt", 10, "clip", "completed", result={"metric": 2})
    assert tasks.completed_result("ckpt", 10, "clip") == {"metric": 2}

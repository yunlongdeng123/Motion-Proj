"""运行入口及真实49帧输入的CPU审计；不加载CUDA背景代码或申请GPU。"""
from __future__ import annotations

import ast
import contextlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.worldsim_v82.drivingeditor_instances import runtime, train
from scripts.worldsim_v82.drivingeditor_instances.rigid_model import RigidInstanceGaussianModel


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[2]
SCENES = ("scene0017_000_048_attempt2", "scene0008_023_071", "scene0029_069_117")


def prepared_root():
    if os.environ.get("DRIVINGEDITOR_PREPARED_ROOT"):
        return Path(os.environ["DRIVINGEDITOR_PREPARED_ROOT"])
    for ancestor in PACKAGE.parents:
        candidate = ancestor / "drivingeditor_instances" / "prepared"
        if candidate.is_dir():
            return candidate
    return None


def tiny_input(tmp_path, *, complete_masks=False):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    timestamps = np.linspace(-.48, .48, 49)
    poses = np.broadcast_to(np.eye(4), (49, 2, 4, 4)).copy()
    poses[:, 1, 0, 3] = 4
    bundle = dict(local_xyz=np.array([[.1, 0, 0], [0, .1, 0]], dtype=np.float32),
                  rgb=np.array([[.9, .1, .1], [.1, .1, .9]], dtype=np.float32),
                  scale=np.ones((2, 3), dtype=np.float32) * .02,
                  point_ids=np.array([0, 1]), instance_ids=np.array(["car-a", "car-b"]),
                  timestamps=timestamps, poses=poses, valid=np.ones((49, 2), dtype=bool),
                  box_size=np.ones((49, 2, 3)), frame_ids=np.array([f"frame_{i:03d}" for i in range(49)]),
                  mask_known=np.full((49, 2), complete_masks, dtype=bool), normalization=np.diag([.1, .1, .1, 1]))
    np.savez(prepared / "instances.npz", **bundle)
    report = dict(schema="waymo_rigid_instances_v1", geometry_ready=True, instance_count=2,
                  real_instance_points=2, frames=[{"frame_id": name} for name in bundle["frame_ids"].tolist()],
                  mask_known_pairs=int(bundle["mask_known"].sum()),
                  full_mask_supervision_ready=complete_masks, masks={"mask_known_pairs": int(bundle["mask_known"].sum())})
    (prepared / "preparation_report.json").write_text(json.dumps(report), encoding="utf-8")
    spec = {"scene_id": "cpu-contract-fixture", "prepared": str(prepared)}
    for key in ("source_repository", "source_scene", "matrix_tools"):
        location = tmp_path / key
        location.mkdir()
        spec[key] = str(location)
    for key in ("reference_config", "background_checkpoint", "environment_checkpoint"):
        location = tmp_path / (key + ".json")
        location.write_text("{}", encoding="utf-8")
        spec[key] = str(location)
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec, report, bundle


def dry_validation(path, *extra):
    # 独立进程封住GPU库导入及所有内部进程调用，区别于父测试所需的CPU进程。
    script = """import builtins,os,runpy,subprocess,sys
original_import=builtins.__import__
def guarded_import(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','tensorflow','cupy','jax'}:
  raise AssertionError('dry validation imported GPU-capable library: '+name)
 return original_import(name,*args,**kwargs)
builtins.__import__=guarded_import
def denied(*args,**kwargs): raise AssertionError('dry validation launched or inspected a process')
subprocess.run=denied
subprocess.Popen=denied
os.system=denied
sys.argv=['train','--spec',sys.argv[1],*sys.argv[2:]]
runpy.run_module('scripts.worldsim_v82.drivingeditor_instances.train',run_name='__main__')
assert 'torch' not in sys.modules
"""
    return subprocess.run([sys.executable, "-X", "utf8", "-c", script, str(path), *extra],
                          cwd=REPOSITORY, capture_output=True, text=True, timeout=30)


def test_default_cli_validates_without_torch_gpu_probe_or_output(tmp_path):
    path, _, _, _ = tiny_input(tmp_path)
    output = tmp_path / "must_not_be_created"
    result = dry_validation(path, "--output", str(output))
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "CPU_SPEC_VALIDATED_GPU_NOT_STARTED"
    assert receipt["default_background_frozen"] is True
    assert receipt["trajectory_fixed"] is True
    assert not output.exists()


def test_nonrun_rejects_incomplete_joint_mask_contract_without_gpu(tmp_path):
    path, _, _, _ = tiny_input(tmp_path)
    result = dry_validation(path, "--joint-background")
    assert result.returncode != 0
    assert "mask" in result.stderr.lower() or "标签" in result.stderr
    assert "dry validation" not in result.stderr


def test_missing_bundle_cannot_receive_geometry_ready_receipt(tmp_path):
    path, spec, _, _ = tiny_input(tmp_path)
    (Path(spec["prepared"]) / "instances.npz").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        runtime.validate_spec(path)


@pytest.mark.parametrize("field,replacement", [
    ("timestamps", np.zeros(49)),
    ("point_ids", np.array([0, 2])),
    ("rgb", np.zeros((2, 2))),
])
def test_nonrun_validation_rejects_bundle_the_actual_core_cannot_consume(tmp_path, field, replacement):
    path, spec, _, bundle = tiny_input(tmp_path)
    bundle[field] = replacement
    np.savez(Path(spec["prepared"]) / "instances.npz", **bundle)
    with pytest.raises(ValueError):
        runtime.validate_spec(path)


def test_zero_mask_long_run_is_rejected_before_gpu_admission(tmp_path):
    path, _, _, _ = tiny_input(tmp_path)
    output = tmp_path / "not_started"
    result = dry_validation(path, "--run", "--output", str(output))
    assert result.returncode != 0
    assert "geometry-only-smoke" in result.stderr
    assert "dry validation" not in result.stderr
    assert not output.exists()


def test_existing_run_status_is_preserved_when_output_is_rejected(tmp_path, monkeypatch):
    output = tmp_path / "existing_run"
    output.mkdir()
    sentinel = '{"state":"old_complete","protected":true}'
    (output / "status.json").write_text(sentinel, encoding="utf-8")
    closed = []
    slot = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(train, "require_gpu_slot", lambda path: (slot, {"allowed": True}))
    args = SimpleNamespace(output=str(output), gpu_lock="CPU-stub-only")
    with pytest.raises(FileExistsError):
        train.run(args, {}, {})
    assert (output / "status.json").read_text(encoding="utf-8") == sentinel
    assert closed == [True]


def test_python39_syntax_for_all_new_runtime_sources():
    for source in PACKAGE.glob("*.py"):
        text = source.read_text(encoding="utf-8-sig")
        ast.parse(text, filename=str(source), feature_version=(3, 9))
        if " | None" in text or "str | " in text:
            assert "from __future__ import annotations" in text, source.name


class BackgroundStub:
    """只替代GPU-only旧代码边界；实例模型和Adam均使用真实实现。"""
    def __init__(self, args):
        self._xyz = torch.nn.Parameter(torch.tensor([[0., 0, 0], [10, 10, 10]]))
        self.optimizer = torch.optim.Adam([self._xyz])
        self.restored = None

    @property
    def get_xyz(self):
        return self._xyz

    def get_xyz_SHM(self, timestamp):
        return self._xyz

    def restore(self, capture, args):
        self.restored = capture

    def prune_points(self, mask):
        self._xyz = torch.nn.Parameter(self._xyz.detach()[~mask])
        self.optimizer = torch.optim.Adam([self._xyz])


class EnvironmentStub(torch.nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor([1.0]))
        self.optimizer = torch.optim.Adam([self.values])
        self.restored = None

    def cuda(self):
        return self

    def restore(self, capture, args):
        self.restored = capture


def test_runtime_restores_actual_core_and_adam_including_refined_pose_groups(tmp_path, monkeypatch):
    _, spec, _, bundle = tiny_input(tmp_path)
    config = {"sh_degree": 1, "env_map_res": 4}
    Path(spec["reference_config"]).write_text(json.dumps(config), encoding="utf-8")
    instances = RigidInstanceGaussianModel.from_bundle(bundle, sh_degree=1, refine_poses=True,
                                                       pose_translation_limit=.031)
    optimizer = instances.make_optimizer(lr=.002, pose_lr=.0003)
    state = instances.evaluate(0.0)
    (state.xyz.square().sum() + state.features.square().sum() + state.opacity.sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    saved = instances.capture(optimizer)
    restore = {"background": ("actual-old-capture-passthrough",), "instances": saved,
               "environment": ("actual-old-env-passthrough",), "background_exclusion": {"removed_points": 17}}
    cameras = [SimpleNamespace(timestamp=float(t), image_name=str(name))
               for t, name in zip(bundle["timestamps"], bundle["frame_ids"])]
    fake_scene = ModuleType("scene")
    fake_scene.GaussianModel, fake_scene.EnvLight = BackgroundStub, EnvironmentStub
    fake_scene.Scene = lambda *args, **kwargs: SimpleNamespace(getTrainCameras=lambda: cameras)
    fake_config = ModuleType("omegaconf")
    fake_config.OmegaConf = SimpleNamespace(create=lambda value: SimpleNamespace(**value))
    monkeypatch.setitem(sys.modules, "scene", fake_scene)
    monkeypatch.setitem(sys.modules, "omegaconf", fake_config)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    sidecar = tmp_path / "new" / "sidecar"
    sidecar.mkdir(parents=True)
    # 实际旧 loader 会重算并写这个矩阵；测试替代边界必须履行同一输出合同。
    np.savetxt(sidecar / "transform_matrix.txt", bundle["normalization"])
    monkeypatch.setattr(runtime, "make_sidecar", lambda spec, output: sidecar)
    original_restore = RigidInstanceGaussianModel.from_capture
    monkeypatch.setattr(RigidInstanceGaussianModel, "from_capture", lambda checkpoint, **kwargs: original_restore(checkpoint, device="cpu"))
    _, _, background, actual, actual_optimizer, environment, exclusion = runtime.load_models(spec, tmp_path / "new", restore=restore)
    assert actual_optimizer is actual.optimizer
    assert actual.refine_poses and actual.pose_translation_limit == .031
    assert [group["name"] for group in actual_optimizer.param_groups][-2:] == ["pose_translation", "pose_rotation"]
    for name, parameter in actual.named_parameters():
        assert torch.equal(parameter, saved["state_dict"][name])
    assert actual_optimizer.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
    assert len(actual_optimizer.state) == len(optimizer.state)
    assert background.restored == restore["background"] and environment.restored == restore["environment"]
    assert not background.get_xyz.requires_grad and not environment.values.requires_grad
    assert exclusion == restore["background_exclusion"]


def test_training_failure_seeds_before_loading_and_closes_monitor_as_failed(tmp_path, monkeypatch):
    _, spec, report, _ = tiny_input(tmp_path)
    events = []
    monitors = []
    class Monitor:
        def __init__(self, *args, **kwargs):
            monitors.append(self)
            self.started = False
        def start(self):
            self.started = True
            return self
        @contextlib.contextmanager
        def phase(self, *args, **kwargs):
            yield
        def close(self, *args, **kwargs):
            events.append(("monitor_close", args, kwargs))
    fake_monitor = ModuleType("memory_monitor")
    fake_monitor.MemoryMonitor = Monitor
    monkeypatch.setitem(sys.modules, "memory_monitor", fake_monitor)
    monkeypatch.setattr(sys, "path", list(sys.path))
    slot = SimpleNamespace(close=lambda: events.append(("slot_close", (), {})))
    monkeypatch.setattr(train, "require_gpu_slot", lambda path: (slot, {"allowed": True}))
    expected = (random.Random(41).random(), np.random.RandomState(41).random_sample(),
                torch.rand((), generator=torch.Generator().manual_seed(41)).item())
    observed = []
    def fail_load(*args, **kwargs):
        observed.extend((random.random(), np.random.random(), torch.rand(()).item()))
        raise RuntimeError("controlled_CPU_boundary_failure")
    monkeypatch.setattr(train, "load_models", fail_load)
    args = SimpleNamespace(output=str(tmp_path / "run"), iterations=1, resume=None, joint_background=False,
                           seed=41, gpu_lock=str(tmp_path / "never-real-gpu.lock"), refine_poses_after=0,
                           pose_translation_limit_m=.5, box_boundary_weight=1., pose_prior_weight=.01,
                           max_features=10, geometry_only_smoke=False)
    Path(spec['reference_config']).write_text(json.dumps({'iterations': 30000}), encoding='utf-8')
    old_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        with pytest.raises(RuntimeError, match="controlled_CPU_boundary_failure"):
            train.run(args, spec, report)
    finally:
        random.setstate(old_rng[0]); np.random.set_state(old_rng[1]); torch.set_rng_state(old_rng[2])
    assert tuple(observed) == expected
    assert monitors[0].started
    close = [event for event in events if event[0] == "monitor_close"]
    assert close and (close[0][1][:1] == ("failed",) or close[0][2].get("status") == "failed")
    assert events[-1][0] == "slot_close"


@pytest.mark.parametrize("scene_name", SCENES)
def test_real49_bundles_raw_waymo_pose_named_edits_and_mask_loading(scene_name):
    root = prepared_root()
    if root is None or not (root / scene_name / "instances.npz").is_file():
        pytest.skip("真实49帧准备副本不在当前主机；设置 DRIVINGEDITOR_PREPARED_ROOT")
    location = root / scene_name
    trajectories = json.loads((location / "trajectories.json").read_text(encoding="utf-8-sig"))
    with np.load(location / "instances.npz", allow_pickle=False) as loaded:
        bundle = {key: loaded[key] for key in loaded.files}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        instances = RigidInstanceGaussianModel.from_bundle(bundle, sh_degree=0)
        assert len(instances.timestamps) == 49
        row = 24
        time = float(bundle["timestamps"][row])
        with torch.no_grad():
            baseline = instances.evaluate(time)
            assert torch.isfinite(baseline.xyz).all()
            assert torch.isfinite(baseline.scaling).all() and (baseline.scaling > 0).all()
            assert set(baseline.point_ids.tolist()).issubset(set(np.flatnonzero(bundle["valid"][row]).tolist()))
            sampled = torch.linspace(0, len(baseline.xyz)-1, min(128, len(baseline.xyz))).long()
            scale = float(bundle["world_scale"])
            normalization = bundle["normalization"].astype(np.float64)
            ego = np.asarray(trajectories["frames"][row]["frame_pose_world_from_vehicle"], dtype=np.float64)
            independent = []
            for position in sampled.tolist():
                instance_index = int(baseline.point_ids[position])
                raw = trajectories["trajectories"][instance_index]["frames"][row]["source_label"]["box"]
                heading = raw["heading"]
                rotation = np.array([[math.cos(heading), -math.sin(heading), 0],
                                     [math.sin(heading), math.cos(heading), 0], [0, 0, 1]])
                local = bundle["local_xyz"][int(baseline.row_ids[position])].astype(np.float64) / scale
                vehicle = rotation @ local + np.array([raw["center_x"], raw["center_y"], raw["center_z"]])
                world = ego[:3, :3] @ vehicle + ego[:3, 3]
                independent.append(normalization[:3, :3] @ world + normalization[:3, 3])
            np.testing.assert_allclose(baseline.xyz[sampled].numpy(), independent, atol=1e-4, rtol=1e-5)
            known_at_frame = bundle["mask_known"][row]
            active = sorted(set(baseline.point_ids.tolist()))
            preferred = [index for index in active if known_at_frame[index]]
            target = preferred[0] if preferred else active[0]
            target_name = instances.instance_ids[target]
            changed = instances.evaluate(time, {target_name: {"translation": [.1, -.2, .3]}})
            owned = baseline.point_ids == target
            torch.testing.assert_close(changed.xyz[owned], baseline.xyz[owned] + torch.tensor([.1, -.2, .3]))
            assert torch.equal(changed.xyz[~owned], baseline.xyz[~owned])
            del changed
            deleted = instances.evaluate(time, {target_name: {"delete": True}})
            assert not (deleted.point_ids == target).any()
            assert torch.equal(deleted.xyz, baseline.xyz[~owned])
            del deleted
            rotated = instances.evaluate(time, {target_name: {"yaw_radians": .4}})
            assert not torch.equal(rotated.xyz[owned], baseline.xyz[owned])
            assert torch.equal(rotated.xyz[~owned], baseline.xyz[~owned])
            assert torch.isfinite(rotated.rotation).all()
            del rotated
            if len(active) > 1:
                source = next(index for index in active if index != target)
                replaced = instances.evaluate(time, {target_name: {"replacement": instances.instance_ids[source]}})
                target_rows = replaced.row_ids[replaced.point_ids == target]
                assert torch.equal(instances.point_ids[target_rows], torch.full_like(target_rows, source))
                assert (replaced.point_ids == source).any()
                assert torch.isfinite(replaced.xyz).all()
                del replaced
            camera = SimpleNamespace(image_name=str(bundle["frame_ids"][row]), image_height=23, image_width=37)
            masks, known = runtime.masks_for_frame({"prepared": str(location)}, instances, camera)
            assert torch.equal(known, torch.from_numpy(known_at_frame))
            assert masks.shape == (instances.num_instances, 23, 37)
            assert torch.all((masks == 0) | (masks == 1)) and not (masks.sum(0) > 1).any()
            assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(old_threads)

"""GMC 队列的窄 CPU 合同；虚拟 flock，不申请GPU或启动任何训练进程。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.worldsim_v82.drivingeditor_instances import priority, queue_gmc


def queue_fixture(tmp_path, monkeypatch, *, busy_gpu=False, busy_controller=False):
    root, gpu_lock = tmp_path / "queue", tmp_path / "legacy_gpu.lock"
    gpu_lock.write_bytes(b"existing_legacy_owner_metadata")
    args = SimpleNamespace(
        spec="CPU_unit_fixture", queue_directory=str(root), gpu_lock=str(gpu_lock),
        steps=300, poll_seconds=30, max_wait_hours=15/3600, cpu_threads=1,
    )
    monkeypatch.setattr(queue_gmc, "validate_spec", lambda path: (
        {"scene_id": "scene0008_023_071"}, {"instance_count": 2, "full_mask_supervision_ready": True},
    ))
    streams = []
    def virtual_flock(stream, flags):
        assert flags == 3  # LOCK_EX | LOCK_NB：不得阻塞等待并抢现有锁。
        streams.append(stream)
        if (busy_gpu and Path(stream.name) == gpu_lock) or (busy_controller and Path(stream.name).name == "controller.lock"):
            raise BlockingIOError("unit_fixture_other_owner_holds_lock")
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=virtual_flock))
    def forbidden_launch(*args, **kwargs):
        raise AssertionError("CPU queue test must not launch a training child")
    monkeypatch.setattr(queue_gmc, "subprocess", SimpleNamespace(Popen=forbidden_launch, STDOUT=-2))
    clock = {"seconds": 0.}
    monkeypatch.setattr(queue_gmc, "time", SimpleNamespace(
        monotonic=lambda: clock["seconds"], sleep=lambda seconds: clock.update(seconds=clock["seconds"]+seconds),
    ))
    def forbidden_gpu_query(*args, **kwargs):
        raise AssertionError("busy shared lock must be rejected before any GPU query")
    monkeypatch.setattr(priority, "inspect_priority", forbidden_gpu_query)
    return args, root, gpu_lock, streams


def test_queue_waits_for_legacy_gpu_lock_without_spawn_or_modifying_owner(tmp_path, monkeypatch):
    args, root, gpu_lock, streams = queue_fixture(tmp_path, monkeypatch, busy_gpu=True)
    result = queue_gmc.queue(args)
    assert result == 3
    receipt = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert receipt["state"] == "resource_wait_timeout" and receipt["GPU_allocated"] is False
    assert gpu_lock.read_bytes() == b"existing_legacy_owner_metadata"
    assert not (root / "gpu_smoke").exists() and not (root / "training.log").exists()
    assert len(streams) == 2 and all(stream.closed for stream in streams)


def test_queue_existing_smoke_preserves_assets_and_closes_controller_without_gpu_probe(tmp_path, monkeypatch):
    args, root, gpu_lock, streams = queue_fixture(tmp_path, monkeypatch)
    output = root / "gpu_smoke"
    output.mkdir(parents=True)
    checkpoint = output / "checkpoint_000300.pt"
    checkpoint.write_bytes(b"preserved_original_checkpoint")
    status = root / "status.json"
    sentinel = '{"state":"previous_complete","preserved":true}'
    status.write_text(sentinel, encoding="utf-8")
    with pytest.raises(FileExistsError):
        queue_gmc.queue(args)
    assert checkpoint.read_bytes() == b"preserved_original_checkpoint"
    assert status.read_text(encoding="utf-8") == sentinel
    assert gpu_lock.read_bytes() == b"existing_legacy_owner_metadata"
    assert len(streams) == 1 and streams[0].closed


def test_second_controller_does_not_dispatch_or_overwrite_first_controller_status(tmp_path, monkeypatch):
    args, root, gpu_lock, streams = queue_fixture(tmp_path, monkeypatch, busy_controller=True)
    root.mkdir()
    status = root / "status.json"
    sentinel = '{"state":"first_controller_waiting","preserved":true}'
    status.write_text(sentinel, encoding="utf-8")
    with pytest.raises(BlockingIOError):
        queue_gmc.queue(args)
    assert status.read_text(encoding="utf-8") == sentinel
    assert gpu_lock.read_bytes() == b"existing_legacy_owner_metadata"
    assert not (root / "training.log").exists()
    assert len(streams) == 1 and streams[0].closed

import json
import os

from motion_proj.resim import io_memory
from resim.event_first_n1_cutin import _read_json_bounded


def test_fadvise_helpers_use_sequential_then_dontneed(tmp_path, monkeypatch):
    path = tmp_path / "payload.json"
    path.write_text('{"value": 3}\n', encoding="utf-8")
    calls = []

    def fake_fadvise(descriptor, offset, length, advice):
        assert descriptor >= 0
        calls.append((offset, length, advice))

    monkeypatch.setattr(os, "posix_fadvise", fake_fadvise)
    assert _read_json_bounded(path) == {"value": 3}
    assert calls == [
        (0, 0, os.POSIX_FADV_SEQUENTIAL),
        (0, 0, os.POSIX_FADV_DONTNEED),
    ]


def test_drop_path_page_cache_closes_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")
    seen = []

    def fake_fadvise(descriptor, offset, length, advice):
        seen.append((descriptor, offset, length, advice))

    monkeypatch.setattr(os, "posix_fadvise", fake_fadvise)
    assert io_memory.drop_path_page_cache(path)
    assert seen[0][1:] == (0, 0, os.POSIX_FADV_DONTNEED)
    try:
        os.fstat(seen[0][0])
    except OSError:
        pass
    else:
        raise AssertionError("descriptor was not closed")


def test_memory_snapshot_has_stable_schema():
    snapshot = io_memory.memory_snapshot()
    assert set(snapshot) == {
        "process_rss_bytes",
        "cgroup_memory_current_bytes",
    }
    json.dumps(snapshot)

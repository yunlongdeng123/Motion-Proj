"""Bounded-memory helpers for large sequential nuScenes JSON scans."""
from __future__ import annotations

import ctypes
import gc
import os
from pathlib import Path
from typing import IO


def page_cache_control_available() -> bool:
    return all(
        hasattr(os, name)
        for name in (
            "posix_fadvise",
            "POSIX_FADV_SEQUENTIAL",
            "POSIX_FADV_DONTNEED",
        )
    )


def advise_sequential(handle: IO) -> bool:
    if not page_cache_control_available():
        return False
    try:
        os.posix_fadvise(
            handle.fileno(),
            0,
            0,
            os.POSIX_FADV_SEQUENTIAL,
        )
    except (AttributeError, OSError):
        return False
    return True


def drop_handle_page_cache(handle: IO) -> bool:
    if not page_cache_control_available():
        return False
    try:
        os.posix_fadvise(
            handle.fileno(),
            0,
            0,
            os.POSIX_FADV_DONTNEED,
        )
    except (AttributeError, OSError):
        return False
    return True


def drop_path_page_cache(path: str | Path) -> bool:
    if not page_cache_control_available():
        return False
    descriptor = None
    try:
        descriptor = os.open(Path(path), os.O_RDONLY)
        os.posix_fadvise(
            descriptor,
            0,
            0,
            os.POSIX_FADV_DONTNEED,
        )
    except (AttributeError, OSError):
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return True


def trim_process_heap() -> bool:
    gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return bool(trim(0))
    except (AttributeError, OSError):
        return False


def memory_snapshot() -> dict[str, int | None]:
    rss_bytes = None
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss_bytes = int(line.split()[1]) * 1024
                break
    cgroup_bytes = None
    cgroup_path = Path("/sys/fs/cgroup/memory.current")
    if cgroup_path.is_file():
        try:
            cgroup_bytes = int(cgroup_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    return {
        "process_rss_bytes": rss_bytes,
        "cgroup_memory_current_bytes": cgroup_bytes,
    }

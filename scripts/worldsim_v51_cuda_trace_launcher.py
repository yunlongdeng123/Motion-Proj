#!/usr/bin/env python3
"""以 source-neutral Python trace 记录 DEVA spatial-alignment tensor/allocator metadata。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import runpy
import sys
from typing import Any, Mapping


TRACE_SCHEMA = "worldsim_v51_deva_spatial_alignment_cuda_trace_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_metadata(value: Any) -> dict[str, Any] | None:
    if not all(hasattr(value, name) for name in ("shape", "dtype", "device", "stride")):
        return None
    return {
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "stride": [int(item) for item in value.stride()],
        "contiguous": bool(value.is_contiguous()),
        "storage_offset": int(value.storage_offset()),
        "numel": int(value.numel()),
        "element_size": int(value.element_size()),
        "logical_bytes": int(value.numel()) * int(value.element_size()),
        "requires_grad": bool(value.requires_grad),
    }


def allocator_metadata(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    selected_keys = (
        "active_bytes.all.current",
        "active_bytes.all.peak",
        "inactive_split_bytes.all.current",
        "inactive_split_bytes.all.peak",
        "reserved_bytes.all.current",
        "reserved_bytes.all.peak",
        "num_alloc_retries",
        "num_ooms",
    )
    return {
        "cuda_available": True,
        "device": int(torch.cuda.current_device()),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "memory_stats": {key: int(stats.get(key, 0)) for key in selected_keys},
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _frame_snapshot(frame: Any, torch: Any) -> dict[str, Any]:
    local_names = (
        "src_image",
        "tar_image",
        "src_mask",
        "src_key",
        "src_shrinkage",
        "tar_key",
        "tar_selection",
        "value",
        "similarity",
        "affinity",
        "memory_readout",
    )
    tensors = {}
    for name in local_names:
        record = tensor_metadata(frame.f_locals.get(name))
        if record is not None:
            tensors[name] = record
    config = frame.f_locals.get("config", {})
    config_subset = {
        name: config.get(name)
        for name in ("value_dim", "chunk_size", "top_k", "size")
        if isinstance(config, Mapping) and name in config
    }
    return {
        "source_line": int(frame.f_lineno),
        "scalars": {
            name: int(frame.f_locals[name])
            for name in ("src_ti", "tar_ti", "num_objects", "h", "w")
            if name in frame.f_locals
        },
        "config": config_subset,
        "tensors": tensors,
        "allocator": allocator_metadata(torch),
    }


def _write_trace(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def launch(
    target_script: Path,
    target_args: list[str],
    trace_output: Path,
    pre_matmul_empty_cache: bool = False,
) -> None:
    import torch

    target_script = target_script.resolve()
    trace_target = (target_script.parent.parent / "deva/inference/consensus_associated.py").resolve()
    payload: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA,
        "target_script": str(target_script),
        "trace_source": {
            "path": str(trace_target),
            "bytes": trace_target.stat().st_size,
            "sha256": _sha256(trace_target),
        },
        "tensor_content_read": False,
        "operator_monkeypatch": False,
        "pre_matmul_empty_cache": pre_matmul_empty_cache,
        "events": [],
    }
    frame_calls: dict[int, int] = {}

    def local_trace(frame: Any, event: str, arg: Any) -> Any:
        frame_id = id(frame)
        if frame_id not in frame_calls:
            frame_calls[frame_id] = len(frame_calls)
        call_index = frame_calls[frame_id]
        if event == "line" and frame.f_lineno in (58, 59):
            record = {
                "event": "pre_matmul" if frame.f_lineno == 58 else "post_matmul",
                "call_index": call_index,
            }
            if frame.f_lineno == 58 and pre_matmul_empty_cache:
                before = allocator_metadata(torch)
                torch.cuda.empty_cache()
                after = allocator_metadata(torch)
                record["empty_cache"] = {"before": before, "after": after}
            record.update(_frame_snapshot(frame, torch))
            payload["events"].append(record)
        elif event == "exception":
            exception_type, exception, _ = arg
            payload["events"].append(
                {
                    "event": "exception",
                    "call_index": call_index,
                    "exception_type": exception_type.__name__,
                    "exception_message": str(exception),
                    **_frame_snapshot(frame, torch),
                }
            )
        return local_trace

    def global_trace(frame: Any, event: str, arg: Any) -> Any:
        del arg
        if (
            event == "call"
            and frame.f_code.co_name == "spatial_alignment"
            and Path(frame.f_code.co_filename).resolve() == trace_target
        ):
            return local_trace
        return None

    original_argv = sys.argv
    try:
        sys.argv = [str(target_script), *target_args]
        sys.settrace(global_trace)
        runpy.run_path(str(target_script), run_name="__main__")
    except BaseException as error:
        payload["terminal"] = {
            "status": "exception",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        }
        _write_trace(trace_output, payload)
        raise
    else:
        payload["terminal"] = {"status": "success"}
        _write_trace(trace_output, payload)
    finally:
        sys.settrace(None)
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--target-script", type=Path, required=True)
    parser.add_argument("--pre-matmul-empty-cache", action="store_true")
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    target_args = list(args.target_args)
    if target_args and target_args[0] == "--":
        target_args = target_args[1:]
    launch(
        args.target_script,
        target_args,
        args.trace_output.resolve(),
        pre_matmul_empty_cache=args.pre_matmul_empty_cache,
    )


if __name__ == "__main__":
    main()

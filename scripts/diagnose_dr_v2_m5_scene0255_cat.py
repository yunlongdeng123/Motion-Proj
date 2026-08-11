#!/usr/bin/env python
"""在不修改 DriveStudio 源码的前提下定位 scene-0255 的 CUDA cat 失败。"""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import torch


RUN_DIR = Path(
    "/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M5-STRESS-3SCENE-01/"
    "20260802T194900Z__scene0255-cat-diagnostic-s0-r27"
)
PROJECT = Path("/root/autodl-tmp/motion_proj")
UPSTREAM = Path("/root/autodl-tmp/third_party/drivestudio")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def tensor_list_contract(values: object) -> dict:
    tensors = list(values) if isinstance(values, (list, tuple)) else []
    shapes = collections.Counter(str(tuple(value.shape)) for value in tensors)
    return {
        "tensor_count": len(tensors),
        "shape_counts": dict(shapes),
        "total_numel": sum(value.numel() for value in tensors),
        "devices": sorted({str(value.device) for value in tensors}),
        "dtypes": sorted({str(value.dtype) for value in tensors}),
        "all_contiguous": all(value.is_contiguous() for value in tensors),
    }


def main() -> None:
    if RUN_DIR.exists():
        raise FileExistsError(RUN_DIR)
    for name in ("logs", "source_snapshot"):
        (RUN_DIR / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        Path(__file__), RUN_DIR / "source_snapshot" / Path(__file__).name
    )
    command = [
        str(UPSTREAM / "tools/train.py"),
        "--config_file",
        "configs/streetgs.yaml",
        "--output_root",
        str(RUN_DIR / "work_dirs"),
        "--project",
        "m5_stress",
        "--run_name",
        "scene0255_cat_probe_s0",
        "dataset=nuscenes/3cams",
        "data.data_root=/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval",
        "data.scene_idx=204",
        "data.start_timestep=0",
        "data.end_timestep=-1",
        "data.pixel_source.load_smpl=false",
        "data.pixel_source.test_image_stride=10",
        "trainer.optim.num_iters=30000",
        "logging.saveckpt_freq=30000",
        "render.render_full=false",
        "render.render_test=false",
        "render.render_novel=null",
    ]
    write_json(
        RUN_DIR / "manifest.json",
        {
            "schema_version": 1,
            "task_id": "DR-V2-M5-STRESS-3SCENE-01",
            "component": "scene-0255 CUDA cat diagnostic",
            "command": command,
            "project_commit": subprocess.check_output(
                ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "upstream_commit": subprocess.check_output(
                ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
            ).strip(),
            "started_at": now(),
        },
    )
    write_json(
        RUN_DIR / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )
    original_cat = torch.cat
    failed_contract = None

    def traced_cat(values, *args, **kwargs):
        nonlocal failed_contract
        try:
            output = original_cat(values, *args, **kwargs)
            if output.is_cuda:
                torch.cuda.synchronize(output.device)
            return output
        except RuntimeError:
            failed_contract = tensor_list_contract(values)
            print(
                "DR_V2_FAILED_CAT_CONTRACT="
                + json.dumps(failed_contract, sort_keys=True),
                flush=True,
            )
            raise

    torch.cat = traced_cat
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTHONPATH"] = str(UPSTREAM)
    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, str(UPSTREAM))
    os.chdir(UPSTREAM)
    sys.argv = command
    try:
        runpy.run_path(str(UPSTREAM / "tools/train.py"), run_name="__main__")
    except BaseException as error:
        write_json(
            RUN_DIR / "report.json",
            {
                "status": "done",
                "diagnostic_observed": failed_contract is not None,
                "failed_cat_contract": failed_contract,
                "exception": f"{type(error).__name__}: {error}",
            },
        )
        write_json(
            RUN_DIR / "terminal.json",
            {
                "status": "done" if failed_contract is not None else "blocked",
                "updated_at": now(),
                "failure": None
                if failed_contract is not None
                else {"code": "CAT_DIAGNOSTIC_DID_NOT_LOCALIZE"},
            },
        )
        if failed_contract is None:
            raise
        return
    write_json(
        RUN_DIR / "terminal.json",
        {
            "status": "blocked",
            "updated_at": now(),
            "failure": {"code": "EXPECTED_CUDA_CAT_FAILURE_NOT_REPRODUCED"},
        },
    )
    raise RuntimeError("预期 CUDA cat 失败没有复现")


if __name__ == "__main__":
    main()

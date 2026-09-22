#!/usr/bin/env python3
"""在不调用 GPU 和模型权重的前提下检查六个运行环境。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


CHECKS = (
    {
        "model_id": "omnidreams",
        "python": "/root/autodl-tmp/envs/worldsim-v75/bin/python",
        "cwd": "/root/autodl-tmp/external/worldsim_v75/flashdreams",
        "code": "import torch, flashdreams; print(torch.__version__)",
    },
    {
        "model_id": "resim",
        "python": "/root/autodl-tmp/envs/resim/bin/python",
        "cwd": "/root/autodl-tmp/external/worldsim_v75_downstream_bench/ReSim",
        "code": "import torch, torchvision, diffusers, transformers, sat; print(torch.__version__)",
    },
    {
        "model_id": "driveeditor",
        "python": "/root/autodl-tmp/envs/driveeditor/bin/python",
        "cwd": "/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor",
        "code": "import torch, torchvision, xformers, sgm, gradio, nuscenes; print(torch.__version__)",
    },
    {
        "model_id": "gaussiandwm",
        "python": "/root/autodl-tmp/envs/gaussiandwm/bin/python",
        "cwd": "/root/autodl-tmp/external/worldsim_v75_downstream_bench/GaussianDWM",
        "code": "import torch, gaussiandwm_cvpr; print(torch.__version__)",
    },
    {
        "model_id": "street_gaussians",
        "python": "/root/autodl-tmp/envs/drivestudio/bin/python",
        "cwd": "/root/autodl-tmp/external/worldsim_v75_downstream_bench/street_gaussians",
        "code": "import torch, simple_knn, diff_gaussian_rasterization; print(torch.__version__)",
    },
    {
        "model_id": "hugsim",
        "python": "/root/autodl-tmp/envs/hugsim-impact/bin/python",
        "cwd": "/root/autodl-tmp/external/worldsim_simimpact/HUGSIM",
        "code": "import torch, gymnasium, open3d, hugsim_env; print(torch.__version__)",
    },
)


def run_check(check: dict[str, str], timeout: int) -> dict[str, Any]:
    python = Path(check["python"])
    cwd = Path(check["cwd"])
    if not python.is_file() or not cwd.is_dir():
        return {
            "model_id": check["model_id"],
            "passed": False,
            "reason": "missing_python_or_source",
            "python": str(python),
            "cwd": str(cwd),
        }
    try:
        result = subprocess.run(
            [str(python), "-c", check["code"]],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "model_id": check["model_id"],
            "passed": False,
            "reason": "timeout",
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
        }
    return {
        "model_id": check["model_id"],
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    rows = [run_check(check, args.timeout) for check in CHECKS]
    report = {
        "schema_version": "worldsim_v75_environment_smoke_v1",
        "gpu_operations_run": False,
        "checks": rows,
        "summary": {
            "passed": sum(row["passed"] for row in rows),
            "failed": sum(not row["passed"] for row in rows),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

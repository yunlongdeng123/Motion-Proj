#!/usr/bin/env python3
"""正式训练 / smoke 跑通前的资产与环境检查。

读取项目配置（默认 configs/train/motionproj_v1.yaml），检查 Python、conda、
核心依赖、CUDA、nuScenes mini、SVD 权重、输出目录可写性与磁盘余量。
支持 OmegaConf dotlist overrides，例如：
  model.pretrained=/root/autodl-tmp/weights/svd-xt

用法：
  python scripts/check_assets.py
  python scripts/check_assets.py --require-gpu model.pretrained=/path/to/svd-xt
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.config import load_config  # noqa: E402


class CheckReport:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def pass_(self, name: str, detail: str = "") -> None:
        print(f"[PASS] {name}{': ' + detail if detail else ''}")

    def warn(self, name: str, detail: str) -> None:
        self.warnings.append(f"{name}: {detail}")
        print(f"[WARN] {name}: {detail}")

    def fail(self, name: str, detail: str) -> None:
        self.failures.append(f"{name}: {detail}")
        print(f"[FAIL] {name}: {detail}")

    def finish(self) -> int:
        print("")
        print(f"Summary: {len(self.failures)} failure(s), {len(self.warnings)} warning(s)")
        if self.failures:
            return 1
        return 0


def _path_from_cfg(value: Any) -> Path:
    return Path(str(value)).expanduser()


def _find_spec(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _check_python(report: CheckReport) -> None:
    version = sys.version_info
    detail = f"{version.major}.{version.minor}.{version.micro}"
    if version >= (3, 10):
        report.pass_("python", detail)
    else:
        report.fail("python", f"{detail}; Python >= 3.10 is required")


def _check_imports(report: CheckReport) -> None:
    required = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("safetensors", "safetensors"),
        ("omegaconf", "omegaconf"),
        ("nuscenes-devkit", "nuscenes"),
        ("Pillow", "PIL"),
        ("kornia", "kornia"),
        ("tqdm", "tqdm"),
    ]
    missing = [name for name, module in required if not _find_spec(module)]
    if missing:
        report.fail("python packages", "missing: " + ", ".join(missing))
    else:
        report.pass_("python packages", f"{len(required)} required modules importable")


def _check_torch(report: CheckReport, require_gpu: bool) -> None:
    try:
        import torch
    except Exception as exc:
        report.fail("torch", repr(exc))
        return

    report.pass_("torch", getattr(torch, "__version__", "unknown"))
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "cuda device"
        report.pass_("cuda", name)
    elif require_gpu:
        report.fail("cuda", "CUDA is required for the smoke pipeline")
    else:
        report.warn("cuda", "not available; cache/train smoke will need a GPU later")


def _check_conda(report: CheckReport) -> None:
    env = os.environ.get("CONDA_DEFAULT_ENV") or ""
    prefix = os.environ.get("CONDA_PREFIX") or ""
    if env == "motionproj" or prefix.endswith("/motionproj") or prefix.endswith("\\motionproj"):
        report.pass_("conda env", env or prefix)
    else:
        report.warn("conda env", f"expected motionproj, got {env or prefix or 'unset'}")


def _check_data(report: CheckReport, cfg: Any) -> None:
    dataroot = _path_from_cfg(cfg.data.dataroot)
    version = str(cfg.data.version)
    meta_dir = dataroot / version
    if meta_dir.is_dir():
        report.pass_("nuScenes metadata", str(meta_dir))
    else:
        report.fail("nuScenes metadata", f"missing directory: {meta_dir}")

    cam = str(list(cfg.data.cameras)[0]) if cfg.data.get("cameras") else "CAM_FRONT"
    sample_dir = dataroot / "samples" / cam
    if sample_dir.is_dir():
        report.pass_("nuScenes samples", str(sample_dir))
    else:
        report.fail("nuScenes samples", f"missing directory: {sample_dir}")


def _hf_cache_candidates(repo_id: str) -> list[Path]:
    safe = "models--" + repo_id.replace("/", "--")
    roots = [
        Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
        Path("/root/autodl-tmp/hf_cache/hub"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    return [root / safe for root in roots if root is not None]


def _check_model(report: CheckReport, cfg: Any) -> None:
    pretrained = str(cfg.model.pretrained)
    as_path = Path(pretrained).expanduser()
    required_subdirs = ["vae", "unet", "image_encoder", "scheduler", "feature_extractor"]

    if as_path.exists():
        missing = [name for name in required_subdirs if not (as_path / name).exists()]
        if missing:
            report.fail("SVD weights", f"{as_path} exists but misses: {', '.join(missing)}")
        else:
            report.pass_("SVD weights", str(as_path))
        return

    if "/" in pretrained and not pretrained.startswith("."):
        candidates = [p for p in _hf_cache_candidates(pretrained) if p.exists()]
        if candidates:
            report.pass_("SVD HF cache", str(candidates[0]))
        else:
            report.warn(
                "SVD weights",
                f"{pretrained} is a remote repo id and no local HF cache candidate was found",
            )
        return

    report.fail("SVD weights", f"missing path: {as_path}")


def _check_writable(report: CheckReport, cfg: Any) -> None:
    paths = [
        ("cache parent", _path_from_cfg(cfg.paths.cache_dir).parent),
        ("ckpt parent", _path_from_cfg(cfg.paths.ckpt_dir).parent),
        ("log parent", _path_from_cfg(cfg.paths.log_dir).parent),
    ]
    for name, path in paths:
        existing = path
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        if existing.exists() and os.access(existing, os.W_OK):
            report.pass_(name, str(path))
        else:
            report.fail(name, f"not writable: {path}")


def _check_disk(report: CheckReport, cfg: Any, min_free_gb: float) -> None:
    target = _path_from_cfg(cfg.paths.cache_dir)
    existing = target
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if not existing.exists():
        report.fail("disk", f"no existing parent for {target}")
        return

    usage = shutil.disk_usage(existing)
    free_gb = usage.free / (1024**3)
    detail = f"{free_gb:.1f} GiB free under {existing}"
    if free_gb < min_free_gb:
        report.fail("disk", f"{detail}; need at least {min_free_gb:.1f} GiB")
    else:
        report.pass_("disk", detail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Motion-Proj assets before smoke runs.")
    parser.add_argument("--config", default="configs/train/motionproj_v1.yaml")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = CheckReport()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        report.fail("config", f"missing file: {config_path}")
        return report.finish()

    cfg = load_config(str(config_path), args.overrides)
    report.pass_("config", str(config_path))

    _check_python(report)
    _check_conda(report)
    _check_imports(report)
    _check_torch(report, require_gpu=bool(args.require_gpu))
    _check_data(report, cfg)
    _check_model(report, cfg)
    _check_writable(report, cfg)
    _check_disk(report, cfg, min_free_gb=float(args.min_free_gb))
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())

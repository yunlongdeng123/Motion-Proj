"""评估命令行工具。

两种模式：
  - ``--mode cache``（默认，无需权重）：从投影缓存的元数据中聚合投影质量
    相关指标（第 3 周的评判标准）。
  - ``--mode auditor``：在数据集上运行 auditor，按静态漂移对片段排序，
    并报告腐蚀敏感度（第 2 周的评判标准）。
"""
from __future__ import annotations

import argparse
import glob
import os

from ..config import get_paths, load_config
from ..utils.io import load_json
from ..utils.logging import get_logger
from .diagnostics import corruption_sensitivity, projection_quality, rank_clips

log = get_logger(__name__)


def eval_cache(cfg) -> dict:
    paths = get_paths(cfg)
    metas = [
        load_json(p)
        for p in glob.glob(os.path.join(paths.cache_dir, "*", "metadata.json"))
    ]
    if not metas:
        raise FileNotFoundError(f"no cache metadata under {paths.cache_dir}")
    report = projection_quality(metas)
    log.info(
        "Projection quality: %d clips | improved E_obj in %.1f%% | mean reduction %.4f",
        report["clips"], 100 * report["frac_improved"], report["mean_obj_reduction"],
    )
    return report


def eval_auditor(cfg, n: int = 20) -> dict:
    from ..auditor import MotionAuditor
    from ..data import NuScenesFutureVideoDataset

    ds = NuScenesFutureVideoDataset(cfg.data)
    auditor = MotionAuditor(device=cfg.device)
    ranked = rank_clips(ds, auditor, n=n)
    log.info("Top drift clips: %s", ranked[:5])
    sens = corruption_sensitivity(auditor.audit(ds[0]))
    log.info("Corruption sensitivity (clip 0): %s", sens)
    return {"ranked": ranked, "sensitivity": sens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["cache", "auditor"], default="cache")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.mode == "cache":
        eval_cache(cfg)
    else:
        eval_auditor(cfg, n=args.n)


if __name__ == "__main__":
    main()

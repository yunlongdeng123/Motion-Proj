"""评估命令行工具。

三种模式：
  - ``--mode cache``（默认，无需权重）：从投影缓存的元数据中聚合投影质量
    相关指标（第 3 周的评判标准）。
  - ``--mode auditor``：在数据集上运行 auditor，按静态漂移对片段排序，
    并报告腐蚀敏感度（第 2 周的评判标准）。
  - ``--mode generate``：固定 seed，对 base SVD 与各 LoRA checkpoint 生成未来
    片段并评估（静态漂移 + LPIPS/PSNR/SSIM）。更细的参数见
    ``python -m motion_proj.eval.generate_eval``。
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


def eval_generate(cfg, args) -> dict:
    import os

    from .generate_eval import resolve_adapters, run_generate_eval

    paths = get_paths(cfg)
    adapters = resolve_adapters(args.adapters.split(","), paths.ckpt_dir)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 1234))
    out_dir = args.out_dir or os.path.join(str(cfg.work_dir), "eval", "generate")
    explicit = [int(x) for x in args.clip_indices.split(",")] if args.clip_indices else None
    return run_generate_eval(
        cfg,
        adapters=adapters,
        seed=seed,
        out_dir=out_dir,
        num_frames=int(cfg.data.num_frames),
        num_inference_steps=int(args.num_inference_steps),
        num_clips=int(args.num_clips),
        clip_indices=explicit,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["cache", "auditor", "generate"], default="cache")
    ap.add_argument("--n", type=int, default=20)
    # generate 模式参数
    ap.add_argument("--adapters", default="base,adapter_final")
    ap.add_argument("--num-clips", type=int, default=4)
    ap.add_argument("--clip-indices", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-inference-steps", type=int, default=25)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.mode == "cache":
        eval_cache(cfg)
    elif args.mode == "generate":
        eval_generate(cfg, args)
    else:
        eval_auditor(cfg, n=args.n)


if __name__ == "__main__":
    main()

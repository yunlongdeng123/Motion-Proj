"""单个 Optuna trial：按参数训练，再写入含评估字段的 summary.json。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..config import load_config
from ..runtime.atomic import atomic_write_json
from ..train.trainer import Trainer
from ..utils.logging import get_logger
from .trial_eval import run_trial_eval

log = get_logger(__name__)


def trial_overrides(params: dict, steps: int, run_dir: str, parent_run_id: str | None) -> list[str]:
    """把 Optuna 搜索参数映射为配置 override。"""
    tube_upper = float(params["tube_upper"])
    overrides = [
        f"work_dir={run_dir}",
        f"paths.ckpt_dir={run_dir}/ckpts",
        f"paths.log_dir={run_dir}/logs",
        f"run_id={Path(run_dir).name}",
        f"train.max_steps={int(steps)}",
        f"train.lr={float(params['lr'])}",
        f"train.lambda_proj={float(params['lambda_proj'])}",
        f"train.beta_anchor={float(params['beta_anchor'])}",
        f"train.tube.bound_B={float(params['bound_B'])}",
        f"train.tube.sigma_quantile_range=[0.0,{tube_upper}]",
        f"model.lora.rank={int(params.get('lora_rank', 16))}",
        "train.resume=none",
        "train.logger=none",
    ]
    if parent_run_id and parent_run_id != "none":
        overrides.append(f"parent_run_id={parent_run_id}")
    return overrides


def run_trial(
    *,
    config: str,
    params: dict,
    steps: int,
    run_dir: str,
    parent_run_id: str | None,
    base_metrics_path: str,
    num_clips: int,
    clip_indices: list[int] | None,
    num_inference_steps: int,
    extra_overrides: list[str] | None = None,
) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.isfile(summary_path):
        return json.loads(Path(summary_path).read_text(encoding="utf-8"))

    overrides = trial_overrides(params, steps, run_dir, parent_run_id)
    if extra_overrides:
        overrides.extend(extra_overrides)
    cfg = load_config(config, overrides)
    Trainer(cfg).train()
    train_summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))

    adapter = None
    checkpoint = train_summary.get("checkpoint")
    if checkpoint and checkpoint != "pretrained":
        candidate = os.path.join(checkpoint, "adapter.safetensors")
        if os.path.isfile(candidate):
            adapter = candidate
        else:
            # 兼容目录内其它 adapter 命名
            for name in ("adapter_final.safetensors", "pytorch_lora_weights.safetensors"):
                path = os.path.join(checkpoint, name)
                if os.path.isfile(path):
                    adapter = path
                    break
    if adapter is None and not train_summary.get("frozen"):
        raise FileNotFoundError(f"trial checkpoint 缺少 adapter: {checkpoint}")

    base_metrics = json.loads(Path(base_metrics_path).read_text(encoding="utf-8"))
    # 调参筛选在官方 val 固定子集上评估，避免与 mini train cache 完全重叠。
    eval_cfg = load_config(config, overrides + [
        "data.split=val",
        "data.expected_scene_count=150",
        "data.expected_clip_count=732",
    ])
    summary = run_trial_eval(
        eval_cfg,
        adapter_path=adapter,
        base_metrics=base_metrics,
        out_summary=summary_path,
        train_summary=train_summary,
        num_clips=num_clips,
        clip_indices=clip_indices,
        num_inference_steps=num_inference_steps,
    )
    atomic_write_json(os.path.join(run_dir, "trial_params.json"), {
        "params": params,
        "steps": steps,
        "parent_run_id": parent_run_id,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-metrics", required=True)
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--clip-indices", default=None)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    raw = os.environ.get("MOTIONPROJ_TRIAL_JSON")
    if not raw:
        raise SystemExit("缺少 MOTIONPROJ_TRIAL_JSON")
    trial = json.loads(raw)
    explicit = (
        [int(x) for x in args.clip_indices.split(",")] if args.clip_indices else None
    )
    summary = run_trial(
        config=args.config,
        params={
            "lr": float(trial["lr"]),
            "lambda_proj": float(trial["lambda_proj"]),
            "beta_anchor": float(trial["beta_anchor"]),
            "bound_B": float(trial["bound_B"]),
            "tube_upper": float(trial["tube_upper"]),
            "lora_rank": int(trial.get("lora_rank", 16)),
        },
        steps=int(trial["target_steps"]),
        run_dir=str(trial["run_dir"]),
        parent_run_id=trial.get("parent_run_id"),
        base_metrics_path=args.base_metrics,
        num_clips=int(args.num_clips),
        clip_indices=explicit,
        num_inference_steps=int(args.num_inference_steps),
        extra_overrides=list(args.overrides),
    )
    log.info(
        "trial done score_fields drift=%.4f track=%.4f lpips=%.4f eligible=%.4f",
        summary["static_drift"], summary["track_acceleration"],
        summary["lpips"], summary["projection_eligible_fraction"],
    )


if __name__ == "__main__":
    main()

"""P3 五个非前视相机的独立、可恢复零样本评估。"""
from __future__ import annotations

import argparse
import os
import statistics

from omegaconf import OmegaConf

from ..config import get_paths, load_config, to_container
from ..utils.io import save_json
from .drivinggen import PROTOCOL
from .generate_eval import _METRIC_KEYS, resolve_adapters, run_generate_eval

P3_CAMERAS = ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")


def aggregate_camera_summaries(summaries: dict[str, dict]) -> dict:
    adapter_names = sorted({name for summary in summaries.values()
                            for name in summary.get("adapters", {})})
    macro = {}
    for adapter in adapter_names:
        values = {}
        for metric in _METRIC_KEYS:
            key = f"{metric}_mean"
            rows = [summary["adapters"][adapter]["aggregate"].get(key)
                    for summary in summaries.values()
                    if adapter in summary.get("adapters", {})]
            finite = [float(value) for value in rows if value is not None]
            values[key] = statistics.fmean(finite) if finite else None
            values[f"{metric}_camera_n"] = len(finite)
        macro[adapter] = values
    return {"cameras": list(summaries), "macro": macro,
            "metric_protocol": PROTOCOL, "multi_camera_sync": False}


def run_multicamera_eval(cfg, adapters, seeds: list[int], out_dir: str,
                         num_inference_steps: int, num_clips: int) -> dict:
    summaries = {}
    for camera in P3_CAMERAS:
        seed_summaries = []
        for seed in seeds:
            mutable = OmegaConf.create(to_container(cfg))
            mutable.data.cameras = [camera]
            mutable.data.split = "val"
            mutable.data.expected_scene_count = 150
            mutable.data.expected_clip_count = 732
            OmegaConf.set_readonly(mutable, True)
            seed_out = os.path.join(out_dir, camera, f"seed-{seed}")
            seed_summaries.append(run_generate_eval(
                mutable, adapters, seed, seed_out, int(mutable.data.num_frames),
                num_inference_steps, num_clips=num_clips,
            ))
        # 当前 P2/P3 固定单 seed；保留列表结构以便后续三 seed 论文实验。
        summaries[camera] = seed_summaries[0] if len(seed_summaries) == 1 else {
            "seeds": seed_summaries
        }
    result = {"per_camera": summaries, **aggregate_camera_summaries(summaries)}
    save_json(result, os.path.join(out_dir, "summary.json"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--seeds", default="1234")
    parser.add_argument("--num-clips", type=int, default=732)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    paths = get_paths(cfg)
    adapters = resolve_adapters(args.adapters.split(","), paths.ckpt_dir)
    seeds = [int(value) for value in args.seeds.split(",")]
    run_multicamera_eval(cfg, adapters, seeds, args.out_dir,
                         args.num_inference_steps, args.num_clips)


if __name__ == "__main__":
    main()

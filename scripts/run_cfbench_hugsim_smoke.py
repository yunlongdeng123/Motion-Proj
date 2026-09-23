"""官方导出场景的 actor removal paired render；固定开环动作，不冒充 AD 闭环。"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", choices=["factual", "counterfactual"], required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path("/root/autodl-tmp/external/worldsim_simimpact/HUGSIM")
    sys.path[:0] = [str(source), str(source / "sim")]
    os.chdir(source)
    from hugsim_env.envs.hug_sim import HUGSimEnv

    index = json.loads(Path("/root/autodl-tmp/data/worldsim_v75_downstream_bench/adapter_inputs/hugsim/index.json").read_text())
    scenario = next(row for row in index["scenarios"] if row["name"] == "scene-0383-medium-00.yaml")
    result = {"task_id": "WS-V75-DOWNSTREAM-GPU-SMOKE-01", "run_id": "20260923-r1",
              "model_id": "hugsim", "status": "started", "benchmark_aligned": False,
              "input_role": "official_scene_engineering_smoke", "scene": "scene-0383",
              "intervention": "remove the only scenario actor", "seed": 42,
              "ad_client_used": False, "feedback_policy": False, "actions": {"acc": 0.0, "steer_rate": 0.0},
              "human_verdict": None, "failure_ledger_refs": [], "branches": {}}
    started = time.monotonic()
    try:
        # Camera 的默认 dynamics 字典会被 renderer 原位写入；分支用独立进程
        # 避免同进程 sequential environments 串扰，不修改官方实现。
        result["branch_process_isolation"] = True
        for branch in [args.branch]:
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            branch_dir = output / branch
            branch_dir.mkdir()
            cfg = OmegaConf.create({
                "scenario": OmegaConf.load(scenario["path"]),
                "base": OmegaConf.load(index["base_config"]),
                "camera": OmegaConf.load(index["camera_config"]),
                "kinematic": OmegaConf.load(index["kinematic_config"]),
            })
            model_path = Path(index["scene_root"]) / "scene-0383"
            cfg.update(OmegaConf.load(model_path / "cfg.yaml"))
            cfg.model_path = str(model_path)
            if branch == "counterfactual":
                cfg.scenario.plan_list = []
            OmegaConf.save(cfg, branch_dir / "resolved.yaml")
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                env = HUGSimEnv(cfg, str(branch_dir))
                obs, info = env.reset(seed=42)
                infos, frames = [], []
                for frame in range(9):
                    if frame:
                        obs, reward, terminated, truncated, info = env.step({"acc": 0.0, "steer_rate": 0.0})
                    else:
                        terminated, truncated = False, False
                    for camera, rgb in obs["rgb"].items():
                        Image.fromarray(rgb).save(branch_dir / f"{frame:03d}-{camera}.png")
                    frames.append(obs["rgb"]["CAM_FRONT"])
                    infos.append({"frame": frame, "timestamp": info["timestamp"], "ego_pos": info["ego_pos"],
                                  "ego_velo": float(info["ego_velo"]), "obj_boxes": info["obj_boxes"],
                                  "collision": bool(info.get("collision", False)), "terminated": bool(terminated),
                                  "depth_finite": all(bool(np.isfinite(d).all()) for d in obs["depth"].values())})
                    print(json.dumps({"branch": branch, "frame": frame, "actor_count": len(info["obj_boxes"])}), flush=True)
                    if terminated or truncated:
                        break
                imageio.mimwrite(branch_dir / "front.mp4", frames, fps=4, macro_block_size=None)
                (branch_dir / "states.json").write_text(json.dumps(infos, indent=2) + "\n")
                result["branches"][branch] = {"frames": len(frames), "views": list(obs["rgb"]),
                    "render_calls": len(frames)*len(obs["rgb"]), "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                    "actor_counts": [len(r["obj_boxes"]) for r in infos]}
                del env, obs, frames
            gc.collect()
            torch.cuda.empty_cache()
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="oom_stopped" if isinstance(exc, torch.OutOfMemoryError) else "failed_stopped",
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result["wall_s"] = time.monotonic() - started
        result["failure_ledger_delta"] = "none"
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

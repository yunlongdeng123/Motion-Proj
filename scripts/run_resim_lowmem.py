"""ReSim 单卡分阶段权重搬运；保留49帧、原精度、采样步数和模型计算。"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resim-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--vae-chunk-frames", type=int, choices=[49, 17], default=49)
    args = parser.parse_args()
    repo, cfg_path, result_path = args.resim_root.resolve(), args.config.resolve(), args.result.resolve()
    if result_path.exists():
        raise FileExistsError(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(repo / "SwissArmyTransformer"), str(repo / "sat")]
    os.chdir(repo / "sat")
    import torch
    from diffusion_video import SATVideoDiffusionEngine

    result = {"status": "started", "implementation": "phasewise_cpu_offload_v1",
              "config": str(cfg_path), "precision_change": False, "resolution_change": False,
              "sampling_step_change": False, "temporal_chunk_change": args.vae_chunk_frames != 49,
              "vae_chunk_frames": args.vae_chunk_frames,
              "model_source_modified": False, "human_verdict": None, "stages": []}

    def stage(name):
        gc.collect()
        torch.cuda.empty_cache()
        row = {"stage": name, "allocated_gib": torch.cuda.memory_allocated()/2**30,
               "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30}
        result["stages"].append(row)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    original_init = SATVideoDiffusionEngine.__init__
    original_encode = SATVideoDiffusionEngine.encode_first_stage
    original_sample = SATVideoDiffusionEngine.sample

    def patched_init(self, *init_args, **init_kwargs):
        original_init(self, *init_args, **init_kwargs)
        # 17帧分块为上游显式支持的路径；GroupNorm跨时域，故不宣称数值等价。
        self.en_and_decode_n_frames_a_time = args.vae_chunk_frames
        original_condition = self.conditioner.get_unconditional_conditioning
        def condition(*condition_args, **condition_kwargs):
            self.model.to("cpu")
            self.first_stage_model.to("cpu")
            self.conditioner.to(self.device)
            stage("text_and_trajectory_conditioning")
            values = original_condition(*condition_args, **condition_kwargs)
            self.conditioner.to("cpu")
            stage("conditioner_offloaded")
            return values
        self.conditioner.get_unconditional_conditioning = condition

    def encode(self, *encode_args, **encode_kwargs):
        self.model.to("cpu")
        self.conditioner.to("cpu")
        self.first_stage_model.to(self.device)
        stage("vae_encode_only_on_gpu")
        values = original_encode(self, *encode_args, **encode_kwargs)
        self.first_stage_model.to("cpu")
        stage("vae_encoded_and_offloaded")
        return values

    def sample(self, *sample_args, **sample_kwargs):
        self.first_stage_model.to("cpu")
        self.conditioner.to("cpu")
        self.model.to(self.device)
        stage("diffusion_only_on_gpu")
        return original_sample(self, *sample_args, **sample_kwargs)

    SATVideoDiffusionEngine.__init__ = patched_init
    SATVideoDiffusionEngine.encode_first_stage = encode
    SATVideoDiffusionEngine.sample = sample
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        sys.argv = [str(repo / "sat/sample_video.py"), f"--base={cfg_path}"]
        runpy.run_path(sys.argv[0], run_name="__main__")
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="oom_stopped" if isinstance(exc, torch.OutOfMemoryError) else "failed_stopped",
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result.update(wall_s=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

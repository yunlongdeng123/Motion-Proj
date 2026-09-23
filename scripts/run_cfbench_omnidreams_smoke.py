"""官方场景的单变量 ego 减速 paired smoke；不冒充冻结 pilot case。"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent / "worldsim_v75"))
from common import CAMERA, RUN, SCENE, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "task_id": "WS-V75-DOWNSTREAM-GPU-SMOKE-01", "run_id": "20260923-r1",
        "model_id": "omnidreams", "input_role": "official_scene_engineering_smoke",
        "bench_case_id": None, "seed": 42, "blocks_per_branch": 8,
        "intervention": {"ego_speed_scale": [1.0, 0.5]},
        "held_fixed": ["initial_image", "prompt", "actor_states", "map", "timestamps", "seed"],
        "human_verdict": None, "failure_ledger_refs": [], "branches": {},
        "status": "started", "source_scene": str(SCENE),
    }
    started = time.monotonic()
    try:
        from interactive_drive.config import RasterConfig
        from interactive_drive.scene_loader import load_scene_bundle
        from interactive_drive.rasterizer import LudusConditionRasterizer

        scene = load_scene_bundle(SCENE, CAMERA, "default", None, RasterConfig())
        data = np.load(RUN / "trajectory.npz")
        count = 5 + 7 * 8
        timestamps = data["timestamps_us"][:count]
        original = data["rig_poses_world"][:count]
        relative_t = (timestamps - timestamps[0]) / 1e6
        slowed = original.copy()
        slowed[:, :3, :3] = Slerp(relative_t, Rotation.from_matrix(original[:, :3, :3]))(relative_t * 0.5).as_matrix()
        for axis in range(3):
            slowed[:, axis, 3] = np.interp(relative_t * 0.5, relative_t, original[:, axis, 3])
        np.savez(out / "paired_trajectory.npz", timestamps_us=timestamps, factual=original, counterfactual=slowed)
        Image.fromarray(scene.initial_rgb).save(out / "initial_rgb.png")
        renderer = LudusConditionRasterizer(RasterConfig(), max_chunk_frames=8)
        conditions = {}
        try:
            renderer.load_scene(scene)
            for branch, poses in [("factual", original), ("counterfactual", slowed)]:
                frames = []
                for start in range(0, count, 8):
                    rendered = renderer.render_chunk(poses[start:start+8], timestamps[start:start+8])
                    frames.extend(np.asarray(frame.rgb_host_uint8).copy() for frame in rendered.frames)
                conditions[branch] = np.stack(frames)
                np.save(out / f"{branch}-conditions.npy", conditions[branch])
                Image.fromarray(conditions[branch][-1]).save(out / f"{branch}-condition-last.png")
        finally:
            renderer.cleanup()
        assert np.array_equal(conditions["factual"][0], conditions["counterfactual"][0])
        assert np.any(conditions["factual"][-1] != conditions["counterfactual"][-1])
        result["condition_changed_pixels_last"] = int(np.any(conditions["factual"][-1] != conditions["counterfactual"][-1], axis=-1).sum())
        embeddings = torch.load(RUN / "embeddings.pt", map_location="cpu", weights_only=True)
        for branch in ["factual", "counterfactual"]:
            cfg = config()
            cfg.text_encoder = None
            cfg.image_encoder = None
            cfg.diffusion_model.seed = 42
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.cuda.reset_peak_memory_stats()
            pipeline = cfg.setup().to("cuda").eval()
            cache = pipeline.initialize_cache_from_embeddings(**embeddings, view_names=[CAMERA])
            cursor = 0
            timings = []
            with av.open(str(out / f"{branch}.mp4"), "w") as writer:
                stream = writer.add_stream("libx264", rate=30)
                stream.width, stream.height, stream.pix_fmt = 1280, 704, "yuv420p"
                stream.options = {"crf": "18"}
                for block in range(8):
                    n = 5 if block == 0 else 8
                    batch = torch.from_numpy(conditions[branch][cursor:cursor+n].copy())
                    batch = batch.permute(0, 3, 1, 2)[None, None].to("cuda", dtype=torch.bfloat16) / 127.5 - 1
                    began = time.monotonic()
                    with torch.inference_mode():
                        generated = pipeline.generate(autoregressive_index=block, cache=cache, input=batch)
                        pipeline.finalize(autoregressive_index=block, cache=cache)
                    assert torch.isfinite(generated).all()
                    frames = ((generated[0, 0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                    for frame in frames:
                        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                            writer.mux(packet)
                    cursor += n
                    row = {"branch": branch, "block": block, "frames": cursor, "wall_s": time.monotonic() - began}
                    timings.append(row)
                    print(json.dumps(row), flush=True)
                for packet in stream.encode():
                    writer.mux(packet)
            result["branches"][branch] = {"frames": cursor, "generation_calls": 8, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "timings": timings}
            del pipeline, cache, generated, batch
            gc.collect()
            torch.cuda.empty_cache()
            (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="oom_stopped" if isinstance(exc, torch.OutOfMemoryError) else "failed_stopped", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result["wall_s"] = time.monotonic() - started
        result["failure_ledger_delta"] = "none"
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

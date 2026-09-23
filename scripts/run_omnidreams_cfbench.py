"""冻结24-case的官方Ludus渲染、条件编码和成对生成；三个阶段分进程。"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import av
import numpy as np
from PIL import Image, ImageDraw
from add_cfbench_original_videos import make_original
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "worldsim_v75"))
from common import CAMERA, config


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def make_nuscenes_camera(K):
    """完整可见域上的五阶minimax针孔近似；独立密网格残差须小于0.5像素。"""
    from scipy.optimize import linprog
    from ludus_renderer import FThetaCamera
    from render_argoverse import tensor
    fx, fy, cx, cy = K
    limit = np.arctan(np.hypot(max(cx, 1280-cx)/fx, max(cy, 704-cy)/fy))
    u = np.linspace(0, 1, 4097)
    A = np.stack([u**i for i in range(1, 6)], axis=1)
    y = fx*np.tan(u*limit)
    fit = linprog([0, 0, 0, 0, 0, 1], A_ub=np.concatenate([np.c_[A, -np.ones(len(u))], np.c_[-A, -np.ones(len(u))]]),
                  b_ub=np.r_[y, -y], bounds=[(None, None)]*5+[(0, None)], method="highs")
    assert fit.success, fit.message
    poly = np.r_[0, fit.x[:5]/limit**np.arange(1, 6)]
    check = np.linspace(0, limit, 32769)
    error = np.abs(np.polynomial.polynomial.polyval(check, poly)-fx*np.tan(check)).max()
    assert error < .5, error
    assert np.all(np.polynomial.polynomial.polyval(check, np.arange(1, 6)*poly[1:]) > 0)
    camera = FThetaCamera(tensor([cx, cy]), tensor([1280, 704]), tensor(poly), float(limit),
                          linear_distortion=tensor([[1, 0], [0, fy/fx]]), depth_max=200)
    return camera, {"method": "minimax_full_visible_pinhole_domain", "max_radial_residual_px": float(error),
                    "max_angle_rad": float(limit), "tolerance_px": .5, "fifth_order_coefficients": poly.tolist()}


def render_case(row, out):
    from render_argoverse import (make_camera, independent_check, context, render,
        tensor, pool_for_tracks, TimestampedScene, _polylines_to_pool, _polygons_to_pool,
        PRIM_ROAD_BOUNDARY, PRIM_CROSSWALK, PRIM_LANE_LINE_WHITE_SOLID,
        PRIM_LANE_LINE_WHITE_DASHED, PRIM_LANE_LINE_YELLOW_SOLID, PRIM_LANE_LINE_YELLOW_DASHED)
    src = Path(row["input_dir"])
    traj = np.load(src / "trajectory.npz")
    camera, fit = make_nuscenes_camera(traj["K"])
    checks = independent_check(camera, traj["K"])
    arrays = {}
    for branch in ["factual", "counterfactual"]:
        data = json.loads((src / f"{branch}-scene.json").read_text())
        buckets = {}
        for line in data["lines"]:
            if line["kind"] == "road_boundary":
                kind = PRIM_ROAD_BOUNDARY
            else:
                yellow, dashed = "YELLOW" in line["mark"], "DASH" in line["mark"]
                kind = ([PRIM_LANE_LINE_YELLOW_SOLID, PRIM_LANE_LINE_YELLOW_DASHED] if yellow else
                        [PRIM_LANE_LINE_WHITE_SOLID, PRIM_LANE_LINE_WHITE_DASHED])[int(dashed)]
            buckets.setdefault(kind, []).append(tensor(line["xyz"]))
        lines = [_polylines_to_pool(values, kind, torch.device("cuda")) for kind, values in buckets.items()]
        polygon = _polygons_to_pool([tensor(x) for x in data["crossings"]], PRIM_CROSSWALK, torch.device("cuda")) if data["crossings"] else None
        scene = TimestampedScene(lines, [polygon] if polygon else [], [pool_for_tracks(data["tracks"], traj["timestamps_us"])])
        ctx = context(camera)
        sid = ctx.upload_scene(scene)
        path = out / f"{branch}-conditions.npy"
        assert not path.exists(), path
        arrays[branch] = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(77, 704, 1280, 3))
        for start in range(0, 77, 8):
            arrays[branch][start:start+8] = render(ctx, sid, traj["timestamps_us"][start:start+8], traj[branch][start:start+8])
        arrays[branch].flush()
        del ctx, scene, lines, polygon
        gc.collect()
        torch.cuda.empty_cache()
    changed = np.any(arrays["factual"] != arrays["counterfactual"], axis=-1).sum((1, 2))
    assert np.all(changed[:row["event_output_frame"]] == 0), "干预前条件不一致"
    # 不按结果改case；条件无可见变化照样登记，但不视为有效编辑证据。
    result = {"status": "complete", "pinhole_fit": fit, "independent_projection_checks": checks,
              "changed_pixels_by_frame": changed.tolist(), "prefix_identical": True,
              "edit_visible_in_condition": bool(np.any(changed[15:70] > 0)),
              "first_condition_nonzero_pixels": int(np.any(arrays["factual"][0], axis=-1).sum()),
              "human_verdict": None}
    sheet = Image.new("RGB", (1280, 4*378), "#152033")
    draw = ImageDraw.Draw(sheet)
    for n, f in enumerate([0, 15, 42, 69]):
        y = n*378
        draw.text((8, y+5), f"{row['case_id']} | f={f} | factual / counterfactual | changed pixels={changed[f]}", fill="white")
        for k, branch in enumerate(["factual", "counterfactual"]):
            im = Image.fromarray(arrays[branch][f])
            if f == 0:
                im = Image.blend(Image.open(src / "initial_rgb.png").convert("RGB"), im, .5)
            sheet.paste(im.resize((640, 352)), (k*640, y+26))
    sheet.save(out / "condition-review.jpg", quality=92)
    save(out / "render-result.json", result)
    return result


def encode(rows, output):
    prompts = [(Path(row["input_dir"]) / "prompt.txt").read_text().strip() for row in rows]
    assert len(set(prompts)) == 1
    cfg = config()
    text = cfg.text_encoder.setup().cuda().eval()
    with torch.inference_mode():
        embedding = text([prompts[0]]).unsqueeze(0).cpu()
    assert torch.isfinite(embedding).all()
    torch.save(embedding, output / "text-embeddings.pt")
    print(json.dumps({"phase": "text_encoded", "shape": list(embedding.shape)}), flush=True)
    del text, embedding
    gc.collect()
    torch.cuda.empty_cache()
    encoder = cfg.image_encoder.setup().cuda().eval()
    for row in rows:
        out = output / row["case_id"]
        image = np.array(Image.open(Path(row["input_dir"]) / "initial_rgb.png").convert("RGB"))
        pixels = torch.from_numpy(image).permute(2, 0, 1)[None, None, None].cuda().to(torch.bfloat16)/127.5-1
        with torch.inference_mode():
            embedding = encoder(pixels).cpu()
        assert torch.isfinite(embedding).all()
        torch.save(embedding, out / "image-embeddings.pt")
        print(json.dumps({"phase": "image_encoded", "case_id": row["case_id"], "shape": list(embedding.shape)}), flush=True)


def generate(rows, output):
    cfg = config()
    cfg.text_encoder = cfg.image_encoder = None
    cfg.diffusion_model.seed = 42
    pipeline = cfg.setup().cuda().eval()
    text = torch.load(output / "text-embeddings.pt", map_location="cpu", weights_only=True)
    for row in rows:
        out = output / row["case_id"]
        if (out / "result.json").exists():
            previous = json.loads((out / "result.json").read_text())
            if previous["status"] == "generation_complete":
                assert all((out / f"{b}.mp4").exists() for b in ["factual", "counterfactual"])
                print(json.dumps({"verified_skip": row["case_id"]}), flush=True)
                continue
            raise RuntimeError(f"保留失败结果，需显式新run再重试: {out}")
        began = time.monotonic()
        result = {"task_id": "WS-V75-DOWNSTREAM-FULL-01", "run_id": "20260923-r1", "model_id": "omnidreams",
                  "case_id": row["case_id"], "status": "started", "seed": 42,
                  "input_role": row["conditioning_variant"], "input_dir": row["input_dir"],
                  "frames_per_branch": 77, "scored_frames_per_branch": 70, "tail_padding_not_scored": 7,
                  "blocks_per_branch": 10, "fps": 30, "human_verdict": None, "ai_scores": None,
                  "failure_ledger_refs": [], "failure_ledger_delta": "none", "branches": {},
                  "paired_randomness": "new rollout cache and RNG seed42 reset for both branches; loaded weights reused"}
        save(out / "result.json", result)
        image = torch.load(out / "image-embeddings.pt", map_location="cpu", weights_only=True)
        prefixes = {}
        try:
            for branch in ["factual", "counterfactual"]:
                assert not (out / f"{branch}.mp4").exists()
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                # 官方惰性rng读取config.seed；每个rollout必须独立重置，避免分支随机数偏移。
                pipeline.diffusion_model._rng = None
                torch.cuda.reset_peak_memory_stats()
                cache = pipeline.initialize_cache_from_embeddings(text_embeddings=text, image_embeddings=image, view_names=[CAMERA])
                conditions = np.load(out / f"{branch}-conditions.npy", mmap_mode="r")
                cursor, timings, early = 0, [], []
                with av.open(str(out / f"{branch}.mp4"), "w") as writer:
                    stream = writer.add_stream("libx264", rate=30)
                    stream.width, stream.height, stream.pix_fmt = 1280, 704, "yuv420p"
                    stream.options = {"crf": "18"}
                    for block in range(10):
                        n = 5 if block == 0 else 8
                        batch = torch.from_numpy(conditions[cursor:cursor+n].copy()).permute(0, 3, 1, 2)[None, None].cuda().to(torch.bfloat16)/127.5-1
                        step = time.monotonic()
                        with torch.inference_mode():
                            generated = pipeline.generate(autoregressive_index=block, cache=cache, input=batch)
                            pipeline.finalize(autoregressive_index=block, cache=cache)
                        assert torch.isfinite(generated).all(), "nonfinite output"
                        frames = ((generated[0, 0].float().clamp(-1, 1)+1)*127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                        assert len(frames) == n
                        if block < 2:
                            early.append(frames.copy())
                        for frame in frames:
                            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                                writer.mux(packet)
                        cursor += n
                        entry = {"case_id": row["case_id"], "branch": branch, "block": block, "frames": cursor, "wall_s": time.monotonic()-step}
                        timings.append(entry)
                        print(json.dumps(entry), flush=True)
                    for packet in stream.encode():
                        writer.mux(packet)
                with av.open(str(out / f"{branch}.mp4")) as reader:
                    decoded = sum(1 for _ in reader.decode(video=0))
                assert decoded == 77
                prefixes[branch] = np.concatenate(early)
                result["branches"][branch] = {"decoded_frames": decoded, "generation_calls": 10,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30, "timings": timings}
                save(out / "result.json", result)
                del cache, conditions, generated, batch
                gc.collect()
                torch.cuda.empty_cache()
            result["raw_prefix_13frames_exactly_equal"] = bool(np.array_equal(prefixes["factual"], prefixes["counterfactual"]))
            assert result["raw_prefix_13frames_exactly_equal"], "相同输入前缀的随机性/状态一致性检查失败"
            result["status"] = "generation_complete"
        except BaseException as exc:
            result.update(status="oom_stopped" if isinstance(exc, torch.OutOfMemoryError) else "failed_stopped", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            result["wall_s"] = time.monotonic()-began
            save(out / "result.json", result)
        print(json.dumps({"case_complete": row["case_id"], "wall_s": result["wall_s"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["render", "encode", "generate"], required=True)
    args = parser.parse_args()
    rows = json.loads((args.inputs / "index.json").read_text())["cases"]
    args.output.mkdir(parents=True, exist_ok=True)
    stage_result = {"stage": args.stage, "status": "started", "cases": len(rows), "human_verdict": None}
    started = time.monotonic()
    try:
        if args.stage == "render":
            for row in rows:
                out = args.output / row["case_id"]
                out.mkdir(exist_ok=False)
                make_original(row["case"], row["camera_index"], out)
                result = render_case(row, out)
                print(json.dumps({"rendered": row["case_id"], "condition_visible": result["edit_visible_in_condition"]}), flush=True)
        elif args.stage == "encode":
            encode(rows, args.output)
        else:
            generate(rows, args.output)
        stage_result["status"] = "complete"
    except BaseException as exc:
        stage_result.update(status="oom_stopped" if isinstance(exc, torch.OutOfMemoryError) else "failed_stopped", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        stage_result["wall_s"] = time.monotonic()-started
        save(args.output / f"{args.stage}-result.json", stage_result)


if __name__ == "__main__":
    main()

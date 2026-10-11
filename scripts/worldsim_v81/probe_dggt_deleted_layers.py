#!/usr/bin/env python3
"""Frame-000 DGGT deletion layers: five saved-Gaussian RGB+ED renders, no Difix."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from probe_dggt_alpha_compositing import (
    HEIGHT, WIDTH, RUN_DEFAULT, acquire_gpu_locks, gpu_idle_preflight,
    paths_and_contract, quantize, rgb_png, write_json,
)

LAYERS = ("static_before", "static_delete", "dynamic_before", "dynamic_delete", "joint_delete")
VIEWS = ("black", "white", "alpha", "expected_depth", "depth_valid")
FRAME = 0
SOFT_SECONDS = 590  # The caller must also use an external 600-second hard process timeout.
JOINT_FLOAT_TOL = 1e-6
DISPLAY_ALPHA_MIN = 1 / 255  # Visualization validity only; never filters Gaussians or changes edits.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--output-dir", type=Path, help="Default: RUN/diagnostics/deleted_layers_frame000_r1")
    parser.add_argument("--external", type=Path, default=Path("/root/autodl-tmp/dggt"))
    parser.add_argument("--check-only", action="store_true", help="CPU path/manifest check without importing torch/CUDA")
    return parser.parse_args()


def asset_names() -> list[str]:
    return [f"assets/{layer}/{view}.png" for layer in LAYERS for view in VIEWS]


def cpu_contract(cli: argparse.Namespace) -> tuple[Path, Path, dict, dict, Path]:
    run = cli.run_root.resolve()
    out = (cli.output_dir or run / "diagnostics/deleted_layers_frame000_r1").resolve()
    if not out.is_relative_to(run) or out == run:
        raise ValueError("output-dir must be a new directory inside run-root")
    # Reuse the exact saved-state/trace/selector manifest contract from the paired probe.
    cli.output_dir = out
    _, _, manifest, source = paths_and_contract(cli)
    pair = run / "diagnostics/alpha_compositing_pair_r1"
    pair_record = json.loads((pair / "run.json").read_text(encoding="utf-8"))
    calibration = pair_record.get("calibration", {})
    if pair_record.get("status") != "complete" or pair_record.get("render_calls") != 8:
        raise ValueError("paired RGB+ED diagnostic has not completed")
    if calibration.get("old_noop_float_max_abs", 1) > 1e-4 or calibration.get("official_trace_noop_max_abs", 1) > 1e-4:
        raise ValueError("paired no-op calibration did not pass")
    if any(item.get("rgb_pixels") or item.get("alpha_pixels")
           for item in calibration.get("old_png_mismatch_pixels", {}).values()):
        raise ValueError("paired saved-PNG calibration did not pass")
    if len(calibration.get("old_png_mismatch_pixels", {})) != 8:
        raise ValueError("paired saved-PNG calibration is incomplete")
    if not (pair / "components_float.pt").is_file():
        raise FileNotFoundError(pair / "components_float.pt")
    return run, out, manifest, source, pair


def set_queue_environment(run: Path, external: Path) -> dict:
    # Match queue_dggt_waymo_inference.py so gsplat JIT finds CUDA, Ninja, and cache.
    updates = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "CUDA_HOME": "/usr/local/cuda", "MAX_JOBS": "1", "TORCH_CUDA_ARCH_LIST": "8.6",
               "TMPDIR": str(run / "tmp"), "TORCH_HOME": str(external / ".cache/torch"),
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": str(external)}
    (run / "tmp").mkdir(exist_ok=True)
    os.environ.update(updates)
    os.environ["PATH"] = (f"{Path(sys.executable).parent}:/root/autodl-tmp/envs/motionproj/bin:"
                          f"/usr/local/cuda/bin:{os.environ['PATH']}")
    return updates


def main() -> int:
    cli = parse_args()
    run, out, manifest, source, pair = cpu_contract(cli)
    record_path = out / "run.json"
    if out.exists():
        if record_path.is_file() and json.loads(record_path.read_text(encoding="utf-8")).get("status") == "complete":
            missing = [name for name in asset_names() + ["layers_float.pt"] if not (out / name).is_file()]
            if missing:
                raise RuntimeError(f"completed layer probe is missing files: {missing}")
            print(json.dumps({"status": "already_complete", "output_dir": str(out)}))
            return 0
        raise FileExistsError(f"preserving incomplete layer probe: {out}")
    if cli.check_only:
        print(json.dumps({"status": "cpu_contract_ok", "scene": manifest["scene"], "frame": FRAME,
                          "renders": len(LAYERS), "paired_run": str(pair),
                          "output_dir": str(out), "gpu_imported": False}))
        return 0

    locks = acquire_gpu_locks(run)  # P1 controller, P1 launch, shared DGGT diagnostic lock.
    try:
        preflight = gpu_idle_preflight()
        environment = set_queue_environment(run, cli.external.resolve())
        out.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        record = {"status": "running", "task": "dggt_deleted_layers_frame000_r1", "scene": "128",
                  "frame": FRAME, "render_budget": 5, "source_manifest": str(source["manifest"]),
                  "source_state": str(source["state"]), "source_trace": str(source["trace"]),
                  "paired_components": str(pair / "components_float.pt"), "gpu_preflight": preflight,
                  "environment": environment, "external_hard_timeout_seconds_required": 600,
                  "training_updates": 0, "model_forward_calls": 0, "difix_calls": 0,
                  "failure_ledger_delta": "none"}
        write_json(record_path, record)

        def soft_timeout(_signum, _frame):
            raise TimeoutError(f"layer probe exceeded {SOFT_SECONDS} seconds")

        signal.signal(signal.SIGALRM, soft_timeout)
        signal.alarm(SOFT_SECONDS)
        try:
            import numpy as np
            import torch
            from PIL import Image

            sys.path.insert(0, str(cli.external.resolve()))
            from dggt.utils.gs import concat_list
            from gsplat.rendering import rasterization

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("expected one available CUDA GPU after idle preflight")
            state = torch.load(source["state"], map_location="cpu", weights_only=False)
            trace = torch.load(source["trace"], map_location="cpu", weights_only=False)
            paired = torch.load(pair / "components_float.pt", map_location="cpu", weights_only=False)
            if (len(state["static"]) != 5 or len(state["dynamic"]) != 4 or
                    tuple(state["sky"].shape) != (4, HEIGHT, WIDTH, 3) or
                    tuple(trace["timestamps"].shape) != (4,) or
                    tuple(paired["hole_mask"].shape) != (4, HEIGHT, WIDTH)):
                raise ValueError("saved frame/state/hole contract changed")
            hole = paired["hole_mask"][FRAME].numpy().astype(bool)
            if not hole.any():
                raise ValueError("saved paired fixed hole is empty")
            device = torch.device("cuda:0")
            static = [item.to(device) for item in state["static"]]
            dynamic = [item.to(device) for item in state["dynamic"][FRAME]]
            static_selected = state["static_selected"].to(device)
            dynamic_selected = state["dynamic_selected"][FRAME].to(device)
            t0 = trace["timestamps"][FRAME].to(device)
            static_timestamps = state["static_timestamps"].to(device)
            static_conf = state["static_conf"].to(device)
            extrinsic = state["extrinsic"][FRAME:FRAME + 1].to(device)
            intrinsic = state["intrinsic"][FRAME:FRAME + 1].to(device)
            if tuple(static_selected.shape) != (len(static[0]),) or tuple(dynamic_selected.shape) != (len(dynamic[0]),):
                raise ValueError("saved Gaussian selection dimensions changed")

            sigma = torch.log(torch.tensor(0.1)).to(device) / (static_conf ** 2 + 1e-6)
            conf = torch.exp(sigma * (t0 - static_timestamps) ** 2)
            static_before = [static[0], static[1], (static[2] * conf).float(), static[3], static[4]]
            static_delete = [item[~static_selected] for item in static_before]
            dynamic_delete = [item[~dynamic_selected] for item in dynamic]
            joint_delete = concat_list(static_delete, dynamic_delete)
            groups = {"static_before": static_before, "static_delete": static_delete,
                      "dynamic_before": dynamic, "dynamic_delete": dynamic_delete,
                      "joint_delete": joint_delete}
            layers = {}
            render_seconds = {}
            for name in LAYERS:
                start_layer = time.monotonic()
                points, colors, opacity, scales, rotation = groups[name]
                render, alpha, _ = rasterization(
                    means=points, quats=rotation, scales=scales, opacities=opacity,
                    colors=colors, viewmats=extrinsic, Ks=intrinsic,
                    width=WIDTH, height=HEIGHT, render_mode="RGB+ED")
                layers[name] = {"G": render[0, ..., :3].detach().float().cpu(),
                                "ED": render[0, ..., 3].detach().float().cpu(),
                                "A": alpha[0, ..., 0].detach().float().cpu()}
                torch.cuda.synchronize()
                render_seconds[name] = round(time.monotonic() - start_layer, 3)
            torch.save({"layers": layers, "hole_mask": torch.from_numpy(hole),
                        "sky": state["sky"][FRAME],
                        "meaning": "G is gsplat accumulated RGB; ED is gsplat expected depth, not GT"},
                       out / "layers_float.pt")

            joint = layers["joint_delete"]
            pair_g = paired["G"]["delete"][FRAME]
            pair_a = paired["A"]["delete"][FRAME, ..., 0]
            if tuple(joint["G"].shape) != tuple(pair_g.shape) or tuple(joint["A"].shape) != tuple(pair_a.shape):
                raise ValueError("paired joint-delete float shape changed")
            g_max_abs = float((joint["G"] - pair_g).abs().max())
            a_max_abs = float((joint["A"] - pair_a).abs().max())
            g = joint["G"].numpy()
            a = joint["A"].numpy()[..., None]
            sky = state["sky"][FRAME].float().numpy()
            official_rgb = a * g + (1 - a) * sky
            old_rgb = rgb_png(source["delete_rgb_0"], np, Image)
            old_alpha = rgb_png(source["delete_alpha_0"], np, Image)
            rgb_mismatch = int(np.count_nonzero(np.any(quantize(official_rgb, np) != old_rgb, axis=-1)))
            alpha_mismatch = int(np.count_nonzero(np.any(
                np.repeat(quantize(a, np), 3, axis=-1) != old_alpha, axis=-1)))
            record["calibration"] = {"paired_joint_G_max_abs": g_max_abs,
                                     "paired_joint_A_max_abs": a_max_abs,
                                     "float_tolerance": JOINT_FLOAT_TOL,
                                     "old_delete_rgb_mismatch_pixels": rgb_mismatch,
                                     "old_delete_alpha_mismatch_pixels": alpha_mismatch}
            if g_max_abs > JOINT_FLOAT_TOL or a_max_abs > JOINT_FLOAT_TOL or rgb_mismatch or alpha_mismatch:
                raise RuntimeError("joint-delete calibration failed; layer PNGs withheld")

            valid = {}
            depth_union = []
            for name, data in layers.items():
                depth = data["ED"].numpy()
                alpha = data["A"].numpy()
                valid[name] = np.isfinite(depth) & (depth > 0) & np.isfinite(alpha) & (alpha >= DISPLAY_ALPHA_MIN)
                depth_union.append(depth[valid[name]])
            depth_union = np.concatenate(depth_union)
            if not len(depth_union):
                raise ValueError("no valid predicted expected-depth pixels in five layers")
            depth_min, depth_max = float(depth_union.min()), float(depth_union.max())
            display_low, display_high = [float(x) for x in np.percentile(depth_union, [2, 98])]
            if not display_high > display_low:
                raise ValueError("expected-depth display range collapsed")

            weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            stats = {}
            for name, data in layers.items():
                folder = out / "assets" / name
                folder.mkdir(parents=True, exist_ok=False)
                g = data["G"].numpy()
                a = data["A"].numpy()
                ed = data["ED"].numpy()
                white = g + (1 - a[..., None])  # Pure white, not DGGT sky or road GT.
                Image.fromarray(quantize(g, np)).save(folder / "black.png")
                Image.fromarray(quantize(white, np)).save(folder / "white.png")
                Image.fromarray(quantize(a, np)).save(folder / "alpha.png")
                shown = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                shown[valid[name]] = quantize((ed[valid[name]] - display_low) / (display_high - display_low), np)
                Image.fromarray(shown).save(folder / "expected_depth.png")
                Image.fromarray((valid[name].astype(np.uint8) * 255)).save(folder / "depth_valid.png")
                depth_hole = valid[name] & hole
                stats[name] = {"gaussians": int(len(groups[name][0])),
                               "hole_pixels": int(hole.sum()),
                               "hole_alpha_mean": float(a[hole].mean()),
                               "hole_black_luma_mean": float((g[hole] @ weights).mean()),
                               "hole_white_luma_mean": float((white[hole] @ weights).mean()),
                               "hole_valid_depth_pixels": int(depth_hole.sum()),
                               "hole_valid_expected_depth_mean": float(ed[depth_hole].mean()) if depth_hole.any() else None}

            missing = [name for name in asset_names() + ["layers_float.pt"] if not (out / name).is_file()]
            if missing:
                raise RuntimeError(f"layer outputs missing: {missing}")
            import resource
            record.update({"status": "complete", "render_calls": 5, "render_mode": "RGB+ED",
                           "asset_files": asset_names(), "float_layers": "layers_float.pt",
                           "fixed_hole_source": str(pair / "components_float.pt") + ":hole_mask[0]",
                           "depth_display": {"method": "shared 2nd-98th percentile clipped grayscale over all five valid layers",
                                             "valid_rule": "finite ED>0 and alpha>=1/255 (display only)",
                                             "raw_valid_min": depth_min, "raw_valid_max": depth_max,
                                             "display_low": display_low, "display_high": display_high},
                           "layer_statistics": stats, "render_seconds": render_seconds,
                           "interpretation": ["Standalone static/dynamic renders are not their actual contributions to the jointly occluded result.",
                                              "Before layers are predictions from the current model, not ground truth.",
                                              "Expected depth uses predicted coordinates, not ground truth; zero-alpha depths are not interpreted.",
                                              "White means a pure white visualization background, not an inpainted road."],
                           "seconds": round(time.monotonic() - started, 3),
                           "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)})
            write_json(record_path, record)
            print(json.dumps({"status": "complete", "output_dir": str(out),
                              "renders": 5, "assets": len(record["asset_files"]),
                              "joint_G_max_abs": g_max_abs, "joint_A_max_abs": a_max_abs}))
            return 0
        except BaseException as error:
            record.update({"status": "failed", "error_type": type(error).__name__, "error": str(error),
                           "traceback": traceback.format_exc(), "seconds": round(time.monotonic() - started, 3)})
            write_json(record_path, record)
            raise
        finally:
            signal.alarm(0)
    finally:
        for handle in locks:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

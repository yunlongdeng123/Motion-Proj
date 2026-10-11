#!/usr/bin/env python3
"""One-shot paired alpha-compositing diagnostic from the saved DGGT Gaussian state.

No DGGT prediction, target selection, training, or Difix is run. The same eight
gsplat RGB+ED renders feed both CPU compositing formulas. This script does not
change the official formula used by the existing editor.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

RUN_DEFAULT = Path("/root/autodl-tmp/runs/worldsim_v81/WS-V81-DGGT-WAYMO-INFERENCE-20261011/r1")
FRAMES = 4
HEIGHT, WIDTH = 350, 518
MAX_SECONDS = 600
MAX_NOOP_ABS = 1e-4


def acquire_gpu_locks(run: Path):
    import fcntl
    phase = Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-SEEN-TO-SCENE-P1-20261009/r1/paper_bidirectional_m4')
    handles = []
    try:
        for path in (phase/'controller.lock', phase/'review_cycles/launch.lock',
                     run/'diagnostics/alpha_compositing_pair.lock'):
            handle = path.open('a')
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        for handle in handles:
            handle.close()
        raise
    return handles


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--output-dir", type=Path, help="Default: RUN/diagnostics/alpha_compositing_pair_r1")
    parser.add_argument("--external", type=Path, default=Path("/root/autodl-tmp/dggt"))
    parser.add_argument("--check-only", action="store_true", help="CPU path/manifest check; never imports torch or touches CUDA")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def paths_and_contract(cli: argparse.Namespace) -> tuple[Path, Path, dict, dict]:
    run = cli.run_root.resolve()
    out = (cli.output_dir or run / "diagnostics/alpha_compositing_pair_r1").resolve()
    if not out.is_relative_to(run) or out == run:
        raise ValueError("output-dir must be a new directory inside run-root")
    manifest_path = run / "gaussian_edits/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status"), manifest.get("scene"), manifest.get("frames"),
            manifest.get("official_mode"), manifest.get("views")) != ("completed", "128", 4, 2, 1):
        raise ValueError("saved Gaussian edit manifest is not completed scene 128, mode 2, four frames, one view")
    if manifest.get("selection_config", {}).get("source") != "masks" or manifest.get("selection_config", {}).get("provenance") != "rgb_derived":
        raise ValueError("saved selector contract changed")
    trace_path = Path(manifest["official_trace"]).resolve()
    if trace_path != (run / "baseline_official/official_trace.pt").resolve():
        raise ValueError("manifest points outside the expected official trace")
    if len(manifest.get("selector_paths", [])) != FRAMES:
        raise ValueError("expected the four original selector paths")
    source = {
        "manifest": manifest_path,
        "state": run / "gaussian_edits/gaussian_state.pt",
        "trace": trace_path,
        "noop_float": run / "gaussian_edits/noop_rgb_float.npy",
    }
    for branch in ("noop", "delete"):
        for index in range(FRAMES):
            source[f"{branch}_rgb_{index}"] = run / f"gaussian_edits/{branch}/{index:03d}.png"
            source[f"{branch}_alpha_{index}"] = run / f"gaussian_edits/{branch}_alpha/{index:03d}.png"
    for index in range(FRAMES):
        source[f"input_{index}"] = run / f"gaussian_edits/input_rgb/{index:03d}.png"
        source[f"hole_{index}"] = run / f"gaussian_edits/delete_alpha_loss_hole/{index:03d}.png"
        source[f"selector_{index}"] = Path(manifest["selector_paths"][index]).resolve()
    missing = [str(path) for path in source.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("saved source contract missing: " + ", ".join(missing))
    if not (cli.external.resolve() / "dggt/utils/gs.py").is_file():
        raise FileNotFoundError("official DGGT source not found at --external")
    return run, out, manifest, source


def expected_assets() -> list[str]:
    files = []
    for formula in ("official", "premult"):
        for branch in ("noop", "delete"):
            files.extend(f"assets/{formula}/{branch}/{index:03d}.png" for index in range(FRAMES))
    files.extend(f"assets/delta/delete/{index:03d}.png" for index in range(FRAMES))
    files.extend(f"assets/input/{index:03d}.png" for index in range(FRAMES))
    files.extend(f"assets/sky/{index:03d}.png" for index in range(FRAMES))
    return files


def gpu_idle_preflight() -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES is empty; diagnostic needs the idle GPU")
    proc = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          text=True, capture_output=True, check=True, timeout=10)
    pids = [line.strip() for line in proc.stdout.splitlines() if line.strip() and line.strip().isdigit()]
    if pids or (proc.stdout.strip() and not pids):
        raise RuntimeError(f"GPU preflight found active/ambiguous compute processes: {proc.stdout.strip()}")
    memory = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True, timeout=10)
    memory_mb = [int(line.strip()) for line in memory.stdout.splitlines() if line.strip()]
    if len(memory_mb) != 1 or memory_mb[0] > 512:
        raise RuntimeError(f"GPU preflight expected one idle GPU, memory used MiB={memory_mb}")
    return {"compute_pids": [], "memory_used_mib": memory_mb[0]}


def quantize(array, np):
    # Same clamp -> multiply -> numpy.round -> uint8 order as infer_dggt_waymo_edits._rgb_u8.
    return (np.clip(array.astype(np.float32, copy=False), 0, 1) * 255).round().astype(np.uint8)


def rgb_png(path: Path, np, Image):
    with Image.open(path) as image:
        data = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if data.shape != (HEIGHT, WIDTH, 3):
        raise ValueError(f"unexpected RGB shape {path}: {data.shape}")
    return data


def mask_png(path: Path, np, Image):
    # Fixed domain from prepare_depth_order_cpu.py, with its original resize/crop.
    with Image.open(path) as image:
        if image.mode not in ("P", "L", "I", "I;16"):
            image = image.convert("L")
        height = round(image.height * (518 / image.width) / 14) * 14
        image = image.resize((518, height), Image.Resampling.NEAREST)
        if height > 518:
            top = (height - 518) // 2
            image = image.crop((0, top, 518, top + 518))
        data = np.asarray(image) > 0
    if data.shape != (HEIGHT, WIDTH):
        raise ValueError(f"unexpected selector shape {path}: {data.shape}")
    return data


def hole_domain(manifest: dict, state: dict, trace: dict, source: dict, np, Image):
    point_map = trace["point_map"]
    if tuple(point_map.shape) != (1, FRAMES, HEIGHT, WIDTH, 3):
        raise ValueError("official trace point_map shape changed")
    xyz_all = point_map[0].numpy()
    extrinsic = state["extrinsic"].numpy().astype(np.float64)
    result = []
    for index in range(FRAMES):
        selector = mask_png(source[f"selector_{index}"], np, Image)
        with Image.open(source[f"hole_{index}"]) as image:
            loss = np.asarray(image.convert("L"), dtype=np.uint8)
        if loss.shape != (HEIGHT, WIDTH):
            raise ValueError("saved loss-hole shape changed")
        xyz = xyz_all[index].reshape(-1, 3).astype(np.float64)
        camera_z = (xyz @ extrinsic[index, :3, :3].T + extrinsic[index, :3, 3])[:, 2].reshape(HEIGHT, WIDTH)
        result.append(selector & (loss > 0) & np.isfinite(camera_z) & (camera_z > 0))
    return np.stack(result)


def main() -> int:
    cli = args()
    run, out, manifest, source = paths_and_contract(cli)
    existing = out / "run.json"
    if out.exists():
        if existing.is_file() and json.loads(existing.read_text(encoding="utf-8")).get("status") == "complete":
            missing = [name for name in expected_assets() + ["components_float.pt"] if not (out / name).is_file()]
            if missing:
                raise RuntimeError(f"completed diagnostic is missing assets: {missing}")
            print(json.dumps({"status": "already_complete", "output_dir": str(out)}))
            return 0
        raise FileExistsError(f"preserving existing incomplete diagnostic directory: {out}")
    if cli.check_only:
        print(json.dumps({"status": "cpu_contract_ok", "scene": manifest["scene"], "frames": FRAMES,
                          "output_dir": str(out), "source_files": len(source), "gpu_imported": False}))
        return 0

    gpu_locks = acquire_gpu_locks(run)
    preflight = gpu_idle_preflight()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"status": "running", "task": "dggt_alpha_compositing_pair_r1", "scene": "128", "frames": FRAMES,
              "source_manifest": str(source["manifest"]), "source_trace": str(source["trace"]),
              "source_state": str(source["state"]), "gpu_preflight": preflight,
              "max_seconds": MAX_SECONDS, "training_updates": 0,
              "model_forward_calls": 0, "difix_calls": 0, "failure_ledger_delta": "none"}
    write_json(existing, record)

    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"diagnostic exceeded {MAX_SECONDS} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MAX_SECONDS)
    try:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        import numpy as np
        import torch
        from PIL import Image
        from gsplat.rendering import rasterization

        # Import exact official concat helper; no DGGT network is constructed.
        sys.path.insert(0, str(cli.external.resolve()))
        from dggt.utils.gs import concat_list

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("expected exactly one available CUDA GPU after idle preflight")
        device = torch.device("cuda:0")
        state = torch.load(source["state"], map_location="cpu", weights_only=False)
        trace = torch.load(source["trace"], map_location="cpu", weights_only=False)
        if (len(state["static"]) != 5 or len(state["dynamic"]) != FRAMES or
                len(state["dynamic_selected"]) != FRAMES or
                tuple(state["sky"].shape) != (FRAMES, HEIGHT, WIDTH, 3) or
                tuple(trace["timestamps"].shape) != (FRAMES,)):
            raise ValueError("saved Gaussian/timestamp/sky contract changed")
        hole = hole_domain(manifest, state, trace, source, np, Image)
        sky = state["sky"].detach().float().cpu().numpy()
        timestamps = trace["timestamps"].to(device)
        static = [item.to(device) for item in state["static"]]
        static_selected = state["static_selected"].to(device)
        static_timestamps = state["static_timestamps"].to(device)
        static_conf = state["static_conf"].to(device)
        dynamic = [[item.to(device) for item in frame] for frame in state["dynamic"]]
        dynamic_selected = [item.to(device) for item in state["dynamic_selected"]]
        extrinsic = state["extrinsic"].to(device)
        intrinsic = state["intrinsic"].to(device)
        if tuple(static_selected.shape) != (len(static[0]),) or any(
                tuple(dynamic_selected[i].shape) != (len(dynamic[i][0]),) for i in range(FRAMES)):
            raise ValueError("saved Gaussian selection dimensions changed")

        components = {"G": {}, "A": {}, "sky": torch.from_numpy(sky),
                      "hole_mask": torch.from_numpy(hole),
                      "meaning": {"G": "gsplat RGB+ED RGB channels, before sky", "A": "gsplat alpha",
                                  "sky": "saved DGGT sky prediction", "hole_mask": "fixed original selector AND quantized alpha-loss AND positive finite trace camera-z"}}
        render_seconds = {}
        for branch in ("noop", "delete"):
            branch_started = time.monotonic()
            g_frames, a_frames = [], []
            for index in range(FRAMES):
                # Exact static confidence-time weighting from the existing editor.
                sigma = torch.log(torch.tensor(0.1)).to(device) / (static_conf ** 2 + 1e-6)
                conf = torch.exp(sigma * (timestamps[index] - static_timestamps) ** 2)
                static_opacity_t = (static[2] * conf).float()
                static_frame = [static[0], static[1], static_opacity_t, static[3], static[4]]
                dynamic_frame = dynamic[index]
                if branch == "delete":
                    static_frame = [item[~static_selected] for item in static_frame]
                    dynamic_frame = [item[~dynamic_selected[index]] for item in dynamic_frame]
                means, colors, opacity, scales, rotation = concat_list(static_frame, dynamic_frame)
                render, alpha, _ = rasterization(
                    means=means, quats=rotation, scales=scales, opacities=opacity,
                    colors=colors, viewmats=extrinsic[index:index + 1], Ks=intrinsic[index:index + 1],
                    width=WIDTH, height=HEIGHT, render_mode="RGB+ED")
                g_frames.append(render[0, ..., :-1].detach().float().cpu())
                a_frames.append(alpha[0].detach().float().cpu())
            torch.cuda.synchronize()
            components["G"][branch] = torch.stack(g_frames)
            components["A"][branch] = torch.stack(a_frames)
            render_seconds[branch] = round(time.monotonic() - branch_started, 3)
        torch.save(components, out / "components_float.pt")

        results = {}
        for branch in ("noop", "delete"):
            g = components["G"][branch].numpy()
            a = components["A"][branch].numpy()
            official = a * g + (1 - a) * sky
            premult = g + (1 - a) * sky
            results[branch] = {"official": official, "premult": premult, "alpha": a}

        old_noop_float = np.load(source["noop_float"])
        if old_noop_float.shape != results["noop"]["official"].shape:
            raise ValueError("old noop float shape changed")
        max_abs = float(np.max(np.abs(old_noop_float - results["noop"]["official"])))
        official_trace_rgb = trace["rgb"].detach().float().cpu()
        if tuple(official_trace_rgb.shape) != (FRAMES, 3, HEIGHT, WIDTH):
            raise ValueError("official trace RGB shape changed")
        trace_max_abs = float(np.max(np.abs(
            official_trace_rgb.permute(0, 2, 3, 1).numpy() - results["noop"]["official"])))
        mismatch = {}
        for branch in ("noop", "delete"):
            for index in range(FRAMES):
                old_rgb = rgb_png(source[f"{branch}_rgb_{index}"], np, Image)
                old_alpha = rgb_png(source[f"{branch}_alpha_{index}"], np, Image)
                new_rgb = quantize(results[branch]["official"][index], np)
                new_alpha = np.repeat(quantize(results[branch]["alpha"][index], np), 3, axis=-1)
                mismatch[f"{branch}_{index:03d}"] = {
                    "rgb_pixels": int(np.count_nonzero(np.any(old_rgb != new_rgb, axis=-1))),
                    "alpha_pixels": int(np.count_nonzero(np.any(old_alpha != new_alpha, axis=-1)))}
        record["calibration"] = {"old_noop_float_max_abs": max_abs,
                                 "official_trace_noop_max_abs": trace_max_abs, "threshold": MAX_NOOP_ABS,
                                 "old_png_mismatch_pixels": mismatch}
        if max_abs > MAX_NOOP_ABS or trace_max_abs > MAX_NOOP_ABS or any(
                item["rgb_pixels"] or item["alpha_pixels"] for item in mismatch.values()):
            raise RuntimeError("saved-render calibration failed; paired PNGs withheld")

        def luma_mean(rgb, region):
            values = rgb[region]
            return float(np.mean(values @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))) if len(values) else None

        stats = []
        for index in range(FRAMES):
            region = hole[index]
            official_delete = results["delete"]["official"][index]
            premult_delete = results["delete"]["premult"][index]
            difference = premult_delete - official_delete
            abs_difference = np.abs(difference)
            stats.append({"frame": index, "hole_pixels": int(region.sum()),
                          "official_delete_hole_luma": luma_mean(official_delete, region),
                          "premult_delete_hole_luma": luma_mean(premult_delete, region),
                          "official_noop_hole_luma": luma_mean(results["noop"]["official"][index], region),
                          "sky_hole_luma": luma_mean(sky[index], region),
                          "G_delete_hole_luma": luma_mean(components["G"]["delete"][index].numpy(), region),
                          "alpha_delete_hole_mean": float(results["delete"]["alpha"][index, ..., 0][region].mean()) if region.any() else None,
                          "premult_minus_official_hole_luma": luma_mean(difference, region),
                          "premult_minus_official_hole_abs_mean": float(abs_difference[region].mean()) if region.any() else None,
                          "premult_minus_official_hole_abs_max": float(abs_difference[region].max()) if region.any() else None})

        for branch in ("noop", "delete"):
            for formula in ("official", "premult"):
                folder = out / "assets" / formula / branch
                folder.mkdir(parents=True, exist_ok=False)
                for index in range(FRAMES):
                    Image.fromarray(quantize(results[branch][formula][index], np)).save(folder / f"{index:03d}.png")
        for category in ("delta/delete", "input", "sky"):
            (out / "assets" / category).mkdir(parents=True, exist_ok=False)
        for index in range(FRAMES):
            delta = np.abs(results["delete"]["premult"][index] - results["delete"]["official"][index])
            Image.fromarray(quantize(delta, np)).save(out / f"assets/delta/delete/{index:03d}.png")
            shutil.copy2(source[f"input_{index}"], out / f"assets/input/{index:03d}.png")
            Image.fromarray(quantize(sky[index], np)).save(out / f"assets/sky/{index:03d}.png")

        missing = [name for name in expected_assets() + ["components_float.pt"] if not (out / name).is_file()]
        if missing:
            raise RuntimeError(f"output files missing: {missing}")
        import resource
        record.update({"status": "complete", "render_calls": 8, "render_mode": "RGB+ED",
                       "formulas": {"official": "A*G+(1-A)*sky", "premult": "G+(1-A)*sky"},
                       "fixed_hole_domain": "original RGB selector AND saved quantized delete_alpha_loss_hole>0 AND finite positive official-trace camera-z",
                       "hole_statistics": stats, "render_seconds": render_seconds,
                       "asset_files": expected_assets(), "float_components": "components_float.pt",
                       "seconds": round(time.monotonic() - started, 3),
                       "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                       "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)})
        write_json(existing, record)
        print(json.dumps({"status": "complete", "output_dir": str(out), "assets": len(record["asset_files"]),
                          "seconds": record["seconds"], "noop_max_abs": max_abs}))
        return 0
    except BaseException as error:
        record.update({"status": "failed", "error_type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc(), "seconds": round(time.monotonic() - started, 3)})
        write_json(existing, record)
        raise
    finally:
        signal.alarm(0)
        for handle in gpu_locks:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

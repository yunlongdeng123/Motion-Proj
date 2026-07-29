#!/usr/bin/env python3
"""把 M5 DGGT 固定窗口与 M4 AD-GS 冻结渲染做 common-observation 诊断。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path("/root/autodl-tmp/motion_proj")
ADGS = Path("/root/autodl-tmp/third_party/AD-GS")
ADGS_COMMIT = "9a208512e49c8ddbaa20387921d9648adcd21cb4"
FROZEN_EVALUATOR_FILES = {
    "adgs_image_utils.py": (
        ADGS / "utils/image_utils.py",
        "872a4507773b9378db9e0fefc90104dc474b27551af18ada73f000a7a00a4ba0",
    ),
    "adgs_loss_utils.py": (
        ADGS / "utils/loss_utils.py",
        "292aa275aa46e85a6f1564e374b2b4fd5e7e9b5b628957ceea753bfd3d0c601f",
    ),
    "adgs_lpips.py": (
        ADGS / "lpipsPyTorch/modules/lpips.py",
        "3bece0b9cf9943af5b043026458819d259df9bfe7c2a1d3ffc6c905e7e5aa2b4",
    ),
    "adgs_lpips_networks.py": (
        ADGS / "lpipsPyTorch/modules/networks.py",
        "dfa6f152b0e3fbc23ac3bfaea52b46e9473e64ad539ea22ab030c57af51c7f14",
    ),
    "adgs_lpips_utils.py": (
        ADGS / "lpipsPyTorch/modules/utils.py",
        "1bd4a7d4e7b43215497675ed852936fe4f27e7ea3068afe0b0e1f7cfb6c48570",
    ),
}
FROZEN_EVALUATOR_WEIGHTS = {
    "alex_lpips_linear": (
        Path("/root/.cache/torch/hub/checkpoints/alex.pth"),
        "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
    ),
    "alexnet_backbone": (
        Path("/root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"),
        "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
    ),
}
M4_AGGREGATE = Path(
    "/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/"
    "20260728T141204__aggregate6-s0-wm3090"
)
TASK_ID = "DR-M5-DGGT-NUSC-01"
COMPONENT = "common-observation-diagnostic"
SCENES = [
    "scene-0230",
    "scene-0242",
    "scene-0255",
    "scene-0295",
    "scene-0518",
    "scene-0749",
]
WINDOWS = [[10, 11, 12, 13], [34, 35, 36, 37], [66, 67, 68, 69]]
TEST_FRAMES = tuple(range(4, 60, 4))
TRAIN_FRAMES = tuple(frame for frame in range(60) if frame not in TEST_FRAMES)
CAMERA_COUNT = 3


def now():
    return dt.datetime.now(
        dt.timezone(dt.timedelta(hours=8))
    ).isoformat()


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload):
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.replace(str(temporary), str(path))


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def split_and_render_index(processed_frame, view):
    """返回 AD-GS shuffle=False render 的 split 与序号。"""

    if processed_frame < 0 or processed_frame >= 60:
        raise ValueError("processed_frame 越界: {}".format(processed_frame))
    if view < 0 or view >= CAMERA_COUNT:
        raise ValueError("view 越界: {}".format(view))
    if processed_frame in TEST_FRAMES:
        split = "test"
        frame_rank = TEST_FRAMES.index(processed_frame)
    else:
        split = "train"
        frame_rank = TRAIN_FRAMES.index(processed_frame)
    return split, frame_rank * CAMERA_COUNT + view


def mean_metrics(rows):
    names = ("PSNR", "SSIM", "LPIPS(ALEX)")
    if not rows:
        return None
    return {
        name: sum(row["metrics"][name] for row in rows) / len(rows)
        for name in names
    }


def source_scene_rows():
    rows = {}
    for row in load_jsonl(M4_AGGREGATE / "metrics.jsonl"):
        if row.get("type") == "scene":
            rows[row["scene"]] = row
    if sorted(rows) != sorted(SCENES):
        raise RuntimeError("M4 scene coverage 不完整: {}".format(sorted(rows)))
    return rows


def verify_frozen_evaluator():
    commit = subprocess.check_output(
        ["git", "-C", str(ADGS), "rev-parse", "HEAD"], universal_newlines=True
    ).strip()
    if commit != ADGS_COMMIT:
        raise RuntimeError(
            "AD-GS commit 已变化: {} != {}".format(commit, ADGS_COMMIT)
        )
    records = {"commit": commit, "sources": {}, "weights": {}}
    for name, (path, expected) in FROZEN_EVALUATOR_FILES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                "evaluator source hash 已变化: {} {} != {}".format(
                    path, actual, expected
                )
            )
        records["sources"][name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    for name, (path, expected) in FROZEN_EVALUATOR_WEIGHTS.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                "evaluator weight hash 已变化: {} {} != {}".format(
                    path, actual, expected
                )
            )
        records["weights"][name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return records


def initialize(run_dir, m5_run):
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("run 目录非空，禁止覆盖: {}".format(run_dir))
    m5_terminal = load_json(m5_run / "terminal.json")
    m5_summary = load_json(m5_run / "summary.json")
    m4_terminal = load_json(M4_AGGREGATE / "terminal.json")
    m4_summary = load_json(M4_AGGREGATE / "summary.json")
    if m5_terminal.get("status") != "done" or m5_summary.get("status") != "done":
        raise RuntimeError("M5 native 不是 done")
    if m4_terminal.get("status") != "done" or not m4_summary.get(
        "all_gates_passed"
    ):
        raise RuntimeError("M4 不是 all-gates-passed done")
    evaluator = verify_frozen_evaluator()

    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = run_dir / "source_snapshot"
    snapshot_dir.mkdir()
    runner = Path(__file__).resolve()
    snapshot = snapshot_dir / runner.name
    shutil.copy2(str(runner), str(snapshot))
    source_snapshot = {
        runner.name: {
            "path": str(snapshot),
            "bytes": snapshot.stat().st_size,
            "sha256": sha256_file(snapshot),
        }
    }
    for name, (source, _) in FROZEN_EVALUATOR_FILES.items():
        destination = snapshot_dir / name
        shutil.copy2(str(source), str(destination))
        source_snapshot[name] = {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    project_commit = subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], universal_newlines=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(PROJECT), "status", "--short"], universal_newlines=True
    )
    resolved = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": COMPONENT,
        "instance_id": run_dir.name,
        "m5_native_run": str(m5_run),
        "m4_aggregate_run": str(M4_AGGREGATE),
        "scenes": SCENES,
        "raw_windows": WINDOWS,
        "processed_frame_rule": "raw_frame - 10",
        "camera_count": CAMERA_COUNT,
        "expected_target_pairs": len(SCENES) * len(WINDOWS) * 4 * CAMERA_COUNT,
        "metric_source": "frozen 8-bit AD-GS render/gt PNG pairs",
        "metrics": ["PSNR", "SSIM", "LPIPS(ALEX)"],
        "metric_lpips_backbone": "Alex",
        "adgs_commit": ADGS_COMMIT,
        "frozen_evaluator": evaluator,
        "protocol_boundary": {
            "dggt": {
                "observations_per_window": "4 for 1-view; 12 for 3-view",
                "poses_as_input": False,
                "per_scene_optimization": False,
                "model_input_hw": [294, 518],
            },
            "adgs": {
                "training_observations_per_scene": 138,
                "held_out_targets_per_scene": 42,
                "poses_as_input": True,
                "per_scene_optimization": True,
                "iterations": 60000,
                "render_hw": [900, 1600],
            },
            "claim": "failure characterization only; not a matched leaderboard",
        },
    }
    resolved["config_fingerprint"] = canonical_sha256(resolved)
    atomic_json(run_dir / "resolved.yaml", resolved)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": COMPONENT,
        "instance_id": run_dir.name,
        "started_at": now(),
        "project_commit": project_commit,
        "project_git_status": git_status.splitlines(),
        "project_git_status_sha256": hashlib.sha256(
            git_status.encode("utf-8")
        ).hexdigest(),
        "config_fingerprint": resolved["config_fingerprint"],
        "m5_terminal_sha256": sha256_file(m5_run / "terminal.json"),
        "m5_summary_sha256": sha256_file(m5_run / "summary.json"),
        "m5_metrics_sha256": sha256_file(m5_run / "metrics.json"),
        "m4_summary_sha256": sha256_file(M4_AGGREGATE / "summary.json"),
        "source_snapshot": source_snapshot,
        "frozen_evaluator": evaluator,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(
        run_dir / "terminal.json",
        {"status": "running", "updated_at": now(), "failure": None},
    )
    return resolved


def evaluate(run_dir, m5_run, resolved):
    sys.path.insert(0, str(ADGS))
    import numpy as np
    import torch
    import torchvision
    from PIL import Image, __version__ as pillow_version
    from torchvision.transforms.functional import to_tensor

    from lpipsPyTorch.modules.lpips import LPIPS
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    if not torch.cuda.is_available():
        raise RuntimeError("AD-GS diagnostic 需要 CUDA 复算 Alex-LPIPS")
    m5_resolved = load_json(m5_run / "resolved.yaml")
    input_root = Path(m5_resolved["input_root"])
    mapping_path = input_root / "raw_to_staged.jsonl"
    mapping_rows = load_jsonl(mapping_path)
    if len(mapping_rows) != resolved["expected_target_pairs"]:
        raise RuntimeError(
            "M5 target mapping coverage 错误: {} != {}".format(
                len(mapping_rows), resolved["expected_target_pairs"]
            )
        )
    m4_rows = source_scene_rows()
    m5_metrics = load_json(m5_run / "metrics.json")
    dggt_1view = {
        row["pseudo_scene"]: row for row in m5_metrics["native_1view_rows"]
    }
    dggt_3view = {
        row["pseudo_scene"]: row for row in m5_metrics["native_3view_rows"]
    }
    criterion = LPIPS("alex", "0.1").cuda().eval()
    metric_rows = []
    metrics_path = run_dir / "metrics.jsonl"
    started = time.time()
    with torch.no_grad():
        for ordinal, mapping in enumerate(mapping_rows):
            scene = mapping["source_scene"]
            processed_frame = int(mapping["processed_frame"])
            view = int(mapping["view"])
            split, render_index = split_and_render_index(processed_frame, view)
            model_root = Path(m4_rows[scene]["results"]["path"]).parent
            pair_root = model_root / split / "ours_60000"
            render_path = pair_root / "renders/{:05d}.png".format(render_index)
            gt_path = pair_root / "gt/{:05d}.png".format(render_index)
            for path in (render_path, gt_path):
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError("AD-GS target pair 缺失: {}".format(path))
            source_path = Path(mapping["source_image"])
            with Image.open(str(source_path)) as image:
                source_pixels = np.asarray(image.convert("RGB")).copy()
            with Image.open(str(gt_path)) as image:
                gt_pixels = np.asarray(image.convert("RGB")).copy()
            if source_pixels.shape != gt_pixels.shape or not np.array_equal(
                source_pixels, gt_pixels
            ):
                raise RuntimeError(
                    "AD-GS gt 与 M5 source 像素不同: {} {}".format(
                        gt_path, source_path
                    )
                )
            source_pixel_sha256 = hashlib.sha256(
                source_pixels.tobytes()
            ).hexdigest()
            gt_pixel_sha256 = hashlib.sha256(gt_pixels.tobytes()).hexdigest()
            with Image.open(str(render_path)) as image:
                render = to_tensor(image.convert("RGB")).cuda()
            with Image.open(str(gt_path)) as image:
                gt = to_tensor(image.convert("RGB")).cuda()
            row = {
                "type": "adgs_common_target",
                "pseudo_scene": mapping["pseudo_scene"],
                "source_scene": scene,
                "raw_frame": int(mapping["raw_frame"]),
                "processed_frame": processed_frame,
                "view": view,
                "camera": mapping["camera"],
                "split": split,
                "render_index": render_index,
                "render": str(render_path),
                "ground_truth": str(gt_path),
                "source_image_file_sha256": mapping["source_image_sha256"],
                "ground_truth_file_sha256": sha256_file(gt_path),
                "source_pixel_sha256": source_pixel_sha256,
                "ground_truth_pixel_sha256": gt_pixel_sha256,
                "metrics": {
                    "PSNR": float(psnr(render[None], gt[None]).item()),
                    "SSIM": float(ssim(render[None], gt[None]).item()),
                    "LPIPS(ALEX)": float(
                        criterion(render[None], gt[None]).item()
                    ),
                },
            }
            metric_rows.append(row)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if (ordinal + 1) % 12 == 0:
                print(
                    "common target {}/{}".format(
                        ordinal + 1, len(mapping_rows)
                    ),
                    flush=True,
                )
            del render, gt

    comparisons = []
    for pseudo_scene in sorted(dggt_1view):
        rows = [row for row in metric_rows if row["pseudo_scene"] == pseudo_scene]
        rows_1view = [row for row in rows if row["view"] == 0]
        mapping = next(
            row for row in mapping_rows if row["pseudo_scene"] == pseudo_scene
        )
        comparisons.append({
            "pseudo_scene": pseudo_scene,
            "source_scene": mapping["source_scene"],
            "raw_window": sorted({
                int(row["raw_frame"])
                for row in mapping_rows
                if row["pseudo_scene"] == pseudo_scene
            }),
            "adgs_same_target_1view": {
                "count": len(rows_1view),
                "metrics": mean_metrics(rows_1view),
            },
            "dggt_native_1view": dggt_1view[pseudo_scene]["metrics"],
            "adgs_same_target_3view": {
                "count": len(rows),
                "metrics": mean_metrics(rows),
            },
            "dggt_native_3view": (
                dggt_3view[pseudo_scene]["metrics"]
                if pseudo_scene in dggt_3view
                else None
            ),
        })

    rows_1view = [row for row in metric_rows if row["view"] == 0]
    target_split_counts = {
        split: sum(row["split"] == split for row in metric_rows)
        for split in ("train", "test")
    }
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": COMPONENT,
        "instance_id": run_dir.name,
        "status": "done",
        "completed_at": now(),
        "target_mapping_count": len(metric_rows),
        "expected_target_mapping_count": resolved["expected_target_pairs"],
        "target_mapping_coverage": len(metric_rows) / resolved[
            "expected_target_pairs"
        ],
        "ground_truth_pixel_identity_count": len(metric_rows),
        "target_split_counts": target_split_counts,
        "adgs_same_target_1view": {
            "image_count": len(rows_1view),
            "window_count": len(comparisons),
            "mean": mean_metrics(rows_1view),
        },
        "adgs_same_target_3view": {
            "image_count": len(metric_rows),
            "window_count": len(comparisons),
            "mean": mean_metrics(metric_rows),
        },
        "dggt_native_1view": m5_metrics["native_1view_summary"],
        "dggt_native_3view": m5_metrics["native_3view_summary"],
        "dggt_native_3view_status": m5_metrics["native_3view_status"],
        "diagnostic_wall_seconds": time.time() - started,
        "evaluator_environment": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pillow": pillow_version,
        },
        "protocol_boundary": resolved["protocol_boundary"],
        "claim": (
            "同 target side-by-side failure characterization；输入帧、pose、分辨率"
            "和逐场景优化不同，禁止解释为 matched leaderboard"
        ),
    }
    atomic_json(run_dir / "mapping_audit.json", {
        "mapping_path": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "mapping_count": len(mapping_rows),
        "ground_truth_pixel_identity_count": len(metric_rows),
        "test_frames_processed": list(TEST_FRAMES),
    })
    atomic_json(run_dir / "comparison.json", comparisons)
    atomic_json(run_dir / "summary.json", summary)
    return summary


def write_artifacts(run_dir):
    names = [
        "manifest.json",
        "resolved.yaml",
        "mapping_audit.json",
        "comparison.json",
        "metrics.jsonl",
        "summary.json",
        "terminal.json",
    ]
    paths = [run_dir / name for name in names]
    paths.extend(sorted((run_dir / "source_snapshot").glob("*")))
    artifacts = []
    for path in paths:
        if path.is_file():
            artifacts.append({
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    atomic_json(run_dir / "artifacts.json", {"artifacts": artifacts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--m5-run", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    m5_run = Path(args.m5_run)
    try:
        resolved = initialize(run_dir, m5_run)
        summary = evaluate(run_dir, m5_run, resolved)
    except Exception as exc:
        if run_dir.exists():
            atomic_json(
                run_dir / "summary.json",
                {
                    "schema_version": 1,
                    "task_id": TASK_ID,
                    "component": COMPONENT,
                    "instance_id": run_dir.name,
                    "status": "blocked",
                    "completed_at": now(),
                    "failure": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            )
            atomic_json(
                run_dir / "terminal.json",
                {
                    "status": "blocked",
                    "updated_at": now(),
                    "failure": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            )
            write_artifacts(run_dir)
        raise
    atomic_json(
        run_dir / "terminal.json",
        {"status": "done", "updated_at": now(), "failure": None},
    )
    write_artifacts(run_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

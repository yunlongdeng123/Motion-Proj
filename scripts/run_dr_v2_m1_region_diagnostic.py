#!/usr/bin/env python3
"""M1 动态区域/边界诊断：只读复用冻结 DGGT 与 AD-GS 输出。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT = Path("/root/autodl-tmp/motion_proj")
TASK_ID = "DR-V2-M1-DGGT-REPAIR-01"
COMPONENT = "dynamic-boundary-diagnostic-v2"


def now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def boundary_band(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """返回二值区域内外各 radius 像素的形态学边界带。"""

    import cv2

    binary = mask.astype(np.uint8)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1).astype(bool)
    eroded = cv2.erode(binary, kernel, iterations=1).astype(bool)
    return np.logical_xor(dilated, eroded)


def region_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[2] != 3:
        raise RuntimeError(f"RGB shape 不一致: {pred.shape} / {gt.shape}")
    if mask.shape != pred.shape[:2]:
        raise RuntimeError(f"mask shape 不一致: {mask.shape} / {pred.shape[:2]}")
    count = int(mask.sum())
    if count == 0:
        raise RuntimeError("诊断区域为空")
    error = pred.astype(np.float64)[mask] - gt.astype(np.float64)[mask]
    mse = float(np.mean(error**2))
    return {
        "pixel_count": count,
        "rgb_mae_0_1": float(np.mean(np.abs(error)) / 255.0),
        "rgb_rmse_0_1": float(math.sqrt(mse) / 255.0),
        "psnr": float(10.0 * math.log10((255.0**2) / max(mse, 1e-12))),
    }


def extract_three_view(composite: np.ndarray, view: int) -> np.ndarray:
    if composite.shape != (294, 1574, 3):
        raise RuntimeError(f"3-view composite shape 错误: {composite.shape}")
    starts = {1: 0, 0: 528, 2: 1056}
    if view not in starts:
        raise RuntimeError(f"view 越界: {view}")
    start = starts[view]
    return composite[:, start : start + 518]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def load_dggt_gt(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((518, 294), Image.Resampling.BICUBIC)
        return np.asarray(resized).copy()


def load_dggt_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("L").resize((518, 294), Image.Resampling.NEAREST)
        return np.asarray(resized) != 0


def output_scene_dir(root: Path) -> Path:
    children = sorted(path for path in root.iterdir() if path.is_dir())
    if len(children) != 1:
        raise RuntimeError(f"DGGT output scene 目录数错误: {root} -> {children}")
    return children[0]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"row_count": len(rows)}
    for region in ("dynamic", "boundary"):
        result[region] = {
            "pixel_count": sum(row[region]["pixel_count"] for row in rows),
            "macro_mean": {
                metric: float(np.mean([row[region][metric] for row in rows]))
                for metric in ("rgb_mae_0_1", "rgb_rmse_0_1", "psnr")
            },
        }
    return result


def initialize(run_dir: Path, native_run: Path, common_run: Path) -> dict[str, Any]:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run 目录非空，禁止覆盖: {run_dir}")
    native_summary = load_json(native_run / "native_summary.json")
    common_summary = load_json(common_run / "summary.json")
    if native_summary.get("status") != "native_done":
        raise RuntimeError("native source 不是 native_done")
    if common_summary.get("status") != "done":
        raise RuntimeError("common source 不是 done")
    run_dir.mkdir(parents=True)
    for name in ("artifacts", "environment", "logs", "source_snapshot", "stages"):
        (run_dir / name).mkdir()
    shutil.copy2(Path(__file__).resolve(), run_dir / "source_snapshot" / Path(__file__).name)
    resolved = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": COMPONENT,
        "instance_id": run_dir.name,
        "native_run": str(native_run),
        "common_run": str(common_run),
        "mask_source": "frozen AD-GS semantic/mask_*.npy != 0",
        "dggt_mask_resize": "nearest to 294x518",
        "dggt_rgb_resize": "Pillow bicubic to 294x518, identical to upstream loader",
        "boundary_definition": "7x7 morphological dilation XOR erosion (radius=3 pixels)",
        "regions": ["dynamic", "boundary"],
        "region_metrics": ["RGB MAE [0,1]", "RGB RMSE [0,1]", "PSNR"],
        "expected_rows": {"adgs": 216, "dggt_1view": 72, "dggt_3view": 216},
        "claim": "failure characterization only; regional scores are resolution-specific",
    }
    write_json(run_dir / "resolved.yaml", resolved)
    status = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT, text=True)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": COMPONENT,
        "instance_id": run_dir.name,
        "status": "running",
        "started_at": now(),
        "project_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
        "project_git_status": status.splitlines(),
        "native_summary_sha256": sha256_file(native_run / "native_summary.json"),
        "native_metrics_sha256": sha256_file(native_run / "metrics.json"),
        "common_summary_sha256": sha256_file(common_run / "summary.json"),
        "common_metrics_sha256": sha256_file(common_run / "metrics.jsonl"),
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "terminal.json", {"status": "running", "updated_at": now(), "failure": None})
    (run_dir / "environment" / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        encoding="utf-8",
    )
    append_jsonl(
        run_dir / "resource.jsonl",
        {"timestamp": now(), "phase": "start", "disk_free_bytes": shutil.disk_usage(run_dir).free},
    )
    append_jsonl(
        run_dir / "logs" / "runner.jsonl",
        {"timestamp": now(), "event": "initialized"},
    )
    return resolved


def evaluate(run_dir: Path, native_run: Path, common_run: Path, resolved: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    native_resolved = load_json(native_run / "resolved.yaml")
    mapping_rows = load_jsonl(Path(native_resolved["input_root"]) / "raw_to_staged.jsonl")
    common_rows = load_jsonl(common_run / "metrics.jsonl")
    if len(mapping_rows) != 216 or len(common_rows) != 216:
        raise RuntimeError(f"source coverage 错误: mapping={len(mapping_rows)}, common={len(common_rows)}")
    common_by_key = {
        (row["pseudo_scene"], row["raw_frame"], row["view"]): row for row in common_rows
    }
    output_1 = {
        scene: output_scene_dir(native_run / "outputs" / f"1view_{scene}")
        for scene in {row["pseudo_scene"] for row in mapping_rows}
    }
    output_3 = {
        scene: output_scene_dir(native_run / "outputs" / f"3view_{scene}")
        for scene in {row["pseudo_scene"] for row in mapping_rows}
    }
    rows: list[dict[str, Any]] = []
    for mapping in mapping_rows:
        key = (mapping["pseudo_scene"], mapping["raw_frame"], mapping["view"])
        common = common_by_key[key]
        raw_mask = np.load(mapping["source_dynamic"]) != 0
        adgs_pred = load_rgb(Path(common["render"]))
        adgs_gt = load_rgb(Path(common["ground_truth"]))
        if raw_mask.shape != adgs_gt.shape[:2]:
            raise RuntimeError(f"AD-GS mask shape 错误: {raw_mask.shape} / {adgs_gt.shape}")
        adgs_boundary = boundary_band(raw_mask)
        rows.append(
            {
                "model": "adgs",
                "pseudo_scene": mapping["pseudo_scene"],
                "raw_frame": mapping["raw_frame"],
                "view": mapping["view"],
                "dynamic": region_metrics(adgs_pred, adgs_gt, raw_mask),
                "boundary": region_metrics(adgs_pred, adgs_gt, adgs_boundary),
            }
        )

        dggt_gt = load_dggt_gt(Path(mapping["staged_image"]))
        dggt_mask = load_dggt_mask(Path(mapping["staged_dynamic"]))
        dggt_boundary = boundary_band(dggt_mask)
        frame = int(mapping["staged_frame"])
        composite = load_rgb(output_3[mapping["pseudo_scene"]] / f"view_{frame:04d}.png")
        pred_3 = extract_three_view(composite, int(mapping["view"]))
        rows.append(
            {
                "model": "dggt_3view",
                "pseudo_scene": mapping["pseudo_scene"],
                "raw_frame": mapping["raw_frame"],
                "view": mapping["view"],
                "dynamic": region_metrics(pred_3, dggt_gt, dggt_mask),
                "boundary": region_metrics(pred_3, dggt_gt, dggt_boundary),
            }
        )
        if int(mapping["view"]) == 0:
            pred_1 = load_rgb(output_1[mapping["pseudo_scene"]] / f"view_{frame}.png")
            rows.append(
                {
                    "model": "dggt_1view",
                    "pseudo_scene": mapping["pseudo_scene"],
                    "raw_frame": mapping["raw_frame"],
                    "view": 0,
                    "dynamic": region_metrics(pred_1, dggt_gt, dggt_mask),
                    "boundary": region_metrics(pred_1, dggt_gt, dggt_boundary),
                }
            )
    for row in rows:
        append_jsonl(run_dir / "metrics.jsonl", row)
    grouped = {
        model: [row for row in rows if row["model"] == model]
        for model in ("adgs", "dggt_1view", "dggt_3view")
    }
    actual = {model: len(values) for model, values in grouped.items()}
    if actual != resolved["expected_rows"]:
        raise RuntimeError(f"regional coverage 错误: {actual}")
    summary = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "component": COMPONENT,
        "instance_id": run_dir.name,
        "status": "done",
        "completed_at": now(),
        "coverage": actual,
        "models": {model: summarize(values) for model, values in grouped.items()},
        "overall_reference": {
            "native": load_json(native_run / "native_summary.json"),
            "common_observation": load_json(common_run / "summary.json"),
        },
        "wall_seconds": time.time() - started,
        "boundary_definition": resolved["boundary_definition"],
        "claim": resolved["claim"],
        "next": "DR-V2-M2-ACTOR-EVAL-01",
    }
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(
        "# DR-V2 M1 dynamic/boundary diagnostic\n\n"
        "- status: `done`\n"
        f"- coverage: `{actual}`\n"
        "- boundary: `7x7 dilation XOR erosion`\n"
        "- claim: failure characterization only; scores are resolution-specific\n",
        encoding="utf-8",
    )
    return summary


def write_artifacts(run_dir: Path) -> None:
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".partial") or path.name == "artifacts.json":
            continue
        artifacts.append(
            {"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    write_json(run_dir / "artifacts.json", {"schema_version": 1, "artifacts": artifacts})


def run(run_dir: Path, native_run: Path, common_run: Path) -> dict[str, Any]:
    resolved = initialize(run_dir, native_run, common_run)
    try:
        summary = evaluate(run_dir, native_run, common_run, resolved)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        append_jsonl(run_dir / "logs" / "runner.jsonl", {"timestamp": now(), "event": "blocked", "failure": failure})
        write_json(run_dir / "summary.json", {"schema_version": 1, "task_id": TASK_ID, "status": "blocked", "failure": failure})
        write_json(run_dir / "terminal.json", {"status": "blocked", "updated_at": now(), "failure": failure})
        write_artifacts(run_dir)
        raise
    append_jsonl(
        run_dir / "resource.jsonl",
        {"timestamp": now(), "phase": "done", "disk_free_bytes": shutil.disk_usage(run_dir).free},
    )
    append_jsonl(
        run_dir / "logs" / "runner.jsonl",
        {"timestamp": now(), "event": "done", "coverage": summary["coverage"]},
    )
    write_json(run_dir / "stages" / "regional_metrics.json", {"status": "done", "coverage": summary["coverage"]})
    write_json(run_dir / "terminal.json", {"status": "done", "updated_at": now(), "failure": None})
    manifest = load_json(run_dir / "manifest.json")
    manifest.update({"status": "done", "completed_at": now()})
    write_json(run_dir / "manifest.json", manifest)
    write_artifacts(run_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--common-run", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run_dir.resolve(), args.native_run.resolve(), args.common_run.resolve())
    print(json.dumps(result["coverage"], ensure_ascii=False))


if __name__ == "__main__":
    main()

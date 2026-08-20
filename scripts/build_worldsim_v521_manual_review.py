#!/usr/bin/env python3
"""Build a compact image/video review package from frozen V5.2.1 panels."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from motion_proj.worldsim_v521.protocol import atomic_json, atomic_jsonl, atomic_text, inventory_files, sha256_file
from motion_proj.worldsim_v521.review_package import select_representative_cases


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def caption(row: dict) -> str:
    metric = row["severity_metric"]
    return (
        f"{row['review_order']:02d} {row['split_role']} {row['base']} {row['review_axis']}  "
        f"{row['scene']} frame={row['canonical_sample_index']} cam={row['camera']}  "
        f"PSNR={metric['psnr']:.3f} LPIPS={metric['lpips_alex']:.3f}  {row['cross_base_status']}"
    )


def labeled_image(source: Path, row: dict, *, width: int = 1400, canvas_height: int | None = None) -> Image.Image:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    height = round(image.height * width / image.width)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    title_height = 42
    target_height = canvas_height or height + title_height
    canvas = Image.new("RGB", (width, target_height), "white" if canvas_height is None else "black")
    y = title_height if canvas_height is None else max(title_height, (target_height - height) // 2)
    canvas.paste(image, (0, y))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), caption(row), fill="black" if canvas_height is None else "white")
    return canvas


def contact_sheet(rows: list[dict], copied: dict[str, Path], output: Path) -> None:
    images = [labeled_image(copied[row["case_id"]], row) for row in rows]
    sheet = Image.new("RGB", (1400, sum(image.height for image in images)), "white")
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(output, quality=94, subsampling=0)


def review_html(rows: list[dict], summary: dict) -> str:
    cards = []
    for row in rows:
        metric = row["severity_metric"]
        cards.append(
            "<article><h3>{}</h3><img src=\"images/{}\" loading=\"lazy\"><p>{}</p>"
            "<p>PSNR {:.3f} · SSIM {:.3f} · LPIPS-Alex {:.3f} · pixels {}</p>"
            "<p>Research direction: <code>{}</code></p></article>".format(
                html.escape(caption(row)), html.escape(row["review_filename"]),
                html.escape(row["case_id"] + " / " + row["event_id"]),
                metric["psnr"], metric["ssim"], metric["lpips_alex"], metric["pixel_count"],
                html.escape(row["research_direction"]),
            )
        )
    return """<!doctype html><meta charset=\"utf-8\"><title>V5.2.1 representative badcase review</title>
<style>body{{font-family:system-ui;margin:24px;background:#f5f5f5}}header,article{{background:white;padding:16px;margin:0 0 18px;border-radius:10px}}img{{width:100%;height:auto}}code{{font-size:13px}}</style>
<header><h1>V5.2.1 representative badcase review</h1><p>{}</p><p>Frozen automatic selection; no threshold/predicate/K refit and no manual case cherry-picking.</p></header>{}
""".format(html.escape(json.dumps(summary, ensure_ascii=False, sort_keys=True)), "".join(cards))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closeout-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.closeout_run.resolve()
    output = args.output_dir.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"review package already exists: {output}")
    registry_path = source / "BADCASE_REGISTRY.jsonl"
    panel_registry_path = source / "PANEL_REGISTRY.jsonl"
    rows, summary = select_representative_cases(read_jsonl(registry_path), read_jsonl(panel_registry_path))
    partial.mkdir(parents=True)
    (partial / "images").mkdir()
    copied: dict[str, Path] = {}
    for row in rows:
        source_panel = Path(row["panel_path"])
        if not source_panel.is_file() or sha256_file(source_panel) != row["panel_sha256"]:
            raise RuntimeError(f"frozen panel missing or hash drift: {source_panel}")
        filename = (
            f"{row['review_order']:02d}_{row['split_role']}_{row['base']}_{row['review_axis'].lower()}_"
            f"{row['scene']}_f{row['canonical_sample_index']:03d}_c{row['camera']}_{row['case_id']}.png"
        )
        destination = partial / "images" / filename
        shutil.copy2(source_panel, destination)
        if sha256_file(destination) != row["panel_sha256"]:
            raise RuntimeError(f"copied panel hash drift: {destination}")
        row["review_filename"] = filename
        row["review_path"] = str((output / "images" / filename))
        copied[row["case_id"]] = destination
    atomic_jsonl(partial / "REVIEW_CASES.jsonl", rows)
    atomic_json(partial / "REVIEW_SUMMARY.json", summary)
    for axis, name in (("GLOBAL_RGB", "01_global_rgb.jpg"), ("ACTOR_RGB", "02_actor_rgb.jpg"), ("BOUNDARY", "03_boundary.jpg")):
        contact_sheet([row for row in rows if row["review_axis"] == axis], copied, partial / name)
    atomic_text(partial / "REVIEW_INDEX.html", review_html(rows, summary))
    frames = partial / "video_frames"
    frames.mkdir()
    for index, row in enumerate(rows, start=1):
        frame = labeled_image(copied[row["case_id"]], row, width=1400, canvas_height=788)
        frame.save(frames / f"{index:03d}.png")
    ffmpeg = shutil.which("ffmpeg")
    video_status = "unavailable"
    video_artifact = None
    if ffmpeg:
        completed = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", "0.5",
                "-i", str(frames / "%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(partial / "review.mp4"),
            ],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {completed.stderr}")
        video_status = "done_mp4"
        video_artifact = "review.mp4"
    else:
        frame_paths = sorted(frames.glob("*.png"))
        animation = []
        for path in frame_paths:
            with Image.open(path) as opened:
                frame = opened.convert("RGB").resize((1000, 562), Image.Resampling.LANCZOS)
            animation.append(frame.quantize(colors=192, method=Image.Quantize.MEDIANCUT))
        animation[0].save(
            partial / "review.gif", save_all=True, append_images=animation[1:],
            duration=2000, loop=0, optimize=False,
        )
        video_status = "done_gif_fallback"
        video_artifact = "review.gif"
    shutil.rmtree(frames)
    readme = f"""# V5.2.1 人工审核包

- `REVIEW_INDEX.html`：逐 case 浏览，含 ID、指标和 research direction。
- `01_global_rgb.jpg` / `02_actor_rgb.jpg` / `03_boundary.jpg`：按 failure axis 的 contact sheet。
- `{video_artifact}`：每张代表面板停留 2 秒的轮播；当前状态 `{video_status}`。
- `images/`：原始冻结面板的逐图副本；每张均已复核 SHA-256。
- `REVIEW_CASES.jsonl`：机器可读的选择理由、指标、cross-base 状态和 provenance。

选择合同：每个 base×axis 固定取 Discovery severity anchor、不同 scene anchor 和 Confirmation direction anchor。
它不修改 taxonomy、threshold、K、predicate 或原 registry，也不把人工判断写回 benchmark。
"""
    atomic_text(partial / "README.md", readme)
    atomic_json(
        partial / "status.json",
        {
            "status": "done", "outcome": "representative_manual_review_package_ready",
            "video_status": video_status, "fresh_validation_test_kitti_read": False,
            "taxonomy_or_registry_modified": False,
        },
    )
    atomic_json(
        partial / "manifest.json",
        {
            "schema": "worldsim_v521_manual_review_manifest_v1",
            "source_closeout_run": str(source),
            "source_registry_sha256": sha256_file(registry_path),
            "source_panel_registry_sha256": sha256_file(panel_registry_path),
            "selection_summary": summary,
            "inventory_before_manifest": inventory_files(partial),
        },
    )
    partial.rename(output)
    print(json.dumps({"output": str(output), "summary": summary, "video_status": video_status}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

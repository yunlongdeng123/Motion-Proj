"""生成 P0 视频审核页，原生预测与可见区合成分开保存。"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import subprocess
import shutil

import numpy as np
from PIL import Image


def cropped_rgb(filename: str) -> np.ndarray:
    with Image.open(filename) as source:
        image = source.convert("RGB")
        scale = max(256 / image.width, 256 / image.height)
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.Resampling.BICUBIC)
        left, top = (image.width - 256) // 2, (image.height - 256) // 2
        return np.asarray(image.crop((left, top, left + 256, top + 256))).copy()


def encode(folder: Path, timestamps: list[int], output: Path) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    frames = sorted(folder.glob("*.png"))
    if len(frames) != 25:
        raise ValueError(f"视频必须恰有25帧: {folder}: {len(frames)}")
    deltas = np.diff(timestamps) / 1e6
    durations = [*deltas, float(np.median(deltas))]
    listing = folder / "frames.txt"
    listing.write_text("".join(f"file '{frame.resolve()}'\nduration {duration:.6f}\n"
                               for frame, duration in zip(frames, durations))
                       + f"file '{frames[-1].resolve()}'\n", encoding="utf-8")
    subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                    "-safe", "0", "-i", str(listing), "-fps_mode", "vfr", "-c:v", "libx264",
                    "-threads", "4", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
                   check=True)
    # 完整解码而非仅核对文件存在。
    subprocess.run([executable, "-v", "error", "-threads", "2", "-i", str(output), "-f", "null", "-"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    inputs = run / "fixed_inputs"
    if not inputs.exists():
        inputs = run / "data"
    output = run / "review"
    output.mkdir(exist_ok=True)
    rows, metadata = [], {}
    for split in ("train", "val"):
        rows.extend(json.loads(line) for line in (inputs / f"{split}.jsonl").read_text().splitlines() if line)
        metadata.update({item["video_id"]: item for item in json.loads(
            (inputs / f"{split}_metadata.json").read_text())})
    sections, videos, predicted_scenes = [], [], []
    for row in rows:
        vid, meta = row["video_id"], metadata[row["video_id"]]
        destination = output / vid
        destination.mkdir(exist_ok=True)
        generated = run / "inference" / vid
        has_prediction = all((generated / f"{i:05d}.png").is_file() for i in range(25))
        if has_prediction:
            predicted_scenes.append(meta["scene_name"])
        kinds = ["original", "condition"] + (["native", "composite"] if has_prediction else [])
        for kind in kinds:
            (destination / kind).mkdir(exist_ok=True)
        for index, filename in enumerate(row["frames"][:25]):
            target = cropped_rgb(filename)
            visible = target.copy()
            visible[:, :84] = 128
            visible[:, 172:] = 128
            Image.fromarray(target).save(destination / "original" / f"{index:05d}.png")
            Image.fromarray(visible).save(destination / "condition" / f"{index:05d}.png")
            if has_prediction:
                with Image.open(generated / f"{index:05d}.png") as image:
                    native = np.asarray(image.convert("RGB")).copy()
                if native.shape != target.shape:
                    raise ValueError("预测尺寸与原始裁切不一致")
                composite = target.copy()
                composite[:, :84], composite[:, 172:] = native[:, :84], native[:, 172:]
                Image.fromarray(native).save(destination / "native" / f"{index:05d}.png")
                Image.fromarray(composite).save(destination / "composite" / f"{index:05d}.png")
        for kind in kinds:
            mp4 = destination / f"{kind}.mp4"
            encode(destination / kind, meta["timestamps_us"], mp4)
            videos.append(str(mp4.relative_to(output)))
        titles = {"original": "真实完整 RGB（仅监督/审核）", "condition": "模型可见输入（灰色为洞）",
                  "native": "两步训练后的原生外绘", "composite": "可见区域保留原 RGB"}
        columns = "".join(f'<div><p>{titles[kind]}</p><video controls muted playsinline '
                          f'preload="metadata" src="{html.escape(vid)}/{kind}.mp4"></video></div>' for kind in kinds)
        if not has_prediction:
            columns += '<div><p>本场景尚未运行生成</p></div>'
        sections.append(f'<section id="{html.escape(meta["scene_name"])}"><h2>{html.escape(meta["scene_name"])} · {row["split"]}</h2>'
                        f'<p>25 个原始 next 帧；{meta["sweep_count"]} 张中间帧；'
                        f'平均 {meta["mean_fps"]:.2f} Hz。左右洞各84像素，中央可见88像素。'
                        '视频按原时间间隔编码；末帧展示一个中位帧间隔。'
                        '训练集按缓存可用性选择，不是泛化评测。人工 verdict 未填。</p>'
                        f'<button onclick="play(this)">本行同步播放</button><div class="grid">{columns}</div></section>')
    architecture = '<svg viewBox="0 0 1000 95" role="img" aria-label="architecture">'
    names = ["可见RGB + mask", "RAFT → Flow补全", "VAE → 潜变量传播", "SVD时序去噪", "解码 + 审核"]
    for i, name in enumerate(names):
        x = 5 + i * 200
        architecture += f'<rect x="{x}" y="12" width="180" height="56" rx="6" fill="#dbeafe"/><text x="{x+90}" y="45" text-anchor="middle" font-size="15">{name}</text>'
        if i < 4:
            architecture += f'<text x="{x+190}" y="46" text-anchor="middle" font-size="18">→</text>'
    architecture += '</svg>'
    page = '<!doctype html><meta charset="utf-8"><title>v8.1 P0 · nuScenes</title>'
    page += '<style>body{font:16px system-ui;margin:28px;background:#f4f6fa;color:#172033}section{background:white;padding:20px;margin:24px 0;border-radius:10px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}video{width:100%}svg{max-width:1100px;width:100%}p{line-height:1.6}button{margin:10px 0;padding:8px}</style>'
    page += '<h1>v8.1 P0：nuScenes 传播—生成闭环</h1><p>使用原始 SVD XT 1.1、RAFT、ProPainter Flow Completion 初始化。只做两步真实工程验收，不能据此评价外绘效果或 DELETE 收益；不是 YouTube-VOS 论文指标复现。采用公开训练源码的全帧传播与 Euler 短片采样，尚未复现 m=4 参考选择和 inversion。VAE每4帧解码，分块边界也需留意。</p>'
    links = '、'.join(f'<a href="#{html.escape(scene)}">{html.escape(scene)} 四列生成对照</a>'
                     for scene in predicted_scenes)
    page += (f'<p><strong>已生成 {len(predicted_scenes)} 例，共 {len(rows)} 例输入预览。</strong> '
             f'{links or "模型生成尚未完成"}。其余场景仅展示输入，不代表已运行模型。</p>')
    page += architecture + "".join(sections)
    page += '<script>function play(b){let vs=b.parentElement.querySelectorAll("video");vs.forEach(v=>{v.currentTime=0;v.play()})}</script>'
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "video_inventory.json").write_text(json.dumps({"clips": len(rows), "videos": videos,
                                                             "decoded_videos": len(videos)}, indent=2))
    print(output / "index.html", flush=True)


if __name__ == "__main__":
    main()

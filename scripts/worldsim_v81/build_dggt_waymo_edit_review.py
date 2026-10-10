#!/usr/bin/env python3
"""Build a local, evidence-only review board for one DGGT Waymo edit run.

Reads gaussian_edits and difix_refined manifests plus their saved PNG/MP4 media.
Copies only media displayed in the HTML. It never loads or copies model weights,
the official tensor trace, or serialized Gaussians.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path


EDIT_BRANCHES = ("noop", "delete", "move", "insert_copy")
GAUSSIAN_DIAGNOSTICS = ("input_rgb", "selected_target_overlay", "predicted_dynamic", "delete_alpha_loss_hole")
GAUSSIAN_ALPHA = ("noop_alpha", "delete_alpha", "move_alpha", "insert_copy_alpha")
TITLES = {
    "input_rgb": "输入 RGB", "selected_target_overlay": "选中目标 · RGB 控制", "predicted_dynamic": "DGGT 预测动态",
    "predicted_semantic_diagnostic": "DGGT 语义头 · 仅诊断", "delete_alpha_loss_hole": "删除后 alpha 缺口",
    "noop_alpha": "不编辑 alpha", "delete_alpha": "删除 alpha", "move_alpha": "平移 alpha", "insert_copy_alpha": "复制插入 alpha",
    "noop": "原始不编辑", "delete": "删除", "move": "平移", "insert_copy": "复制插入",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Run root containing gaussian_edits and difix_refined")
    parser.add_argument("--output-dir", required=True, type=Path, help="New local review directory; existing path is rejected")
    return parser.parse_args()


def _manifest(path: Path, role: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{role} manifest missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("status") != "completed":
        raise ValueError(f"{role} manifest does not report completed: {path}")
    return data


def _sequence(source: Path, name: str, expected_frames: int) -> tuple[list[Path], Path]:
    from PIL import Image

    folder = source / name
    frames = [folder / f"{i:03d}.png" for i in range(expected_frames)]
    if not folder.is_dir() or any(not path.is_file() for path in frames):
        raise FileNotFoundError(f"{name}: expected all {expected_frames} PNGs in {folder}")
    if sorted(folder.glob("*.png")) != frames:
        raise ValueError(f"{folder}: frame set differs from 000..{expected_frames - 1:03d}")
    dimensions = set()
    for path in frames:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            dimensions.add(image.size)
    if len(dimensions) != 1:
        raise ValueError(f"{folder}: PNG sizes vary across frames")
    video = source / f"{name}.mp4"
    if not video.is_file() or video.stat().st_size == 0:
        raise FileNotFoundError(f"{name}: existing MP4 missing or empty: {video}")
    return frames, video


def _card(group: str, item: dict) -> str:
    title = html.escape(item["title"])
    key = html.escape(group + "/" + item["name"], quote=True)
    first = html.escape(item["frames"][0], quote=True)
    video = html.escape(item["video"], quote=True)
    return (f'<article class="media-card" data-key="{key}"><header><strong>{title}</strong>'
            f'<span>4 帧</span></header><div class="media-stage">'
            f'<img class="frame-image" src="{first}" alt="{title} 第 1 帧" loading="lazy">'
            f'<video class="clip" src="{video}" muted loop playsinline preload="metadata" aria-label="{title} 视频"></video>'
            f'</div><footer><a href="{video}" download>下载原有 MP4</a></footer></article>')


def _format_fact(label: str, value) -> str:
    return f'<div class="fact"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'


def _page(payload: dict, gaussian: dict, difix: dict, review: dict | None) -> str:
    groups = payload["groups"]
    diagnostics = "".join(_card("gaussian", item) for item in groups["diagnostics"])
    original = "".join(_card("gaussian", item) for item in groups["gaussian"])
    refined = "".join(_card("difix", item) for item in groups["difix"])
    alpha = "".join(_card("gaussian", item) for item in groups["alpha"])
    selection = gaussian.get("selection", {})
    counts = gaussian.get("gaussian_counts", {})
    comparison = gaussian.get("official_noop_comparison")
    comparison_text = (f"max |ΔRGB| = {comparison['max_abs']:.3g}" if isinstance(comparison, dict) and isinstance(comparison.get("max_abs"), (int, float)) else "未提供官方 trace 数值对照")
    facts = "".join([
        _format_fact("场景", gaussian.get("scene", "未记录")),
        _format_fact("官方 no-op 核对", comparison_text),
        _format_fact("静态高斯 · 选中 / 总量", f"{counts.get('static_selected', '未记录')} / {counts.get('static_total', '未记录')}"),
        _format_fact("目标来源", selection.get("selection_provenance", "未记录")),
        _format_fact("Difix 模型加载次数", difix.get("model_loads", "未记录")),
    ])
    selector_details = json.dumps({
        "RGB_ROI_normalized_xyxy": selection.get("roi_box_normalized_xyxy"),
        "RGB_anchor_normalized_xy": selection.get("anchor_normalized_xy"),
        "first_frame_selection": selection.get("first_frame_selection"),
        "association": selection.get("associations"),
        "Gaussian_counts": counts,
        "edit_roles": gaussian.get("edit_roles"),
    }, ensure_ascii=False, indent=2)
    review_text = (json.dumps(review, ensure_ascii=False, indent=2) if review is not None else "等待 assistant_review.json；尚无助手审查结论。")
    review_label = "已读取原始 assistant_review.json" if review is not None else "等待助手审查"
    summary = html.escape(str((review or {}).get("summary_zh") or
                             (review or {}).get("assistant_assessment", {}).get("summary") or
                             "实际视频已保存，助手评审记录待完成。"))
    serialized = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DGGT Waymo · Gaussian 编辑人工复核</title>
<style>
:root{{--bg:#0a101b;--panel:#111c2c;--panel2:#142237;--line:#293a50;--text:#edf4ff;--muted:#9fb2c9;--cyan:#51d5e9;--amber:#ffca75;--violet:#bda9ff}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 80% 0,#152c48 0,transparent 35%),var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,'Segoe UI',sans-serif}}
main{{max-width:1660px;margin:auto;padding:32px 28px 80px}}h1{{font-size:clamp(27px,3.4vw,48px);line-height:1.13;letter-spacing:-.035em;margin:12px 0}}h2{{font-size:22px;margin:0 0 6px}}p{{margin:0}}.lead{{color:var(--muted);max-width:850px}}
.eyebrow{{color:var(--cyan);font-weight:700;letter-spacing:.16em;text-transform:uppercase;font-size:11px}}.hero{{display:flex;justify-content:space-between;gap:25px;align-items:end;margin-bottom:27px}}.status{{border:1px solid #396076;border-radius:999px;padding:7px 12px;color:#b8eff4;background:#102b3a;white-space:nowrap}}
.facts{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:20px 0 28px}}.fact{{background:var(--panel);border:1px solid var(--line);padding:13px 15px;border-radius:13px;min-height:74px}}.fact span{{display:block;color:var(--muted);font-size:11px;margin-bottom:5px}}.fact strong{{font-size:13px;overflow-wrap:anywhere}}
.flow{{display:flex;align-items:stretch;gap:9px;overflow:auto;padding:14px;background:#0e1929;border:1px solid var(--line);border-radius:16px;margin-bottom:26px}}.node{{min-width:162px;flex:1;background:var(--panel2);border:1px solid #31516c;border-radius:11px;padding:12px}}.node b{{display:block;color:#dff6fa;font-size:13px}}.node small{{display:block;color:var(--muted);font-size:11px;margin-top:4px}}.arrow{{display:flex;align-items:center;color:var(--cyan);font-size:22px}}
.toolbar{{position:sticky;top:0;z-index:10;background:#0a101be9;backdrop-filter:blur(12px);border:1px solid var(--line);border-radius:14px;padding:10px 14px;margin:0 0 30px;display:flex;gap:13px;align-items:center;flex-wrap:wrap}}button{{background:#1d3d57;border:1px solid #3b728b;border-radius:9px;color:var(--text);padding:8px 12px;cursor:pointer}}button:hover{{background:#28546f}}input[type=range]{{width:min(45vw,450px);accent-color:var(--cyan)}}#frame-indicator{{font-variant-numeric:tabular-nums;color:var(--cyan);font-weight:700}}.toolbar small{{color:var(--muted)}}
section{{margin:28px 0 36px}}.section-head{{margin-bottom:14px}}.section-head p{{color:var(--muted);font-size:13px}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}.media-card{{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}}.media-card header{{display:flex;justify-content:space-between;gap:8px;padding:9px 12px}}.media-card header strong{{font-size:13px}}.media-card header span{{font-size:11px;color:var(--muted)}}.media-stage{{aspect-ratio:1.48;background:#03060b;position:relative;display:flex;align-items:center;justify-content:center}}.media-stage img,.media-stage video{{width:100%;height:100%;object-fit:contain}}.media-stage video{{display:none}}body.video-mode .media-stage img{{display:none}}body.video-mode .media-stage video{{display:block}}.media-card footer{{padding:6px 12px 10px}}a{{color:var(--cyan);text-decoration:none;font-size:12px}}a:hover{{text-decoration:underline}}
.note{{padding:14px 17px;border-left:3px solid var(--amber);background:#32281755;color:#eddbc0;border-radius:0 12px 12px 0;font-size:13px;margin:12px 0}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:13px}}.record{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:17px}}.record h3{{margin:0 0 8px;font-size:16px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto;background:#09111e;border:1px solid #20344d;border-radius:9px;padding:13px;color:#cbdff5;font:12px/1.5 ui-monospace,Consolas,monospace}}.verdict{{border-color:#765a36}}.verdict strong{{color:var(--amber)}}
@media(max-width:1120px){{.facts{{grid-template-columns:repeat(3,1fr)}}.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:680px){{main{{padding:20px 12px 50px}}.hero{{display:block}}.status{{display:inline-block;margin-top:13px}}.facts{{grid-template-columns:repeat(2,1fr)}}.grid,.two{{grid-template-columns:1fr}}.flow{{display:grid;grid-template-columns:1fr}}.arrow{{justify-content:center;transform:rotate(90deg)}}input[type=range]{{width:100%}}}}
</style></head><body><main>
<div class="hero"><div><div class="eyebrow">DGGT · Waymo · local evidence board</div><h1>Gaussian 车辆编辑复核</h1><p class="lead">四帧统一浏览。上方是原始 Gaussian 渲染，下方是逐帧 Difix 外观处理；所有画面来自已保存的运行媒体。</p></div><div class="status">人工记录未填写 · 不阻塞实验</div></div>
<div class="facts">{facts}</div>
<div class="flow" aria-label="architecture components 图"><div class="node"><b>输入 RGB + RGB 语义</b><small>人工 ROI / 锚点选车</small></div><div class="arrow">→</div><div class="node"><b>DGGT 预测</b><small>相机 · 深度 · Gaussian · 动态</small></div><div class="arrow">→</div><div class="node"><b>Gaussian 编辑</b><small>删 / 平移 / 同场景复制插入</small></div><div class="arrow">→</div><div class="node"><b>原始四支渲染</b><small>alpha + sky；洞区可见</small></div><div class="arrow">→</div><div class="node"><b>Difix 逐帧</b><small>外观细化，非补洞证据</small></div></div>
<div class="toolbar"><button id="play-all" type="button">同步播放全部视频</button><button id="show-frames" type="button">查看逐帧 PNG</button><label for="frame-slider">共同帧</label><input id="frame-slider" type="range" min="0" max="3" value="0" step="1"><span id="frame-indicator">1 / 4</span><small>拖动滑条同时定位全部视图</small></div>
<section class="record"><h2>实测结论</h2><p>{summary}</p></section>
<section><div class="section-head"><h2>输入与目标选择</h2><p>红色覆盖区是逐帧 RGB 轮廓控制，预测动态仅作诊断；alpha 缺口显示删车后的透明度损失。</p></div><div class="grid">{diagnostics}</div></section>
<section><div class="section-head"><h2>原始 Gaussian 渲染</h2><p>四支共享 DGGT 预测、相机与渲染参数。复制插入仅复制本场景选中 Gaussian，不代表跨场景迁移。</p></div><div class="grid">{original}</div></section>
<section><div class="section-head"><h2>Difix 后处理</h2><p>同一组原始 PNG 逐帧细化。它可能改变外观，不能当作物体删除或几何补洞的证明。</p></div><div class="grid">{refined}</div></section>
<section><div class="section-head"><h2>alpha 与可见缺口</h2><p>对照透明度，检查删车后剩余 Gaussian 或 sky 是否显露空洞。</p></div><div class="grid">{alpha}</div></section>
<div class="note">选车使用 RGB 语义与逐帧轮廓控制，已排除邻车误分；边界仍可能有误差。本例删除后背景未完整补全。原始 Gaussian 编辑与 Difix 外观处理分别保留。</div>
<section class="two"><div class="record"><h3>选择与数量证据</h3><pre>{html.escape(selector_details)}</pre></div><div class="record"><h3>助手审查 · {html.escape(review_label)}</h3><pre>{html.escape(review_text)}</pre></div></section>
<section class="record verdict"><h3>人工记录</h3><strong>human_verdict：未填写</strong><p class="lead">本页面不代填人工判断；独立助手评审已单独记录，不需要人工放行。</p></section>
</main><script id="review-data" type="application/json">{serialized}</script><script>
"use strict";
const data = JSON.parse(document.getElementById("review-data").textContent);
const sequences = Object.values(data.groups).flat();
const byKey = Object.fromEntries(sequences.map(item => [item.group + "/" + item.name, item]));
const slider = document.getElementById("frame-slider");
const indicator = document.getElementById("frame-indicator");
const cards = [...document.querySelectorAll(".media-card")];
const videos = cards.map(card => card.querySelector("video"));
let playing = false;
function frameIndex() {{ return Number(slider.value); }}
function setFrame(index) {{
  slider.value = String(Math.max(0, Math.min(3, index)));
  indicator.textContent = `${{frameIndex() + 1}} / 4`;
  for (const card of cards) {{
    const item = byKey[card.dataset.key];
    const img = card.querySelector("img");
    img.src = item.frames[frameIndex()];
    img.alt = `${{item.title}} 第 ${{frameIndex() + 1}} 帧`;
  }}
}}
function pauseAll() {{ videos.forEach(video => video.pause()); playing = false; document.getElementById("play-all").textContent = "同步播放全部视频"; }}
document.getElementById("show-frames").addEventListener("click", () => {{ pauseAll(); document.body.classList.remove("video-mode"); setFrame(frameIndex()); }});
document.getElementById("play-all").addEventListener("click", async () => {{
  if (playing) {{ pauseAll(); return; }}
  document.body.classList.add("video-mode");
  const time = frameIndex() / 8;
  videos.forEach(video => {{ video.currentTime = time; }});
  try {{ await Promise.all(videos.map(video => video.play())); playing = true; document.getElementById("play-all").textContent = "暂停全部视频"; }}
  catch (error) {{ pauseAll(); alert("浏览器无法播放本地 MP4：" + error.message); }}
}});
slider.addEventListener("input", () => {{ setFrame(frameIndex()); videos.forEach(video => {{ if (video.readyState > 0) video.currentTime = frameIndex() / 8; }}); }});
function followVideos() {{
  if (playing && videos.length) {{
    const leader = videos[0];
    const t = leader.currentTime;
    for (const video of videos.slice(1)) {{ if (Math.abs(video.currentTime - t) > 0.075) video.currentTime = t; }}
    setFrame(Math.min(3, Math.floor(t * 8)));
  }}
  requestAnimationFrame(followVideos);
}}
setFrame(0); requestAnimationFrame(followVideos);
</script></body></html>"""


def main() -> int:
    args = _arguments()
    run = args.run.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing review directory: {output}")
    gaussian_dir = run / "gaussian_edits"
    difix_dir = run / "difix_refined"
    gaussian = _manifest(gaussian_dir / "manifest.json", "Gaussian")
    difix = _manifest(difix_dir / "manifest.json", "Difix")
    frame_count = gaussian.get("frames")
    if frame_count != 4:
        raise ValueError(f"expected four Gaussian input frames, found {frame_count!r}")
    if difix.get("model_loads") != 1:
        raise ValueError("Difix manifest must confirm one model load")
    difix_branches = difix.get("source_pngs", {})
    if not isinstance(difix_branches, dict) or any(not isinstance(difix_branches.get(name), list) or len(difix_branches[name]) != 4 for name in EDIT_BRANCHES):
        raise ValueError("Difix manifest must list four source PNGs for each of the four edit branches")
    if Path(difix.get("source_edits_dir", "")).resolve() != gaussian_dir.resolve():
        raise ValueError("Difix manifest source_edits_dir does not match this run's gaussian_edits")
    for name in EDIT_BRANCHES:
        expected = [gaussian_dir / name / f"{i:03d}.png" for i in range(frame_count)]
        if [Path(path).resolve() for path in difix_branches[name]] != [path.resolve() for path in expected]:
            raise ValueError(f"Difix {name} source PNGs do not match this run's Gaussian branch")
    if not isinstance(gaussian.get("gaussian_counts"), dict) or "insert_copy" not in gaussian.get("edit_roles", {}):
        raise ValueError("Gaussian manifest lacks insert_copy role/count evidence")
    if gaussian.get("official_trace"):
        comparison = gaussian.get("official_noop_comparison")
        if not isinstance(comparison, dict) or not isinstance(comparison.get("max_abs"), (int, float)) or not 0 <= comparison["max_abs"] <= 1e-4:
            raise ValueError("Gaussian run cites an official trace without a valid passing no-op numerical comparison")

    names = {
        "diagnostics": list(GAUSSIAN_DIAGNOSTICS),
        "gaussian": list(EDIT_BRANCHES),
        "difix": list(EDIT_BRANCHES),
        "alpha": list(GAUSSIAN_ALPHA),
    }
    if gaussian.get("predicted_semantic_exported"):
        names["diagnostics"].append("predicted_semantic_diagnostic")
    plan = []
    groups = {key: [] for key in names}
    for group, sequence_names in names.items():
        source = difix_dir if group == "difix" else gaussian_dir
        media_group = "difix" if group == "difix" else "gaussian"
        for name in sequence_names:
            frames, video = _sequence(source, name, frame_count)
            rel_frames = [f"assets/{media_group}/{name}/{frame.name}" for frame in frames]
            rel_video = f"assets/{media_group}/{name}.mp4"
            item = {"group": media_group, "name": name, "title": TITLES[name], "frames": rel_frames, "video": rel_video}
            groups[group].append(item)
            plan.extend(zip(frames, rel_frames))
            plan.append((video, rel_video))
    review_path = run / "assistant_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else None
    if review is not None and not isinstance(review, dict):
        raise ValueError("assistant_review.json must be a JSON object")
    payload = {"frames": frame_count, "groups": groups}
    document = _page(payload, gaussian, difix, review)
    output.mkdir(parents=True, exist_ok=False)
    for source, relative in plan:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if not target.is_file() or target.stat().st_size == 0:
            raise OSError(f"media copy failed: {target}")
    (output / "index.html").write_text(document, encoding="utf-8")
    missing = [str(output / relative) for _, relative in plan if not (output / relative).is_file()]
    if missing:
        raise RuntimeError(f"review media links missing after copy: {missing}")
    print(json.dumps({"status": "completed", "html": str(output / "index.html"),
                      "copied_media": len(plan), "frames": frame_count,
                      "assistant_review": "loaded" if review is not None else "pending",
                      "human_verdict": None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

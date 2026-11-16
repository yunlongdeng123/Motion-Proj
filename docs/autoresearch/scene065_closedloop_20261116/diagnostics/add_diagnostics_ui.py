"""Add synchronized contribution diagnostics only after the complete delivery exists.

Read-only validation:
    python add_diagnostics_ui.py
Publish the authorized UI extension:
    python add_diagnostics_ui.py --apply

All 398 images, hashes, dimensions and per-frame statistics are checked before
any page is written. Existing pages are backed up once, and repeated application
replaces the marked extension without duplicating it. No SSH/model execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape, unescape
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from bs4 import BeautifulSoup
from PIL import Image

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent.parent
PLATFORM = Path("C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a/outputs/GS_Scene_Diagnostics_20261027/15021599536622641101_556_150_576_150/native_closedloop_20261116_r2")
REPORT = ROOT / "outputs/scene065_closedloop_20261116/index.html"
SCHEMA = "scene065_contribution_diagnostics.v1"
RAW_IDS = list(range(199))
BASE_COLUMNS = ["closedloopObserved", "closedloopOldDelete", "closedloopNewDelete",
                "closedloopRoadOnly", "closedloopStaticContribution", "closedloopNative30"]
STOPS = [[0, "#0b1424"], [.25, "#285ca5"], [.5, "#2a9d9b"], [.75, "#ecb84e"], [1, "#fff2bd"]]
COLORS = {"gap": "#ff4d67", "thin": "#ffba49", "occluded": "#39c6e6"}

STYLE = """/* contribution-diagnostics:v1 */
.diagnostics-section{margin:22px 0 0;border-top:1px solid #29425d;padding-top:18px}
.diagnostics-heading{display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin-bottom:12px}
.diagnostics-heading h2{font-size:20px;margin:0}.diagnostics-heading label,.diagnostic-options{font-size:14px;color:#b5c8da}
.native-diagnostics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.native-diagnostics article{min-width:0;background:#101c2b;border:1px solid #29425d;border-radius:10px;padding:12px}
.native-diagnostics h3{font-size:17px;margin:0 0 8px}.native-diagnostics canvas{display:block;width:100%;height:auto;aspect-ratio:3/2;background:#08101b;cursor:crosshair}
.diagnostic-options{display:flex;gap:12px;align-items:center;min-height:28px;margin-bottom:8px}
.diagnostic-caption{font-size:13px;line-height:1.65;color:#c4d6e6;margin:9px 0 0;font-variant-numeric:tabular-nums}
.diagnostic-legend{display:flex;gap:8px 16px;flex-wrap:wrap;color:#b5c8da;font-size:13px;line-height:1.6;margin-top:8px}
.diagnostic-legend span{display:inline-flex;align-items:center;gap:6px}.diagnostic-swatch{width:11px;height:11px;border-radius:2px;display:inline-block;flex:none}
.diagnostic-scale{max-width:440px;margin-top:11px}.diagnostic-scale-bar{height:12px;border:1px solid #52667d;border-radius:3px;background:linear-gradient(to right,#0b1424 0%,#285ca5 25%,#2a9d9b 50%,#ecb84e 75%,#fff2bd 100%)}
.diagnostic-scale-ticks{display:flex;justify-content:space-between;color:#b5c8da;font-size:12px;margin-top:3px}
.diagnostic-definition{color:#a7bbce;font-size:13px;line-height:1.65;margin:8px 0 0}
.diagnostic-flow{margin:12px 0 0;padding:10px 12px;border:1px solid #29425d;border-radius:8px;background:#0c1723}
.diagnostic-flow svg{display:block;max-width:100%;width:860px;height:auto;margin:auto}
@media(max-width:900px){.native-diagnostics{grid-template-columns:1fr}}
"""

REPORT_STYLE = """/* contribution-diagnostics:v1 */
.diagnostic-report-legend{border:1px solid var(--line);border-radius:10px;padding:12px 15px;background:#101d2a;margin:12px 0}
.diagnostic-report-legend p{font-size:12px;color:var(--sub);line-height:1.7}.diagnostic-report-legend p+p{margin-top:6px}
.diagnostic-report-legend .swatch{display:inline-block;width:10px;height:10px;vertical-align:middle;margin-right:4px;border-radius:2px}
.diagnostic-report-scale{max-width:440px;margin:8px 0}.diagnostic-report-scale .bar{height:11px;border:1px solid #52667d;border-radius:3px;background:linear-gradient(to right,#0b1424 0%,#285ca5 25%,#2a9d9b 50%,#ecb84e 75%,#fff2bd 100%)}
.diagnostic-report-scale .ticks{display:flex;justify-content:space-between;font-size:11px;color:var(--sub)}
.diagnostic-dataflow{margin-top:10px}.diagnostic-dataflow svg{display:block;max-width:100%;width:860px;height:auto}
"""

FLOW = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 164" role="img" aria-labelledby="diagnostic-flow-title">
<title id="diagnostic-flow-title">同相机贡献诊断：冻结判定域、统一深度贡献与同步输出</title>
<defs><marker id="diagnostic-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#80a9c7"/></marker></defs>
<g fill="#15283b" stroke="#42617e" stroke-width="1.4"><rect x="8" y="48" width="177" height="68" rx="8"/><rect x="228" y="48" width="192" height="68" rx="8"/><rect x="468" y="9" width="179" height="59" rx="8"/><rect x="468" y="96" width="179" height="59" rx="8"/><rect x="694" y="48" width="158" height="68" rx="8"/></g>
<g fill="#e5eef6" font-family="Segoe UI,Microsoft YaHei,sans-serif" font-size="15" text-anchor="middle"><text x="96" y="75">场景与同帧相机</text><text x="96" y="98" fill="#a7bbce" font-size="12">冻结道路判定域</text><text x="324" y="75">同相机分层渲染</text><text x="324" y="98" fill="#a7bbce" font-size="12">道路单层 α / 同排序 αT</text><text x="557" y="34">道路支撑分类</text><text x="557" y="54" fill="#a7bbce" font-size="12">仅在冻结域内判定</text><text x="557" y="121">静态 αT 热图</text><text x="557" y="141" fill="#a7bbce" font-size="12">全画面 / 固定 0–1</text><text x="773" y="75">199 帧同步对照</text><text x="773" y="98" fill="#a7bbce" font-size="12">原始六视图 + 两项诊断</text></g>
<g fill="none" stroke="#80a9c7" stroke-width="2" marker-end="url(#diagnostic-arrow)"><path d="M185 82H228"/><path d="M420 82H442V38H468"/><path d="M442 82V125H468"/><path d="M647 38H671V82H694"/><path d="M647 125H671V82"/></g></svg>"""

DOMAIN = "只在待恢复道路范围内判定；保留车辆保护区与范围外不高亮。白色细线仅表示冻结判定域边界。"
GAP_LEGEND = """<div class="diagnostic-legend" aria-label="道路支撑诊断图例"><span><i class="diagnostic-swatch" style="background:#ff4d67"></i>道路支撑缺口：α &lt; 0.05</span><span><i class="diagnostic-swatch" style="background:#ffba49"></i>薄覆盖：0.05 ≤ α ≤ 0.9</span><span><i class="diagnostic-swatch" style="background:#39c6e6"></i>支撑充分、成像贡献低：α &gt; 0.9 且道路 αT &lt; 0.5</span></div>"""
SCALE = """<div class="diagnostic-scale" aria-label="静态贡献固定色标 0 到 1"><div class="diagnostic-scale-bar"></div><div class="diagnostic-scale-ticks"><span>0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1</span></div></div>"""
PANELS = """<section class="diagnostics-section" id="contribution-diagnostics" aria-labelledby="diagnostics-title">
<div class="diagnostics-heading"><h2 id="diagnostics-title">贡献诊断</h2><label><input type="checkbox" id="diagnosticsVisible" checked aria-controls="diagnostic-panels"> 显示诊断</label></div>
<div id="diagnostic-panels"><div class="native-diagnostics">
<article><h3>道路支撑与成像贡献</h3><div class="diagnostic-options"><label><input type="checkbox" id="diagnosticsGapHighlight" checked> 支撑诊断高亮</label></div>
<canvas id="closedloopRoadGap" width="960" height="640" tabindex="0" aria-label="冻结道路域内的支撑缺口、薄覆盖与充分支撑低贡献诊断"></canvas>
<p class="diagnostic-caption" id="diagnosticsGapStats" aria-live="polite"></p>""" + GAP_LEGEND + '<p class="diagnostic-definition">' + DOMAIN + """</p><p class="diagnostic-definition">支撑缺口不等于已证实的三维几何缺失；低成像贡献也可能来自正常前景遮挡。</p></article>
<article><h3>静态可见贡献 · αT</h3><div class="diagnostic-options">同一深度排序下 Σ<sub>i ∈ 静态</sub> α<sub>i</sub>T<sub>i</sub></div>
<canvas id="closedloopStaticAlphaT" width="960" height="640" tabindex="0" aria-label="全画面静态可见贡献 alphaT，固定 0 至 1 色标"></canvas>
<p class="diagnostic-caption" id="diagnosticsStaticStats" aria-live="polite"></p>""" + SCALE + """<p class="diagnostic-definition">静态行含未归属点，T 为该点前方透过率。全画面、所有帧共用固定 0–1 色标；0 表示无静态贡献，道路或动态层也可能接管该像素，不能据此判为缺口。</p></article>
</div><details class="diagnostic-flow"><summary>诊断数据流与数据来源</summary>""" + FLOW + """<p class="diagnostic-definition">道路 α 来自道路单独渲染；道路与静态 αT 来自完整场景的同一深度排序。计数和均值来自浮点数据。道路高亮使用 WebP 轻预览，静态热图使用固定 256 级色标 PNG；完整浮点数据与原图已保留。<a href="diagnostic_delivery/manifest.json">诊断清单</a></p></details></div></section>"""

PLUGIN_JS = r"""/* scene065 contribution-diagnostics:v1; shared viewer owns all navigation/crops. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let api = null;
  const number = value => Number(value).toLocaleString('zh-CN');
  const part = (count, total) => total ? `${number(count)} / ${number(total)} px (${(100 * count / total).toFixed(2)}%)` : '—';
  window.ROAD_DIAGNOSTICS = {
    columns: [['closedloopRoadGap', 'diagnostic_road_gap'], ['closedloopStaticAlphaT', 'diagnostic_static_alphaT']],
    image(id, im, state, columns) {
      if (id === 'closedloopRoadGap' && !$('diagnosticsGapHighlight').checked) {
        const i = columns.findIndex(([canvas]) => canvas === 'closedloopNewDelete');
        return state.pictures?.[i] || null;
      }
      return im;
    },
    update(row, state) {
      if (!api) return;
      const stats = row.diagnostic_stats, highlight = $('diagnosticsGapHighlight').checked;
      const total = stats.domain_pixels;
      $('diagnosticsGapStats').textContent = `帧 ${row.raw} · ${total ? '判定域 ' + number(total) + ' px' : '无道路恢复域'} · 支撑缺口 ${part(stats.gap_pixels, total)} · 薄覆盖 ${part(stats.thin_pixels, total)} · 充分支撑低贡献 ${part(stats.occluded_pixels, total)}${highlight ? '' : ' · 高亮已关闭：新删除原图'}`;
      $('diagnosticsStaticStats').textContent = `帧 ${row.raw} · 道路判定域平均静态 αT ${total ? Number(stats.static_alphaT_domain_mean).toFixed(4) + ' · 判定域 ' + number(total) + ' px' : '— · 无道路恢复域'}`;
      for (const [id] of this.columns) {
        const canvas = $(id), i = api.columns.findIndex(([key]) => key === id);
        canvas.dataset.raw = String(row.raw);
        canvas.dataset.loaded = String(!!state.pictures?.[i]);
        canvas.dataset.view = state.view;
        canvas.dataset.pixel = state.pixel.join(',');
      }
      $('contribution-diagnostics').dataset.raw = String(row.raw);
    },
    bind(value) {
      api = value;
      $('diagnosticsVisible').addEventListener('change', event => {
        $('diagnostic-panels').hidden = !event.target.checked;
        if (event.target.checked) api.draw();
      });
      $('diagnosticsGapHighlight').addEventListener('change', () => api.draw());
      $('diagnosticsJump').addEventListener('click', () => {
        $('contribution-diagnostics').scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    }
  };
})();
"""


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_delivery(manifest_path: Path) -> tuple[dict, list[tuple[Path, Path]]]:
    if not manifest_path.is_file():
        raise ValueError(f"Complete local manifest is required first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != SCHEMA or manifest.get("frame_count") != 199:
        raise ValueError("Expected scene065_contribution_diagnostics.v1 with 199 frames")
    frames = manifest.get("frames", [])
    if sorted(row.get("raw") for row in frames) != RAW_IDS:
        raise ValueError("Delivery must contain each raw frame 0..198 exactly once")
    stop_values = manifest.get("legend", {}).get("static_alphaT", {}).get("stops")
    if stop_values is not None:
        normalized = [[float(item[0]), str(item[1]).lower()] for item in stop_values]
        if normalized != STOPS:
            raise ValueError(f"Static colormap differs from the fixed UI legend: {normalized}")
    assets = []
    base = manifest_path.parent.resolve()
    for row in frames:
        stats = row.get("stats", {})
        for name in ["domain_pixels", "gap_pixels", "thin_pixels", "occluded_pixels"]:
            if type(stats.get(name)) is not int or stats[name] < 0:
                raise ValueError(f"Frame {row['raw']} invalid {name}")
        total = stats["domain_pixels"]
        if total > 480 * 320 or sum(stats[k] for k in ["gap_pixels", "thin_pixels", "occluded_pixels"]) > total:
            raise ValueError(f"Frame {row['raw']} diagnostic counts exceed domain")
        mean = stats.get("static_alphaT_domain_mean")
        if mean is None and total == 0:
            pass
        elif not isinstance(mean, (int, float)) or not math.isfinite(mean) or not 0 <= mean <= 1.000001:
            raise ValueError(f"Frame {row['raw']} invalid static alphaT mean")
        for kind in ["road_gap", "static_alphaT"]:
            item = row.get(kind, {})
            relative = Path(item.get("path", ""))
            source = (base / relative).resolve()
            if relative.is_absolute() or not relative.parts or not source.is_relative_to(base):
                raise ValueError(f"Unsafe local asset path: {relative}")
            if not source.is_file() or digest(source) != item.get("sha256"):
                raise ValueError(f"Missing image or SHA256 mismatch: {source}")
            with Image.open(source) as image:
                if image.size != (480, 320) or image.format not in {"PNG", "WEBP"}:
                    raise ValueError(f"Expected 480x320 PNG/WebP: {source}")
                image.load()
                if image.format == "WEBP" and b"VP8L" not in source.read_bytes()[:128]:
                    encoding = manifest.get("preview_encoding", {}).get(kind, "")
                    if kind != "road_gap" or "WebP" not in encoding or not manifest.get("preview_encoding", {}).get("original_images_and_float_kept_remote"):
                        raise ValueError(f"Lossy visual previews require explicit encoding and authoritative-data provenance: {source}")
            assets.append((source, relative))
    return manifest, assets


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError(f"UI source changed; expected exactly one anchor: {before[:100]!r}")
    return text.replace(before, after, 1)


def marked(text: str, label: str, content: str, before: str) -> str:
    start, end = f"<!-- {label}:begin -->", f"<!-- {label}:end -->"
    block = start + "\n" + content + "\n" + end
    pattern = re.escape(start) + r"[\s\S]*?" + re.escape(end)
    if start in text:
        text, count = re.subn(pattern, lambda _: block, text)
        if count != 1:
            raise ValueError(f"Duplicate extension marker: {label}")
        return text
    return replace_once(text, before, block + "\n" + before)


def patch_viewer(source: str) -> str:
    if "// contribution-diagnostics:columns" not in source:
        pattern = r"(  const columns = \[.*?\])(;)"
        source, count = re.subn(pattern, lambda m: m[1] + ".concat(window.ROAD_DIAGNOSTICS?.columns || [])" + m[2] + " // contribution-diagnostics:columns", source, count=1)
        if count != 1:
            raise ValueError("Cannot find the existing shared canvas column list")
    original = "    if (S.pictures) columns.forEach(([id], i) => S.pictures[i] ? paint($(id), S.pictures[i]) : blank(id, S.errors[i]));"
    extended = """    // contribution-diagnostics:draw; use the existing viewport/paint for every panel.
    if (S.pictures) columns.forEach(([id], i) => {
      const im = window.ROAD_DIAGNOSTICS ? window.ROAD_DIAGNOSTICS.image(id, S.pictures[i], S, columns) : S.pictures[i];
      im ? paint($(id), im) : blank(id, S.errors[i]);
    });"""
    if "// contribution-diagnostics:draw" not in source:
        source = replace_once(source, original, extended)
    pixel_line = "    $('pixel').textContent = `选中像素 (${S.pixel.join(', ')}) · 同帧网格 ${D.width}×${D.height}`;"
    if "// contribution-diagnostics:stats" not in source:
        source = replace_once(source, pixel_line, pixel_line + "\n    window.ROAD_DIAGNOSTICS?.update(F.get(S.raw), S); // contribution-diagnostics:stats")
    if "// contribution-diagnostics:bind" not in source:
        source = replace_once(source, "  range();\n  const hash =", "  window.ROAD_DIAGNOSTICS?.bind({ data: D, state: S, draw, render, columns }); // contribution-diagnostics:bind\n  range();\n  const hash =")
    return source


def patch_platform(text: str) -> str:
    if not all(f'id="{item}"' in text for item in BASE_COLUMNS):
        raise ValueError("Expected the published original six-panel scene065 viewer")
    text = marked(text, "contribution-diagnostics-style", "<style>\n" + STYLE + "</style>", "</head>")
    text = marked(text, "contribution-diagnostics-panels", PANELS, '<p class="muted" id="pixel">')
    if 'id="diagnosticsJump"' not in text:
        text = replace_once(text, '<p class="muted" id="closedloop-outcome">', '<button id="diagnosticsJump" aria-controls="contribution-diagnostics">贡献诊断 ↓</button><p class="muted" id="closedloop-outcome">')
    script = '<script src="diagnostics.js"></script>'
    if script not in text:
        text = replace_once(text, '<script src="viewer.js"></script>', script + '\n<script src="viewer.js"></script>')
    return text


def patch_data(source: str, manifest: dict) -> str:
    match = re.fullmatch(r"\s*window\.ROAD_MULTIVIEW_DATA\s*=\s*([\s\S]+?)\s*;?\s*", source)
    if not match:
        raise ValueError("Expected data.js to contain the viewer JSON assignment")
    data = json.loads(match.group(1))
    if [row["raw"] for row in data["frames"]] != RAW_IDS or (data["width"], data["height"]) != (480, 320):
        raise ValueError("Base scene065 data must contain 199 frames at 480x320")
    frames = {row["raw"]: row for row in manifest["frames"]}
    for row in data["frames"]:
        diagnostic = frames[row["raw"]]
        for key in ["road_gap", "static_alphaT"]:
            row["diagnostic_" + key] = "diagnostic_delivery/" + diagnostic[key]["path"].replace("\\", "/")
        row["diagnostic_stats"] = diagnostic["stats"]
    data["diagnostics"] = {"schema": SCHEMA, "manifest": "diagnostic_delivery/manifest.json", "legend": manifest.get("legend", {}), "domain": DOMAIN,
                           "preview_encoding": manifest.get("preview_encoding", {}), "counts_source": "authoritative float diagnostic data, not preview pixels"}
    return "window.ROAD_MULTIVIEW_DATA=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"


def report_legend() -> str:
    return '<div class="diagnostic-report-legend"><p><b>道路支撑与成像贡献</b>　<span class="swatch" style="background:#ff4d67"></span>支撑缺口 α &lt; 0.05　<span class="swatch" style="background:#ffba49"></span>薄覆盖 0.05 ≤ α ≤ 0.9　<span class="swatch" style="background:#39c6e6"></span>支撑充分、成像贡献低：α &gt; 0.9 且道路 αT &lt; 0.5。</p><p>' + DOMAIN + ' 支撑缺口不等于已证实的三维几何缺失；低贡献也可能来自正常前景遮挡。</p><p><b>静态可见贡献</b>　同一深度排序下 Σ<sub>i ∈ 静态</sub> α<sub>i</sub>T<sub>i</sub>，含未归属静态点，覆盖全画面，所有帧固定 0–1；0 表示无静态贡献，道路或动态层也可能接管该像素，不能据此判为缺口。</p><div class="diagnostic-report-scale"><div class="bar"></div><div class="ticks"><span>0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1</span></div></div><p>逐帧像素计数与同步定位见完整平台。<a href="assets/diagnostics/manifest.json">199 帧诊断清单</a></p><details class="diagnostic-dataflow"><summary>诊断数据流与数据来源</summary>' + FLOW + '<p>道路 α 来自道路单独渲染；αT 来自完整场景同一深度排序。计数及均值来自浮点数据。支撑诊断为 WebP 轻预览，静态热图为固定 256 级色标 PNG；原图与浮点数据已保留。</p></details></div>'


def patch_report(text: str, manifest: dict) -> tuple[str, list[tuple[str, str]]]:
    # Keep the original report markup/quality verdict byte-for-byte where possible.
    cards = BeautifulSoup(text, "html.parser").select('.frame[data-raw][data-previews]')
    if len(cards) != 7:
        raise ValueError("Expected seven existing report keyframe cards")
    by_raw = {row["raw"]: row for row in manifest["frames"]}
    copies = []
    for card in cards:
        raw = int(card["data-raw"])
        previews = json.loads(card["data-previews"])
        for kind in ["road_gap", "static_alphaT"]:
            relative = by_raw[raw][kind]["path"].replace("\\", "/")
            destination = f"assets/diagnostics/{raw:06d}_{kind}{Path(relative).suffix.lower()}"
            previews[kind] = destination
            copies.append((relative, destination))
        pattern = r'(<article\b[^>]*\bdata-raw="' + str(raw) + r'"[^>]*\bdata-previews=")([^"]*)(")'
        text, count = re.subn(pattern, lambda m: m[1] + escape(json.dumps(previews, ensure_ascii=False), quote=True) + m[3], text, count=1)
        if count != 1:
            raise ValueError(f"Cannot update keyframe metadata for {raw}")
    buttons = '<button data-view="road_gap" aria-pressed="false">道路支撑诊断</button><button data-view="static_alphaT" aria-pressed="false">静态可见贡献 αT</button>'
    if 'data-view="road_gap"' not in text:
        toolbar = re.search(r'(<div class="toolbar" aria-label="关键帧视图切换">)([\s\S]*?)(</div>)', text)
        if not toolbar:
            raise ValueError("Cannot find report keyframe view toolbar")
        text = text[:toolbar.start()] + toolbar.group(1) + toolbar.group(2) + buttons + toolbar.group(3) + text[toolbar.end():]
    names_old = "const names={old_delete:'旧删除',new_delete:'新删除',road_only:'道路单独渲染',static_contribution:'静态层 RGB 贡献'};"
    names_new = "const names={old_delete:'旧删除',new_delete:'新删除',road_only:'道路单独渲染',static_contribution:'静态层 RGB 贡献',road_gap:'道路支撑诊断',static_alphaT:'静态可见贡献 αT'};"
    if names_new not in text:
        text = replace_once(text, names_old, names_new)
    text = marked(text, "contribution-diagnostics-report-style", "<style>\n" + REPORT_STYLE + "</style>", "</head>")
    text = marked(text, "contribution-diagnostics-report-legend", report_legend(), '<div class="frames">')
    return text, copies


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=WORK / "diagnostic_delivery/manifest.json")
    parser.add_argument("--platform", type=Path, default=PLATFORM)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--apply", action="store_true", help="publish after all validation passes")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    platform, report = args.platform.resolve(), args.report.resolve()
    manifest, assets = load_delivery(manifest_path)
    files = {platform / "index.html": patch_platform((platform / "index.html").read_text(encoding="utf-8")),
             platform / "data.js": patch_data((platform / "data.js").read_text(encoding="utf-8"), manifest),
             platform / "viewer.js": patch_viewer((platform / "viewer.js").read_text(encoding="utf-8")),
             platform / "diagnostics.js": PLUGIN_JS}
    report_text, keyframe_copies = patch_report(report.read_text(encoding="utf-8"), manifest)
    files[report] = report_text
    result = {"schema": "scene065_contribution_diagnostics_ui.v1", "status": "validated", "apply": args.apply,
              "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
              "frames": 199, "validated_images": len(assets), "report_keyframe_images": len(keyframe_copies),
              "platform": str(platform), "report": str(report),
              "changed_files": [str(path) for path, text in files.items() if not path.is_file() or path.read_text(encoding="utf-8") != text]}
    if not args.apply:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    # Preserve first pre-extension originals even when this command is repeated.
    backups = []
    for path in files:
        if path.is_file():
            backup = path.with_name(path.name + ".before_contribution_diagnostics_20261116.bak")
            if not backup.exists():
                shutil.copy2(path, backup)
            backups.append({"path": str(backup), "sha256": digest(backup)})
    # Publish assets before page references, and verify the actual local copies.
    for source, relative in assets:
        destination = platform / "diagnostic_delivery" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or digest(source) != digest(destination):
            shutil.copy2(source, destination)
        if digest(source) != digest(destination):
            raise ValueError(f"Copied asset mismatch: {destination}")
    atomic_write(platform / "diagnostic_delivery/manifest.json", manifest_path.read_bytes())
    for source, destination in keyframe_copies:
        target = report.parent / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path.parent / source, target)
    # A rebased manifest lets each report diagnostic image resolve directly.
    report_manifest = json.loads(json.dumps(manifest))
    for row in report_manifest["frames"]:
        for kind in ["road_gap", "static_alphaT"]:
            actual = platform / "diagnostic_delivery" / row[kind]["path"]
            row[kind]["path"] = os.path.relpath(actual, report.parent / "assets/diagnostics").replace("\\", "/")
    atomic_write(report.parent / "assets/diagnostics/manifest.json", (json.dumps(report_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    for path, text in files.items():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            atomic_write(path, text.encode("utf-8"))
    result.update(status="published", backups=backups, published_at=datetime.now(timezone.utc).isoformat(),
                  output_sha256={str(path): digest(path) for path in files})
    receipt = WORK / "diagnostic_ui_receipt.json"
    atomic_write(receipt, (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

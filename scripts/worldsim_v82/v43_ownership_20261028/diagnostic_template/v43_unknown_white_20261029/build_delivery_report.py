"""Build a factual GPU delivery report from published native evidence and QA.

No renderer, SSH, model loading or inference is invoked. The contact sheet
pastes the twelve original raw113 RGB images at their native pixel dimensions.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CONTEXT = "14705303724557273004_3105_000_3125_000"
GROUPS = {
    "v43_actor_10000": ("10K 重建", "Actor10K reconstruction"),
    "v43_motion_no_edit": ("无编辑", "No edit"),
    "v43_motion_delete_target": ("删除目标车", "Delete target"),
    "v43_motion_translate_lateral_1m": ("横移 1 m", "Lateral shift 1 m"),
    "v43_motion_turn_left15": ("左转 15°", "Turn left 15 deg"),
    "v43_shift_3m": ("横移 3 m", "Lateral shift 3 m"),
}
CHECKS = ("alpha_identical", "radii_identical", "depth_sum_identical", "extra_identical",
          "baseline_saved_png_pixels_identical", "non_color_inputs_identical", "known_row_SH_identical",
          "row_order_unchanged", "native_environment_unchanged", "frozen_model_unchanged")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_file(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    require(path.is_relative_to(root), "Source path escapes published scene")
    return path


def link(path, output):
    return quote(Path(os.path.relpath(Path(path).resolve(), Path(output).resolve())).as_posix(), safe="/")


def font(size, bold=False):
    candidates = (["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
                  if bold else ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def architecture():
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="300" viewBox="0 0 1180 300" role="img" aria-label="Unknown SH recolor native rendering components">
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0 0 L8 3 L0 6 Z" fill="#48627a"/></marker></defs>
<rect width="1180" height="300" fill="#f7fafc"/>
<g font-family="Microsoft YaHei,Arial,sans-serif" font-size="18" text-anchor="middle" fill="#18334a">
<g stroke="#8aa6ba" stroke-width="1.5" fill="#ffffff"><rect x="20" y="30" width="220" height="90" rx="10"/><rect x="20" y="170" width="220" height="90" rx="10"/><rect x="290" y="85" width="230" height="120" rx="10"/><rect x="570" y="85" width="250" height="120" rx="10"/><rect x="870" y="85" width="290" height="120" rx="10"/></g>
<text x="130" y="63">冻结模型 + 当前相机</text><text x="130" y="91" font-size="15">static30K / Actor10K / 编辑</text>
<text x="130" y="203">同帧 R2 unknown IDs</text><text x="130" y="231" font-size="15">static 集合 + Actor 语义/轮廓</text>
<text x="405" y="119">只改 unknown SH 颜色</text><text x="405" y="151" font-size="15">白色材质；known 保留原色</text><text x="405" y="180" font-size="15">几何 / opacity / 行顺序不变</text>
<text x="695" y="121">原生 CUDA 光栅器</text><text x="695" y="153" font-size="15">当前编辑下重新计算可见贡献</text><text x="695" y="182" font-size="15">原生环境继续透明合成</text>
<text x="1015" y="121">同帧 RGB 对照 + 真实 3D</text><text x="1015" y="153" font-size="15">baseline / unknown-white</text><text x="1015" y="182" font-size="15">float、receipt 与独立 QA</text>
</g><g stroke="#48627a" stroke-width="2" fill="none" marker-end="url(#arrow)"><path d="M240 75 H265 V120 H290"/><path d="M240 215 H265 V170 H290"/><path d="M520 145 H570"/><path d="M820 145 H870"/></g>
</svg>'''


def build(manifest_path, qa_paths, root=ROOT, output=None):
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    output = Path(output).resolve() if output else root / "outputs/V43_Ownership_20261027/gpu_v43/unknown_white_20261030"
    scene = root / "outputs/GS_Scene_Diagnostics_20261027" / CONTEXT
    require(manifest_path == (scene / "unknown_white_manifest.json").resolve(), "Report must read the real published scene manifest")
    manifest = read(manifest_path)
    require(manifest.get("schema") == "v43-native-unknown-white-v1" and manifest.get("context") == CONTEXT, "Unexpected native manifest")
    require(manifest.get("status") in ("complete", "partial") and manifest.get("gpu_executed") is True, "Native GPU delivery has not completed")
    require(manifest.get("final_frozen_state_verified") is True and manifest.get("publication", {}).get("validated_on_CPU") is True, "Unpublished or unvalidated native delivery")
    require(manifest.get("training_updates") == 0, "This report is scoped to read-only diagnostic adaptation")
    manifest_hash = sha(manifest_path)
    verified = sum(len(manifest["groups"][gid]["frames"]) for gid in GROUPS)
    require(verified == manifest.get("verified_group_frame_count") and 0 < verified <= 204, "Published count differs")
    require((manifest["status"] == "complete") == (verified == 204), "Completion/count mismatch")
    qa_records, bound_native_qa = [], False
    for qa_path in qa_paths:
        qa_path = Path(qa_path).resolve()
        qa = read(qa_path)
        require(qa.get("passed") is True, "QA did not pass: " + str(qa_path))
        require(qa.get("syntheticImageFixturesOnly") is not True and qa.get("synthetic_ui_fixture") is not True, "Synthetic UI fixtures cannot prove native delivery")
        if qa.get("schema") == "v43-real-unknown-white-canvas-qa-v1":
            require(qa.get("manifestSha256") == manifest_hash and qa.get("verifiedEntries") == verified, "Native QA is bound to a different published manifest")
            require(qa.get("actualCanvas") is True and qa.get("images") == "actual published native PNG bytes; no synthetic fixtures", "QA does not verify actual native images")
            bound_native_qa = True
        qa_records.append(dict(path=str(qa_path), sha256=sha(qa_path), schema=qa.get("schema"),
                               passed=True, actualCanvas=qa.get("actualCanvas"), browserQA=qa.get("browserQA"),
                               images=qa.get("images"), manifestSha256=qa.get("manifestSha256"),
                               verifiedEntries=qa.get("verifiedEntries"), assetFilesChecked=qa.get("assetFilesChecked"),
                               allPNGJsExactBytes=qa.get("allPNGJsExactBytes"),
                               allNativeUnknownIDsMatchCurrentJS=qa.get("allNativeUnknownIDsMatchCurrentJS")))
    require(bound_native_qa, "Need the real Canvas QA bound to this published manifest")

    rows, pairs, reference_errors = [], [], []
    for gid, (chinese, english) in GROUPS.items():
        group = manifest["groups"][gid]
        raws = sorted(map(int, group["frames"]))
        for raw in raws:
            entry = group["frames"][str(raw)]
            require(entry.get("status") == "verified" and entry.get("group_id") == gid and entry.get("raw") == raw, "Invalid published frame")
            require(all(entry.get("checks", {}).get(k) is True for k in CHECKS), "Native invariant missing from published entry")
            error = entry["checks"].get("actual_native_reference_max_error")
            require(isinstance(error, (int, float)) and math.isfinite(error) and 0 <= error <= 2e-6, "Native reference mismatch")
            reference_errors.append(error)
        require("113" in group["frames"], "Missing real raw113 pair for " + gid)
        entry = group["frames"]["113"]
        receipt_path = relative_file(scene, entry["receipt"])
        require(sha(receipt_path) == entry["receipt_sha256"], "Published receipt changed")
        receipt = read(receipt_path)
        images = []
        for channel in ("baseline", "unknownwhite"):
            path = relative_file(scene, entry["images"][channel])
            require(sha(path) == receipt["files"][channel + ".png"]["sha256"], "Published PNG changed")
            with Image.open(path) as image:
                require(image.mode == "RGB" and image.size == (480, 320), "Unexpected native image shape")
                images.append(image.copy())
        pairs.append((english, images))
        rows.append(dict(group_id=gid, title=group.get("title", chinese), short_title=chinese,
                         verified_frames=len(raws), raw_range=[min(raws), max(raws)], status=group["status"],
                         example_raw=113, example_images=entry["images"], example_receipt=entry["receipt"],
                         static_unknown_rows=receipt["static_unknown_rows"], actor_unknown_rows=receipt["actor_unknown_rows"]))

    # Native pixels are pasted unchanged; headings occupy separate canvas space.
    width, top, stride = 1020, 80, 372
    sheet = Image.new("RGB", (width, top + 6 * stride + 20), "#f4f7fa")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 15), "Native RGB / unknown Gaussian SH set to white", font=font(24, True), fill="#18334a")
    draw.text((20, 49), "Baseline native RGB", font=font(18), fill="#304d64")
    draw.text((520, 49), "Unknown-white native RGB", font=font(18), fill="#304d64")
    boxes = []
    for i, (title, images) in enumerate(pairs):
        y = top + i * stride
        draw.text((20, y + 6), f"{i + 1}. {title}  |  raw113", font=font(20, True), fill="#18334a")
        for j, image in enumerate(images):
            x, image_y = 20 + j * 500, y + 40
            sheet.paste(image, (x, image_y))
            box = (x, image_y, x + 480, image_y + 320)
            require(sheet.crop(box).tobytes() == image.tobytes(), "Contact sheet altered source pixels")
            boxes.append((box, image.tobytes()))
    output.mkdir(parents=True, exist_ok=True)
    sheet.save(output / "raw113_pairs.png")
    with Image.open(output / "raw113_pairs.png") as reread:
        require(all(reread.crop(box).tobytes() == pixels for box, pixels in boxes), "Saved montage changed source pixels")
    (output / "architecture.svg").write_text(architecture(), encoding="utf8")
    summary = dict(schema="v43-unknown-white-delivery-report-v1", generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   context=CONTEXT, status=manifest["status"], verified_group_frames=verified, required_group_frames=204,
                   full_render_wall_seconds=manifest.get("wall_seconds"), training_updates=0,
                   classification=manifest["classification"], source_manifest=str(manifest_path), source_manifest_sha256=manifest_hash,
                   native_reference_max_error=max(reference_errors), invariant_checks_all_verified=list(CHECKS),
                   groups=rows, QA=qa_records, human_verdict_provided="human_verdict" in manifest,
                   human_verdict=manifest.get("human_verdict"), research_scope="diagnostic adaptation; no new algorithm or recall-improvement claim",
                   scope_exclusions=["Actor1K/5K need checkpoint-matched attribution", "V1 original dynamic row namespace needs independent adaptation", "Scenes2/3 not migrated to this inherited-R2 unknown-white delivery"],
                   contact_sheet=dict(path="raw113_pairs.png", source_rgb_images=12, native_image_size=[480, 320],
                                      resampling=False, original_pixels_verified_after_save=True))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    elapsed = manifest.get("wall_seconds")
    duration = f"{elapsed:.3f} 秒" if isinstance(elapsed, (int, float)) else "未在 manifest 记录"
    radius, fraction = manifest["classification"]["radius"], manifest["classification"]["fraction"]
    state = "完整交付" if manifest["status"] == "complete" else "部分交付"
    table = "".join(f'<tr><td><a href="{link(scene / "index.html", output)}#workspace?group={quote(row["group_id"])}&amp;raw=113&amp;channel=rgb">{html.escape(row["short_title"])}</a></td><td>{row["verified_frames"]}</td><td>{row["raw_range"][0]:03d}–{row["raw_range"][1]:03d}</td><td>{row["static_unknown_rows"]:,} / {row["actor_unknown_rows"]:,}</td></tr>' for row in rows)
    qa_list = "".join(f'<li><a href="{link(q["path"], output)}">{html.escape(q["schema"] or Path(q["path"]).name)}</a>：通过；actualCanvas={html.escape(str(q["actualCanvas"]))}，browserQA={html.escape(str(q["browserQA"]))}。</li>' for q in qa_records)
    verdict = html.escape(json.dumps(manifest["human_verdict"], ensure_ascii=False)) if manifest.get("human_verdict") is not None else "未提供人工验收结论。"
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V43 同帧 unknown 白材质 GPU 交付</title><style>body{{margin:0;background:#f4f7fa;color:#18334a;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1100px;margin:40px auto;padding:0 24px 60px}}h1{{font-size:30px;line-height:1.3}}h2{{font-size:22px;margin-top:32px}}.lead{{font-size:20px}}.card{{background:white;border:1px solid #dbe5ed;border-radius:12px;padding:22px;margin:18px 0}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #dbe5ed;text-align:left}}a{{color:#176b9b}}.muted{{color:#506c80;font-size:14px}}code{{font-size:.9em}}</style><main>
<h1>V43 同帧 unknown 白材质 GPU 交付</h1><p class="lead">{state}：{verified}/204 个 group/frame。无新增训练；本次工作为诊断适配。</p>
<p>原生渲染耗时：{duration}。基准 RGB 与原保存图逐像素一致；所有已交付帧的 alpha、radii、depth、辅助特征及非颜色输入通过原生一致性检查。</p>
<div class="card"><img src="architecture.svg" alt="冻结模型和同帧unknown IDs进入仅改SH颜色模块，再由原生光栅器与环境合成同帧结果"></div>
<h2>改动与检查</h2><p>只把 unknown 高斯的 SH 材质设为白色：DC=(1−0.5)/C0，高阶系数清零。known 高斯保留原生颜色。Actor 行按当前物理编辑变换；static unknown 保留该时刻原生位置，不随车辆搬动。几何、尺度、opacity、行顺序、遮挡和原生环境保持本组基准值，像素贡献由 CUDA 原生光栅器重新计算。</p>
<p>{radius}px / {fraction:g} 归属来自冻结 Actor10K 原观测语义/轮廓证据与同帧 R2 static 集合。它没有新增训练，也没有证明分类召回、精度或背景补全改善。白色材质仍受原透明合成影响，相关像素不保证纯白。</p>
<div class="card"><table><thead><tr><th>组（打开交互页）</th><th>已核验帧</th><th>raw范围</th><th>raw113 unknown 行：static / actor</th></tr></thead><tbody>{table}</tbody></table></div>
<h2>raw113 同帧成对排图</h2><p>左为本组原生 RGB，右为 unknown 白材质原生 RGB。12 张原图按480×320原尺寸粘贴，标题在图像区域之外；保存后已逐像素检查排图没有修改原图。</p>
<a href="raw113_pairs.png"><img src="raw113_pairs.png" alt="六组raw113原生基准和unknown白材质成对对照"></a>
<h2>适配边界与验收</h2><p>1K、5K 仍需各自 checkpoint 的贡献归因；V1 原生通路使用不同动态行命名空间，需要独立适配。场景2/3尚未迁移到本次继承 R2 与白材质交付。</p><ul>{qa_list}</ul><p>人工结论：{verdict}</p>
<p class="muted">数据来源：<a href="{link(manifest_path, output)}">真实已发布 manifest</a> · <a href="summary.json">结构化摘要</a>。本报告从现有真实证据生成，不重新执行渲染或推断人类验收。</p></main></html>'''
    (output / "index.html").write_text(page, encoding="utf8")
    return dict(status=manifest["status"], verified_group_frames=verified, output=str(output),
                contact_sheet_original_pixels_preserved=True, GPU_executed_by_report=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Real scene1 unknown_white_manifest.json after publication")
    parser.add_argument("--qa", type=Path, required=True, action="append", help="Passed actual-image QA; may be repeated")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.qa, args.root, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()

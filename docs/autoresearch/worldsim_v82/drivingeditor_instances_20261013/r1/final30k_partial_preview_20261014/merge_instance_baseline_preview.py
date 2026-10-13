"""将真实 GMC 两实例 30k baseline 接入唯一既有 HTML；不等待编辑出口。

默认只校验：--evaluation-manifest <outputs内真实manifest>。
资产完整后加 --apply；--validation-output 使用新文件名保留前次receipt。
完整98帧与独立partial schema分别校验；partial只展示已真实导出的观察帧。
没有after或fixed时不放替代图，不把partial变成完整评估。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
module_spec = importlib.util.spec_from_file_location("instance_viewer_merge", HERE / "drivingeditor_instances" / "merge_instance_viewer.py")
lib = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(lib)
MARKER = lib.BASELINE_MARKER
STYLE = lib.STYLE.replace("gmc-instance-style", "gmc-preview-style").replace("gmc-instance-panel", "gmc-preview-panel")
DIAGRAM = re.search(r"<svg.*?</svg>", lib.SECTION, re.S).group(0).replace("gmc-component-arrow", "gmc-preview-component-arrow")
DIAGRAM = DIAGRAM.replace("与ID编辑", "与固定相机诊断").replace("49 观察帧 + ID 编辑", "49 真实观察帧").replace("固定相机无 GT 诊断", "固定相机 49 时间")

SECTION = """<section id="gmc-preview-panel" aria-labelledby="gmc-preview-title" hidden>
<div class="eyebrow">GMC · TWO RIGID INSTANCES · ACTUAL 30K BASELINE PREVIEW</div><h2 id="gmc-preview-title">GMC 两个 ID：30k 重建预览</h2>
<p>GMC77 皮卡与 Jeep82 是两个独立刚体实例；Corolla 与其他停车保留原静态背景。原三场景证据仍由①②按钮查看。</p>
<div class="notice fixed"><strong>两车编辑正在导出，尚未提供 after。</strong> 本区只显示真实30k编辑前重建；不以before代替after，也不使用300步短测图。</div>
__COMPONENTS__
<div class="gmc-row"><button id="gmc-preview-observed" aria-pressed="true">真实49帧 · 重建</button><button id="gmc-preview-fixed" aria-pressed="false">固定相机49时间 · 无 GT</button></div>
<div class="notice" id="gmc-preview-notice"></div><div class="gmc-controls"><div class="gmc-row">
<div class="buttons"><button id="gmc-preview-prev" aria-label="GMC预览上一帧">←</button><button id="gmc-preview-play">播放</button><button id="gmc-preview-next" aria-label="GMC预览下一帧">→</button></div>
<div class="gmc-frame"><strong id="gmc-preview-frame-title"></strong><div class="gmc-detail" id="gmc-preview-frame-detail"></div></div>
<label for="gmc-preview-actor">实例 ID</label><select id="gmc-preview-actor"><option value="0">GMC77 · 蓝色皮卡</option><option value="1">Jeep82 · 灰色 SUV</option></select>
<label for="gmc-preview-branch">分支</label><select id="gmc-preview-branch"><option value="paired">GT / 编辑前合成 / static</option><option value="visible_rgb">所选 ID · 全场景可见 RGB</option><option value="visible_alpha">所选 ID · 全场景可见 alpha</option></select>
<label for="gmc-preview-crop">视野</label><select id="gmc-preview-crop"><option value="full">完整原图</option><option value="roi">真实 body ROI</option></select>
</div><input id="gmc-preview-timeline" aria-label="GMC30k预览49帧进度" type="range" min="0" max="48" value="0"><div id="gmc-preview-crop-hint" class="gmc-note"></div></div>
<p class="gmc-note">稳定 raw track ID：<code id="gmc-preview-track-id"></code></p><div id="gmc-preview-gallery" class="gmc-grid"></div>
<div class="table-wrap"><table><thead><tr><th>当前帧诊断</th><th>实际保存值</th></tr></thead><tbody id="gmc-preview-metric-body"></tbody></table></div>
<p id="gmc-preview-evidence" class="gmc-note"></p><p class="gmc-note">49帧为训练观测重建，不是heldout。固定相机没有对应GT/PSNR。单ID可见RGB是所有几何共同排序后的alpha×T×颜色贡献，黑画布不补环境；alpha未预乘body mask、无逐图归一化。空body保留/null。人工质量评价：null；补景、残影、阴影与遮挡边界未由本接入工具验收。</p>
</section>""".replace("__COMPONENTS__", DIAGRAM)


def validate_partial(path, assets, *, fixture=False):
    """真实partial独立入口：相机GPU矩阵未落盘，仅验证实际native绑定。"""
    manifest = lib.read_json(path)
    lib.require(manifest.get("schema") == "drivingeditor_rigid_scene_partial_preview_v1", "partial schema不符")
    lib.require(manifest.get("status") == ("SYNTHETIC_PARTIAL_FIXTURE" if fixture else "PARTIAL_ACTUAL30K_OBSERVED_PREVIEW"), "partial实际状态不符")
    lib.require(bool(manifest.get("synthetic_fixture")) == fixture, "合成partial只能显式用于work测试")
    lib.require(manifest.get("instance_ids") == lib.EXPECTED_IDS, "partial必须保留两条真实GMC/Jeep track ID")
    checkpoint = manifest.get("checkpoint", {})
    lib.require(checkpoint.get("step") == 30000 and re.search(r"checkpoint_0*30000\.pt$", checkpoint.get("path", "")), "partial必须来自真实30k checkpoint")
    lib.require(manifest.get("spec", {}).get("scene_id", manifest.get("scene_id")) == "scene0008_023_071", "partial scene绑定不符")
    lib.require(manifest.get("human_quality_verdict") is None and manifest.get("spec", {}).get("human_quality_verdict") is None, "partial不能代填人工质量评价")
    contracts = manifest.get("contracts", {})
    for key in ("all_FP32_finite", "identity_and_native_size_verified", "GT_reference_PNG_verified", "real_body_mask_PNG_verified"):
        lib.require(contracts.get(key) is True, "缺少partial真实CPU验证: " + key)
    lib.require(contracts.get("model_and_optimizer_unmodified_after_evaluation") in (None, False), "partial不能冒充尚未完成的全流程readonly postproof")
    source = manifest.get("streams", {}).get("observed", {})
    rows = source.get("frames", [])
    count = len(rows)
    lib.require(source.get("GT_available") is True and 1 <= count <= 49, "partial必须含实际非空观察流")
    lib.require(contracts.get("observed_frames_verified") == count, "partial CPU验证帧数与实际文件行不符")
    fixed = manifest.get("streams", {}).get("fixed_camera", {})
    lib.require(not fixed.get("frames") and fixed.get("GT_available", False) is False
                and contracts.get("fixed_camera_frames") == 0 and contracts.get("edit_operations") == 0, "当前partial不得填未导出的fixed或after")
    lib.require([row.get("frame_index") for row in rows] == list(range(count))
                and [row.get("frame_id") for row in rows] == lib.FRAME_IDS[:count], "partial必须对应已验证prepared的连续真实前N帧")
    lib.require(lib.finite_tree(manifest), "partial metadata含非有限数")
    base = Path(path).resolve().parent
    result = []
    for index, row in enumerate(rows):
        lib.require(abs(row.get("model_time", float("inf")) - (-0.48 + index * 0.02)) < 1e-6, "partial真实模型时间不符")
        camera = row.get("camera", {})
        lib.require(camera.get("width") == 480 and camera.get("height") == 320, "partial仅声明实际native480x320")
        camera_id = row.get("camera_frame_id", camera.get("camera_frame_id"))
        lib.require(lib.raw_frame(camera_id) == lib.FRAME_IDS[index], "partial实际观察相机绑定不符")
        for key in ("R", "T", "world_view_transform", "projection_matrix", "full_proj_transform", "camera_center"):
            lib.require(camera.get(key) is None, "partial未独立保存GPU矩阵，source camera JSON只能作provenance")
        lib.require(row.get("all_FP32_finite") is True, "partial帧缺实际finite FP32验证")
        prefix = f"observed/{index:03d}_{lib.FRAME_IDS[index]}"
        lib.require(lib.relative_path(row.get("npz", "")).as_posix() == prefix + ".npz", "partial NPZ实际frame绑定不符")
        images = row.get("images", {})
        lib.require({"gt", "composite", "static_rgb", "dynamic_visible_alpha"}.issubset(images), "partial缺实际GT/composite/static/visible-alpha")
        lib.require(all(lib.relative_path(value).as_posix() == prefix + "_" + key + ".png" for key, value in images.items()), "partial PNG必须属于该实际frame")
        instances = row.get("instances", [])
        lib.require([item.get("index") for item in instances] == [0, 1] and [item.get("track_id") for item in instances] == lib.EXPECTED_IDS, "partial单ID行身份不符")
        item = json.loads(json.dumps(row))
        item.update(camera_frame_id=camera_id, images=lib.mapped_images(images, base, assets))
        item.setdefault("metrics", {})
        for actor, instance in enumerate(item["instances"]):
            names = instance.get("images", {})
            lib.require({"visible_rgb", "visible_alpha", "body_mask"}.issubset(names), "partial缺真实单ID可见RGB/alpha/body PNG")
            lib.require(all(lib.relative_path(value).as_posix() == prefix + f"_instance{actor:02d}_" + key + ".png" for key, value in names.items()), "partial单ID PNG绑定不符")
            with assets.Image.open(assets.resolve(base, names["body_mask"])) as image:
                lib.require(image.mode == "L", "partial body mask必须单通道")
                histogram = image.histogram()
            body_pixels = instance.get("metrics", {}).get("body_pixels", instance.get("body_pixels"))
            lib.require(histogram[0] + histogram[255] == 480 * 320 and histogram[255] == body_pixels, "partial真实mask二值/像素分母不符")
            instance.setdefault("metrics", {})["body_pixels"] = body_pixels
            instance["images"] = lib.mapped_images(names, base, assets)
        result.append(item)
    return manifest, {"observed": {"frames": result}, "fixed_camera": {"frames": []}}


def build_preview_patch(original, data, script_url):
    lib.require(lib.MARKER not in original and MARKER not in original, "已有完整编辑或预览区域，保留资产，不重复或降级覆盖")
    before = lib.legacy_blocks(original)
    newline = "\r\n" if "\r\n" in original else "\n"
    def block(name, contents):
        return (f"<!-- BEGIN {MARKER} {name} -->\n{contents}\n<!-- END {MARKER} {name} -->\n").replace("\n", newline)
    result = original.replace("</head>", block("STYLE", STYLE) + "</head>", 1)
    nav = re.search(r'<nav class="modes"[^>]*>.*?</nav>', result, re.S)
    lib.require(nav is not None, "原模式按钮锚点缺失")
    button = '<button id="gmc-preview-open" aria-controls="gmc-preview-panel" aria-pressed="false" hidden>③ GMC · 两 ID 30k 预览</button>'
    offset = nav.end() - len("</nav>")
    result = result[:offset] + block("BUTTON", button) + result[offset:]
    anchor = '<section class="controls">'
    lib.require(result.count(anchor) == 1, "旧controls插入锚点不唯一")
    section = SECTION
    if data.get("partial"):
        count = len(data["streams"]["observed"]["frames"])
        section = section.replace("两车编辑正在导出，尚未提供 after。", f"已导出{count}/49观察帧；GPU已让给其他任务，固定相机及两车编辑尚未导出，未提供 after。")
        section = section.replace('id="gmc-preview-fixed" aria-pressed="false"', 'id="gmc-preview-fixed" aria-pressed="false" disabled')
        section = section.replace("固定相机49时间 · 无 GT", "固定相机 · 尚未导出")
        section = section.replace("真实49帧 · 重建", f"已导出{count}/49帧 · 逐帧重建")
        section = section.replace("GMC30k预览49帧进度", f"GMC30k预览{count}帧进度")
        section = section.replace("49 真实观察帧", f"已导出{count}/49观察帧").replace("固定相机 49 时间", "fixed / after 未导出")
        section = section.replace("49帧为训练观测重建，不是heldout。", f"训练使用49个原始观察帧；当前只展示已导出的{count}帧，不是heldout。")
        section = section.replace('max="48" value="0"', f'max="{count-1}" value="0"')
    result = result.replace(anchor, block("SECTION", section) + anchor, 1)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    boot = f'<script id="gmc-preview-data" type="application/json">{payload}</script>\n<script src="{script_url}" defer></script>'
    result = result.replace("</body>", block("DATA_AND_SCRIPT", boot) + "</body>", 1)
    lib.require(lib.legacy_blocks(result) == before, "baseline预览改写了旧三场景data/script")
    restored, removed = lib.strip_baseline_preview(result)
    lib.require(removed and restored == original, "自家preview块不能逐字移除，拒绝接入")
    return result


def prepare(args):
    target = args.target.resolve()
    if args.allow_synthetic_fixture:
        lib.require(lib.within(target, lib.WORKSPACE / "work") and not lib.within(target, lib.OUTPUTS), "合成fixture不得进入outputs")
        if args.validation_output:
            lib.require(lib.within(args.validation_output, lib.WORKSPACE / "work") and not lib.within(args.validation_output, lib.OUTPUTS), "fixture receipt只能放work")
    else:
        lib.require(target == (lib.OUTPUTS / "DrivingEditor_final_report.html").resolve(), "生产只有原统一HTML入口")
    lib.require(target.is_file(), "原统一HTML尚不存在")
    original_bytes = target.read_bytes()
    original = original_bytes.decode("utf-8")
    lib.require(lib.MARKER not in original and MARKER not in original, "已有完整编辑/预览，拒绝重复覆盖")
    assets = lib.BrowserAssets(target.parent)
    partial = lib.read_json(args.evaluation_manifest).get("schema") == "drivingeditor_rigid_scene_partial_preview_v1"
    validator = validate_partial if partial else lib.validate_evaluation
    manifest, streams = validator(args.evaluation_manifest, assets, fixture=args.allow_synthetic_fixture)
    base = args.evaluation_manifest.resolve().parent
    for source, row in zip(manifest["streams"]["observed"]["frames"], streams["observed"]["frames"]):
        for source_instance, instance in zip(source["instances"], row["instances"]):
            mask_path = assets.resolve(base, source_instance["images"]["body_mask"])
            with assets.Image.open(mask_path) as image:
                box = image.getbbox()
            instance["bbox"] = ([max(0, box[0]-40), max(0, box[1]-40), min(480, box[2]+40), min(320, box[3]+40)] if box else None)
    source_js = HERE / "gmc_instance_baseline_preview.js"
    lib.require(source_js.is_file(), "preview专用JS不存在")
    destination = target.parent / "drivingeditor_GMC_instances_30k" / "gmc_instance_baseline_preview.js"
    lib.require(not destination.exists(), "保留已存在preview JS，不覆盖")
    script_url = "/".join(quote(part, safe="_-.") for part in destination.relative_to(target.parent).parts)
    data = {"schema": "drivingeditor_GMC_baseline_preview_data_v1", "synthetic_fixture": args.allow_synthetic_fixture,
            "checkpoint": manifest["checkpoint"], "instance_ids": lib.EXPECTED_IDS, "streams": streams,
            "units": manifest.get("units"), "normalization": manifest.get("normalization"),
            "observed_metrics": None if partial else manifest.get("observed_metrics"), "partial": partial,
            "editing_status": "not_exported_GPU_priority_yield" if partial else "pending_export_no_after",
            "human_quality_verdict": None, "evaluation_contracts": manifest["contracts"]}
    updated = build_preview_patch(original, data, script_url)
    if args.allow_synthetic_fixture:
        updated = updated.replace("ACTUAL 30K BASELINE PREVIEW", "SYNTHETIC FIXTURE · 非结果", 1)
        lib.require(lib.strip_baseline_preview(updated)[0] == original, "合成fixture标记破坏精确升级")
    receipt = {"schema": "drivingeditor_GMC_baseline_preview_merge_receipt_v1",
               "checked_utc": datetime.now(timezone.utc).isoformat(), "status": "BASELINE_PREVIEW_VALIDATED_NOT_APPLIED",
               "synthetic_fixture": args.allow_synthetic_fixture, "target": str(target),
               "evaluation_manifest": str(args.evaluation_manifest.resolve()), "checkpoint_step": 30000,
               "observed_frames": len(streams["observed"]["frames"]), "planned_observed_frames": 49,
               "fixed_camera_frames": len(streams["fixed_camera"]["frames"]), "partial_actual_preview": partial,
               "GPU_camera_matrices_serialized": not partial, "browser_PNGs_verified": len(assets.checked),
               "legacy_three_scene_data_script_and_evidence_exactly_preserved": True,
               "formal_upgrade_removes_own_preview_exactly": True, "after_assets_presented": False,
               "editing_status": data["editing_status"], "GUI_QA_performed": False, "human_quality_verdict": None,
               "photo_quality_admitted": False, "GPU_used": False, "new_production_HTML_entry_created": False}
    if partial:
        receipt["limitations"] = ["仅验证并展示实际已保存的前N观察帧；fixed和after未导出",
                                  "GPU相机矩阵未独立序列化，source camera JSON仅作provenance",
                                  "完整评估后的model/optimizer readonly postproof尚未完成；不认作完整98帧评估"]
    if args.validation_output:
        lib.require(not args.validation_output.exists(), "保留前次receipt，请指定新文件名")
    if args.apply:
        backups = HERE / "drivingeditor_instances" / "viewer_merge_backups"
        backups.mkdir(exist_ok=True, parents=True)
        backup = backups / ("baseline_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".original")
        backup.write_bytes(original_bytes)
        destination.parent.mkdir(exist_ok=True, parents=True)
        with destination.open("xb") as stream:
            stream.write(source_js.read_bytes())
        temporary = target.with_name(target.name + ".baseline_preview.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(updated.encode("utf-8"))
            lib.require(target.read_bytes() == original_bytes, "原统一HTML被并发修改，保留备份/JS，不覆盖")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        receipt.update(status="BASELINE_PREVIEW_APPLIED_TO_EXISTING_UNIFIED_ENTRY", backup=str(backup), script=str(destination))
    if args.validation_output:
        args.validation_output.parent.mkdir(exist_ok=True, parents=True)
        with args.validation_output.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return receipt


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument("--target", type=Path, default=lib.OUTPUTS / "DrivingEditor_final_report.html")
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-synthetic-fixture", action="store_true", help="只在work做明确合成fixture测试")
    args = parser.parse_args()
    try:
        receipt = prepare(args)
    except (ValueError, OSError) as error:
        print("预览尚未接入，保留原统一入口: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

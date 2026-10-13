"""将真实 GMC 两实例 30k 资产接入既有统一 HTML；默认仅校验。

生产用法（先把真实导出 PNG 复制到 outputs 的子目录）：
  python merge_instance_viewer.py --evaluation-manifest <scene_evaluation_manifest.json>
    --edit-summary <collect_edit_summary.json> --edit-root <six_operations_root>
    --validation-output <work/validation.json>
资产通过校验后，沿用输入参数加 --apply，并指定新的 validation-output；
修改既有 DrivingEditor_final_report.html，保留前一次校验 receipt。
不训练、不渲染、不创建生产 HTML 新入口；不把工程校验写成人工质量评价。
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
OUTPUTS = WORKSPACE / "outputs"
EXPECTED_IDS = ["NoQttdyd4C16Y_zkGhiFKg", "fXVhuN-RuJDvZwnkQ-0YSQ"]
FRAME_IDS = [f"0008{raw:03d}" for raw in range(23, 72)]
OPERATIONS = [f"actor{actor}_{operation}" for actor in range(2)
              for operation in ("delete", "translate", "rotate")]
MARKER = "GMC_INSTANCE_VIEWER_V1"
BASELINE_MARKER = "GMC_INSTANCE_BASELINE_PREVIEW_V1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"实际 manifest/receipt 尚不存在，保留统一 HTML: {path}")
    def invalid(value):
        raise ValueError(f"JSON 不接受非有限数: {path}: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid)


def within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def raw_frame(value):
    match = re.fullmatch(r"(\d{7})(?:_\d+)?", str(value))
    require(match is not None, f"实际相机 frame ID 格式不符: {value}")
    return match.group(1)


def relative_path(value):
    require(isinstance(value, str) and bool(value), "缺少实际资产相对路径")
    require(not any(c in value for c in "\\:?#%") and not any(ord(c) < 32 for c in value),
            f"资产路径不能为 URL/绝对路径/转义路径: {value}")
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(part not in ("..", ".") for part in path.parts),
            f"资产路径越界: {value}")
    return path


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


class BrowserAssets:
    """验证实际浏览器引用的 PNG，不宣称已做 GUI 或照片质量审核。"""
    def __init__(self, html_parent):
        from PIL import Image
        self.Image = Image
        self.root = Path(html_parent).resolve()
        self.checked = set()
        self.pixels = {}
        self.pair_checks = 0

    def resolve(self, base, value):
        path = (Path(base) / relative_path(value)).resolve()
        require(within(path, base) and within(path, self.root), f"实际浏览资产不在统一 HTML 目录内: {path}")
        require(path.is_file() and path.suffix.lower() == ".png", f"实际 PNG 尚未就绪: {path}")
        if path not in self.checked:
            with self.Image.open(path) as image:
                require(image.format == "PNG" and image.size == (480, 320), f"真实 PNG 原生尺寸必须为 480x320: {path}")
                image.verify()
            self.checked.add(path)
        return path

    def url(self, base, value):
        path = self.resolve(base, value)
        return "/".join(quote(part, safe="_-.") for part in path.relative_to(self.root).parts)

    def paired(self, first, second):
        def pixels(path):
            if path not in self.pixels:
                with self.Image.open(path) as image:
                    self.pixels[path] = image.convert("RGB").tobytes()
            return self.pixels[path]
        require(pixels(first) == pixels(second), f"当前 30k before/GT/static/visible-alpha 与评估 PNG 不配对: {first} / {second}")
        self.pair_checks += 1
        # 编辑 PNG 不常驻内存；基准最多保留约 70 MiB。
        if first != second:
            self.pixels.pop(first, None)


def mapped_images(images, base, assets):
    require(isinstance(images, dict), "缺少真实 images 字段")
    return {key: assets.url(base, value) for key, value in images.items()}


def validate_evaluation(path, assets, *, fixture=False):
    manifest = read_json(path)
    require(manifest.get("schema") == "drivingeditor_rigid_scene_evaluation_v1", "评估 manifest schema 不符")
    require(manifest.get("status") == ("SYNTHETIC_FIXTURE" if fixture else "SCENE_EVALUATION_EXPORTED"), "评估未完成，不能把 pending 当结果")
    require(bool(manifest.get("synthetic_fixture")) == fixture, "合成 fixture 只能在显式 fixture 模式使用")
    require(manifest.get("instance_ids") == EXPECTED_IDS, "必须使用 GMC77 / Jeep82 两条真实 raw track ID")
    checkpoint = manifest.get("checkpoint", {})
    require(checkpoint.get("step") == 30000 and re.search(r"checkpoint_0*30000\.pt$", checkpoint.get("path", "")), "统一入口仅接入实际 30k checkpoint")
    require(manifest.get("spec", {}).get("scene_id") == "scene0008_023_071", "此工具只接入 GMC scene0008_023_071")
    require(manifest.get("spec", {}).get("human_quality_verdict") is None, "本工具不能代填 human quality verdict")
    contracts = manifest.get("contracts", {})
    for key in ("all_FP32_finite", "identity_and_native_size_verified", "model_and_optimizer_restored_exactly",
                "model_and_optimizer_unmodified_after_evaluation", "fixed_camera_fields_except_timestamp_share_actual_objects",
                "per_ID_RGB_full_scene_composite_alpha_unchanged", "per_ID_RGB_identity_contributions_match_base_pass",
                "mask_metrics_use_joint_visible_contribution", "same49_train_GT_not_heldout"):
        require(contracts.get(key) is True, f"缺少实际评估合同: {key}")
    require(contracts.get("optimizer_updates") == 0 and contracts.get("observed_frames") == 49
            and contracts.get("fixed_camera_frames") == 49, "评估必须只读、两个流各完整49帧")
    require(finite_tree(manifest), "评估 metadata 含非有限数")
    streams = {}
    fixed_camera = None
    base = Path(path).resolve().parent
    for name in ("observed", "fixed_camera"):
        source = manifest.get("streams", {}).get(name, {})
        require(source.get("GT_available") is (name == "observed"), f"{name} 的 GT 角色不符")
        rows = source.get("frames", [])
        require(len(rows) == 49 and [row.get("frame_index") for row in rows] == list(range(49)), f"{name} 缺少完整49帧")
        require([row.get("frame_id") for row in rows] == FRAME_IDS, f"{name} 的真实raw23..71帧身份/顺序不符")
        result = []
        for index, row in enumerate(rows):
            require(math.isclose(row.get("model_time", float("inf")), -0.48 + index * 0.02, abs_tol=1e-6), "模型时间必须继承真实49帧归一化")
            camera = row.get("camera", {})
            require(camera.get("width") == 480 and camera.get("height") == 320, "实际相机 native size 不符")
            for field in ("R", "T", "world_view_transform", "projection_matrix", "full_proj_transform", "camera_center", "fx", "fy", "cx", "cy"):
                require(field in camera and camera[field] is not None, f"实际相机记录缺字段: {field}")
            require(row.get("all_FP32_finite") is True, "帧缺少真实finite FP32验证")
            images = row.get("images", {})
            require({"composite", "static_rgb", "dynamic_visible_alpha"}.issubset(images), "评估缺完整合成/静态/联合可见alpha PNG")
            prefix = f"{name}/{index:03d}_{FRAME_IDS[index]}"
            require(relative_path(row.get("npz", "")) == PurePosixPath(prefix + ".npz"), "评估实际NPZ来源frame/目录绑定不符")
            require(all(relative_path(value) == PurePosixPath(prefix + "_" + key + ".png")
                        for key, value in images.items()), "评估PNG不属于该实际相机/时间帧")
            ids = row.get("instances", [])
            require([item.get("index") for item in ids] == [0, 1] and [item.get("track_id") for item in ids] == EXPECTED_IDS, "逐帧两个 ID 顺序不符")
            item = copy.deepcopy(row)
            item["images"] = mapped_images(images, base, assets)
            for actor, instance in enumerate(item["instances"]):
                require({"visible_rgb", "visible_alpha"}.issubset(instance.get("images", {})), "缺少全场景单 ID visible RGB/alpha")
                require(all(relative_path(value) == PurePosixPath(prefix + f"_instance{actor:02d}_" + key + ".png")
                            for key, value in instance["images"].items()), "单ID可见PNG的frame/ID绑定不符")
                if name == "observed":
                    require("gt" in images and "body_mask" in instance["images"], "真实观察流缺GT/真实body mask")
                    mask_path = assets.resolve(base, instance["images"]["body_mask"])
                    with assets.Image.open(mask_path) as mask:
                        require(mask.mode == "L", "真实单ID body mask 必须为单通道")
                        histogram = mask.histogram()
                    require(histogram[0] + histogram[255] == 480 * 320, "单ID body mask 不能含灰度软值")
                    require(histogram[255] == instance.get("metrics", {}).get("body_pixels"), "实际单ID mask PNG 与body分母不符")
                else:
                    require("gt" not in images and "body_mask" not in instance["images"] and "metrics" not in instance
                            and "metrics" not in row, "固定相机诊断不能携带 GT、mask监督或GT质量度量")
                instance["images"] = mapped_images(instance["images"], base, assets)
            # 本地网页不引用大体积FP32，保留真实远端NPZ相对来源路径。
            if name == "observed":
                require(raw_frame(row.get("camera_frame_id")) == FRAME_IDS[index], "观察流必须使用各帧自己的实际相机")
            else:
                require(raw_frame(row.get("camera_frame_id")) == "0008033", "固定诊断必须锁定真实0008033相机")
                require(row.get("diagnostic", {}).get("counterfactual_GT_available") is False, "固定诊断不能声称反事实GT")
                require(fixed_camera is None or camera == fixed_camera, "固定相机矩阵随模型时间变化")
                fixed_camera = camera
            result.append(item)
        streams[name] = {"frames": result}
    anchor = manifest["streams"]["observed"]["frames"][10]
    require(fixed_camera == anchor["camera"], "固定相机不等于实际0008033观察相机")
    for row in manifest["streams"]["fixed_camera"]["frames"]:
        require(row["camera_original_timestamp"] == anchor["camera_original_timestamp"], "固定相机原始时间被改写")
    return manifest, streams


def validate_edits(summary_path, edit_root, manifest, assets, *, fixture=False):
    summary = read_json(summary_path)
    require(summary.get("schema") == "drivingeditor_GMC_instance_edit_smoke_CPU_summary_v1", "六操作 summary schema 不符")
    require(bool(summary.get("synthetic_fixture")) == fixture, "合成 summary 不能进入实际结果")
    require(summary.get("instance_ids") == EXPECTED_IDS, "编辑summary真实 ID 不符")
    source = summary.get("source", {})
    require(source.get("checkpoint_steps") == 30000 and source.get("checkpoint") == manifest["checkpoint"]["path"], "六编辑必须与评估使用同一个30k checkpoint")
    require(source.get("scene_spec") == manifest["spec"], "编辑/评估实际scene spec不一致")
    require(summary.get("scope", {}).get("human_quality_verdict") is None
            and summary.get("scope", {}).get("edited_GT_available") is False, "编辑不能代填人工评价或反事实GT")
    contract = summary.get("engineering_contract", {})
    for key in ("passed", "all_FP32_finite", "GT_before_static_vehicles_bitwise_paired_across_operations",
                "static_before_after_bitwise_verified_from_saved_FP32", "all_delete_target_contributions_exact_zero"):
        require(contract.get(key) is True, f"六操作CPU验证未通过: {key}")
    require(contract.get("operation_count") == 6 and contract.get("frames_per_operation") == 49
            and contract.get("verified_FP32_pairs") == 294, "六编辑必须完整294对，不能用局部导出")
    require(set(summary.get("operations", {})) == set(OPERATIONS), "需要两个 ID 各 delete/translate/rotate 三项")
    require(finite_tree(summary), "编辑 summary 含非有限数")
    edits = {}
    root = Path(edit_root).resolve()
    evaluation_root = Path(manifest["_local_path"]).parent
    for name in OPERATIONS:
        operation = summary["operations"][name]
        actor = int(name[5])
        op = name.split("_", 1)[1]
        require(operation.get("actor_index") == actor and operation.get("target_track") == EXPECTED_IDS[actor]
                and operation.get("operation") == op, f"实际操作绑定不符: {name}")
        arguments = operation.get("arguments", {})
        require(arguments.get("checkpoint") == manifest["checkpoint"]["path"]
                and arguments.get("track") == EXPECTED_IDS[actor] and arguments.get("operation") == op, f"编辑 arguments 不符: {name}")
        if op == "translate":
            delta = arguments.get("translate_local_m")
            require(isinstance(delta, list) and len(delta) == 3 and all(math.isclose(float(a), b, abs_tol=1e-8)
                    for a, b in zip(delta, [0.5, 0.0, 0.0])), "translation必须是局部前向+0.5物理米")
        if op == "rotate":
            require(math.isclose(arguments.get("rotate_degrees", float("inf")), 30.0, abs_tol=1e-8), "rotation必须是局部yaw+30度")
        receipt = read_json(root / name / "edit_receipt.json")
        require(receipt.get("status") == ("SYNTHETIC_FIXTURE" if fixture else "EDIT_RENDER_EXPORTED_VISUAL_REVIEW_PENDING"), f"操作实际receipt未完整: {name}")
        require(receipt.get("arguments") == arguments and receipt.get("spec") == manifest["spec"], f"summary和实际receipt来源不一致: {name}")
        require(receipt.get("no_model_parameters_mutated_by_edits") is True and receipt.get("human_quality_verdict") is None, f"实际编辑不变合同缺失: {name}")
        rows, receipt_rows = operation.get("frames", []), receipt.get("frames", [])
        require(len(rows) == len(receipt_rows) == 49
                and [row.get("frame") for row in rows] == list(range(49))
                and [raw_frame(row.get("frame_id")) for row in rows] == FRAME_IDS, f"编辑49实际帧不完整: {name}")
        output = []
        for index, (row, execution) in enumerate(zip(rows, receipt_rows)):
            baseline = manifest["streams"]["observed"]["frames"][index]
            require(execution.get("frame") == index and raw_frame(execution.get("frame_id")) == FRAME_IDS[index], "receipt观察帧不符")
            require(math.isclose(row.get("timestamp", float("inf")), baseline["model_time"], abs_tol=1e-7)
                    and math.isclose(execution.get("timestamp", float("inf")), baseline["model_time"], abs_tol=1e-7), "编辑观察时间不符")
            require(row.get("target_mask_known") is True and row.get("edited_GT_available") is False, "未知mask/编辑GT不能获得准入")
            for key in ("static_render_bitwise_unchanged", "other_instance_Gaussians_bitwise_unchanged"):
                require(execution.get(key) is True, f"实际receipt缺静态/另一ID不变验证: {name}/{index}")
            require(row.get("static_FP32_bitwise_unchanged") is True and row.get("other_instance_world_tensors_receipt_unchanged") is True,
                    "summary缺静态/另一ID不变验证")
            require(execution.get("target_track_valid") == row.get("target_pose_valid"), "轨迹有效性receipt不符")
            if op == "delete":
                require(row.get("after_visible_sum") == 0 and row.get("after_visible_max") == 0, "delete后真实目标贡献非零")
            relative = relative_path(row.get("asset_relative_path", ""))
            require(relative == PurePosixPath(name, f"{index:03d}_{FRAME_IDS[index]}.npz"), "编辑实际文件frame/目录绑定不符")
            prefix = relative.with_suffix("").as_posix()
            image_names = ("gt", "before", "after", "static", "after_static", "before_vehicles", "after_vehicles",
                           "before_target_visible", "after_target_visible")
            images = {key: assets.url(root, prefix + "_" + key + ".png") for key in image_names}
            checks = (("gt", baseline["images"]["gt"]), ("before", baseline["images"]["composite"]),
                      ("static", baseline["images"]["static_rgb"]),
                      ("before_target_visible", baseline["instances"][actor]["images"]["visible_alpha"]))
            for key, reference in checks:
                assets.paired(assets.resolve(root, prefix + "_" + key + ".png"), assets.resolve(evaluation_root, reference))
            assets.paired(assets.resolve(root, prefix + "_static.png"), assets.resolve(root, prefix + "_after_static.png"))
            require(row.get("target_body_pixels") == baseline["instances"][actor].get("metrics", {}).get("body_pixels"), "编辑/评估真实body mask面积不一致")
            box = row.get("native_bbox_padded")
            require(box is None or (isinstance(box, list) and len(box) == 4
                    and 0 <= box[0] < box[2] <= 480 and 0 <= box[1] < box[3] <= 320), "实际原生ROI范围不符")
            output.append({"frame_index": index, "frame_id": FRAME_IDS[index], "model_time": row["timestamp"],
                           "images": images, "target_pose_valid": row["target_pose_valid"], "mask_known": True,
                           "body_pixels": row["target_body_pixels"], "bbox": row.get("native_bbox_padded"),
                           "metrics": {key: value for key, value in row.items() if key in ("before_mask_mse_full_HxW",
                                      "before_visible_body_mean", "before_visible_outside_mean", "edit_full_RGB_MAE")}})
        edits[name] = {"actor_index": actor, "operation": op, "arguments": {key: arguments.get(key)
                       for key in ("translate_local_m", "rotate_degrees")}, "frames": output}
    return summary, edits


def legacy_blocks(original):
    data = re.search(r'<script id="scene-data" type="application/json">(.*?)</script>', original, re.S)
    require(data is not None, "原统一入口缺scene-data；不猜测或重建旧三场景")
    require(set(json.loads(data.group(1))) == {"silver", "black", "red"}, "原三场景范围变化，拒绝自动覆盖")
    scripts = re.findall(r"<script>(.*?)</script>", original, re.S)
    require(len(scripts) == 1 and "const D=JSON.parse" in scripts[0], "原查看器脚本结构变化，需审查新的插入锚点")
    return data.group(0), scripts[0]


def strip_baseline_preview(original):
    """仅移除自家四段预览；资产文件保留，原入口必须逐字恢复。"""
    if BASELINE_MARKER not in original:
        return original, False
    before = legacy_blocks(original)
    expected = {"STYLE": '<style id="gmc-preview-style">',
                "BUTTON": '<button id="gmc-preview-open"',
                "SECTION": '<section id="gmc-preview-panel"',
                "DATA_AND_SCRIPT": '<script id="gmc-preview-data" type="application/json">'}
    require(original.count(BASELINE_MARKER) == 8, "自家 baseline preview 标记不完整，拒绝猜测删除")
    result = original
    for name, signature in expected.items():
        pattern = (r"<!-- BEGIN " + BASELINE_MARKER + " " + name + r" -->\r?\n(.*?)\r?\n<!-- END "
                   + BASELINE_MARKER + " " + name + r" -->\r?\n")
        matches = list(re.finditer(pattern, result, re.S))
        require(len(matches) == 1 and signature in matches[0].group(1), "baseline preview 自家区域边界/身份不符: " + name)
        if name == "DATA_AND_SCRIPT":
            match = re.search(r'<script id="gmc-preview-data" type="application/json">(.*?)</script>', matches[0].group(1), re.S)
            require(match is not None and json.loads(match.group(1)).get("schema") == "drivingeditor_GMC_baseline_preview_data_v1"
                    and "gmc_instance_baseline_preview.js" in matches[0].group(1), "自家预览 data/script 身份不符")
        result = result.replace(matches[0].group(0), "", 1)
    require(BASELINE_MARKER not in result and legacy_blocks(result) == before, "移除预览改写了旧三场景证据")
    return result, True


STYLE = """<style id="gmc-instance-style">
#gmc-instance-panel{border:1px solid #598b86;border-radius:13px;background:#0e1d2b;padding:16px;margin:16px 0}
#gmc-instance-panel h2{margin:0 0 6px;font-size:21px}#gmc-instance-panel .gmc-controls{border:1px solid #345068;border-radius:10px;padding:12px;margin:14px 0}
#gmc-instance-panel .gmc-row{display:flex;align-items:center;flex-wrap:wrap;gap:10px}#gmc-instance-panel label{font-size:13px;color:#b7c8dd}
#gmc-instance-panel .gmc-frame{flex:1;min-width:190px}#gmc-instance-panel .gmc-detail,#gmc-instance-panel .gmc-note{color:#a0b2c9;font-size:12px}
#gmc-instance-panel .gmc-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:14px 0}#gmc-instance-panel.gmc-fixed .gmc-grid{grid-template-columns:1fr}
#gmc-instance-panel .gmc-card{border:1px solid #2a3a51;border-radius:11px;overflow:hidden;background:#101c2d;min-width:0}
#gmc-instance-panel .gmc-card header,#gmc-instance-panel .gmc-card footer{padding:8px 12px}#gmc-instance-panel .gmc-card header{font-weight:600}
#gmc-instance-panel .gmc-card footer{font-size:12px;color:#a0b2c9}#gmc-instance-panel .gmc-surface{aspect-ratio:3/2;position:relative;overflow:hidden;background:#000}
#gmc-instance-panel .gmc-surface img{position:absolute;left:0;top:0;width:100%;height:auto;max-width:none;display:block}
#gmc-instance-panel code{overflow-wrap:anywhere;font-size:12px}#gmc-instance-panel .gmc-components{width:100%;height:auto;max-height:155px;margin:12px 0}
#gmc-instance-panel .gmc-missing{padding:10px;border:1px solid #aa7651;color:#efba87}#gmc-instance-panel table{font-size:12px}
@media(max-width:740px){#gmc-instance-panel .gmc-grid{grid-template-columns:1fr}#gmc-instance-panel{padding:12px}#gmc-instance-panel select{max-width:100%}}
</style>"""


SECTION = """<section id="gmc-instance-panel" aria-labelledby="gmc-instance-title" hidden>
<div class="eyebrow">GMC · TWO RIGID INSTANCES · ACTUAL 30K</div><h2 id="gmc-instance-title">GMC 两个 ID：逐帧重建与编辑</h2>
<p>GMC77 皮卡与 Jeep82 分别编辑；Corolla 与其他停车保留原静态背景。下方直接显示保存的 49 帧结果。原三场景 GN11 对照仍由上方①②按钮查看。</p>
<svg class="gmc-components" viewBox="0 0 920 118" role="img" aria-label="真实输入经静态背景和两实例刚体高斯，输出全场景重建、可见贡献与ID编辑">
<defs><marker id="gmc-component-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#68d8c1"/></marker></defs>
<g fill="#172b3a" stroke="#68d8c1"><rect x="4" y="24" width="205" height="70" rx="10"/><rect x="239" y="24" width="220" height="70" rx="10"/><rect x="489" y="24" width="190" height="70" rx="10"/><rect x="709" y="24" width="205" height="70" rx="10"/></g>
<g stroke="#68d8c1" stroke-width="2" marker-end="url(#gmc-component-arrow)"><path d="M209,59 H233"/><path d="M459,59 H483"/><path d="M679,59 H703"/></g>
<g fill="#e3edf9" text-anchor="middle" font-family="system-ui,Microsoft YaHei,sans-serif" font-size="14"><text x="106" y="51">真实 RGB / LiDAR</text><text x="106" y="75">track + reviewed mask</text><text x="349" y="51">静态背景 + 两实例高斯</text><text x="349" y="75">局部形状 → 刚体位姿</text><text x="584" y="51">全场景深度排序</text><text x="584" y="75">每 ID 可见 alpha / RGB</text><text x="811" y="51">49 观察帧 + ID 编辑</text><text x="811" y="75">固定相机无 GT 诊断</text></g></svg>
<div class="gmc-row"><button id="gmc-observed" aria-pressed="true">真实逐帧 · 编辑</button><button id="gmc-fixed" aria-pressed="false">固定相机 · 诊断（无 GT）</button></div>
<div class="notice" id="gmc-notice"></div><div class="gmc-controls"><div class="gmc-row">
<div class="buttons"><button id="gmc-prev" aria-label="GMC上一帧">←</button><button id="gmc-play">播放</button><button id="gmc-next" aria-label="GMC下一帧">→</button></div>
<div class="gmc-frame"><strong id="gmc-frame-title"></strong><div class="gmc-detail" id="gmc-frame-detail"></div></div>
<label for="gmc-actor">实例 ID</label><select id="gmc-actor"><option value="0">GMC77 · 蓝色皮卡</option><option value="1">Jeep82 · 灰色 SUV</option></select>
<label for="gmc-operation">编辑</label><select id="gmc-operation"><option value="delete">删除所选 ID</option><option value="translate">局部前向 +0.5 m</option><option value="rotate">局部 yaw +30°</option></select>
<label for="gmc-branch">分支</label><select id="gmc-branch"><option value="paired">合成 RGB / GT / static</option><option value="visible_alpha">所选 ID 可见 alpha</option><option value="visible_rgb">所选 ID 可见 RGB · 未编辑</option></select>
<label for="gmc-crop">视野</label><select id="gmc-crop"><option value="full">完整原图</option><option value="roi">真实 body ROI</option></select>
</div><input id="gmc-timeline" aria-label="GMC实例49帧进度" type="range" min="0" max="48" value="0"><div id="gmc-crop-hint" class="gmc-note"></div></div>
<p class="gmc-note">稳定 raw track ID：<code id="gmc-track-id"></code></p><div id="gmc-gallery" class="gmc-grid"></div>
<div class="table-wrap"><table><thead><tr><th>当前帧诊断</th><th>实际保存值</th></tr></thead><tbody id="gmc-metric-body"></tbody></table></div>
<p id="gmc-evidence" class="gmc-note"></p><p class="gmc-note">49 帧为训练观测重建，不是 heldout。GT 是未编辑原图；删除、平移、旋转和固定相机没有对应反事实 GT。单 ID visible RGB 为完整场景排序后的加权颜色贡献，黑画布不补环境；alpha 为联合可见贡献，没有预乘 body mask 或逐图归一化。人工质量评价：null；补景、残影、阴影与遮挡边界须保留实际视觉审核。</p>
</section>"""


def build_patch(original, data, script_url):
    require(MARKER not in original, "已有实例区域，保留现入口；不要重复覆盖")
    before = legacy_blocks(original)
    newline = "\r\n" if "\r\n" in original else "\n"
    blocks = []
    def block(name, contents):
        value = f"<!-- BEGIN {MARKER} {name} -->\n{contents}\n<!-- END {MARKER} {name} -->\n".replace("\n", newline)
        blocks.append(value)
        return value
    result = original.replace("</head>", block("STYLE", STYLE) + "</head>", 1)
    nav = re.search(r'<nav class="modes"[^>]*>.*?</nav>', result, re.S)
    require(nav is not None, "旧模式按钮插入锚点缺失")
    button = '<button id="gmc-instance-open" aria-controls="gmc-instance-panel" aria-pressed="false" hidden>③ GMC · 两 ID 编辑（30k）</button>'
    offset = nav.end() - len("</nav>")
    result = result[:offset] + block("BUTTON", button) + result[offset:]
    anchor = '<section class="controls">'
    require(result.count(anchor) == 1, "旧controls插入锚点不唯一")
    result = result.replace(anchor, block("SECTION", SECTION) + anchor, 1)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    boot = f'<script id="gmc-instance-data" type="application/json">{payload}</script>\n<script src="{script_url}" defer></script>'
    result = result.replace("</body>", block("DATA_AND_SCRIPT", boot) + "</body>", 1)
    require(legacy_blocks(result) == before, "新接入意外改写旧三场景 JSON 或脚本")
    restored = result
    for item in blocks:
        require(restored.count(item) == 1, "新增区域定位不唯一")
        restored = restored.replace(item, "", 1)
    require(restored == original, "新增区域之外的旧证据/按钮被改写")
    return result


def prepare(args):
    target = args.target.resolve()
    if args.allow_synthetic_fixture:
        require(within(target, WORKSPACE / "work") and not within(target, OUTPUTS), "合成 fixture 必须留在 work，不得进入 outputs")
        if args.validation_output:
            require(within(args.validation_output, WORKSPACE / "work") and not within(args.validation_output, OUTPUTS),
                    "合成 fixture 的 validation receipt 也必须留在 work")
    else:
        require(target == (OUTPUTS / "DrivingEditor_final_report.html").resolve(), "生产仅修改既有统一入口，不创建另一个 HTML")
    require(target.is_file(), f"既有统一入口不存在: {target}")
    original_bytes = target.read_bytes()
    original, baseline_preview_upgraded = strip_baseline_preview(original_bytes.decode("utf-8"))
    assets = BrowserAssets(target.parent)
    manifest, streams = validate_evaluation(args.evaluation_manifest, assets, fixture=args.allow_synthetic_fixture)
    manifest["_local_path"] = str(args.evaluation_manifest.resolve())
    summary, edits = validate_edits(args.edit_summary, args.edit_root, manifest, assets, fixture=args.allow_synthetic_fixture)
    js_source = HERE / "gmc_instance_viewer.js"
    require(js_source.is_file(), "专用 JS 未就绪")
    script_destination = target.parent / "drivingeditor_GMC_instances_30k" / "gmc_instance_viewer.js"
    require(not script_destination.exists(), f"保留已存在JS资产，不覆盖: {script_destination}")
    script_url = "/".join(quote(part, safe="_-.") for part in script_destination.relative_to(target.parent).parts)
    data = {"schema": "drivingeditor_GMC_unified_viewer_data_v1", "synthetic_fixture": args.allow_synthetic_fixture,
            "checkpoint": manifest["checkpoint"], "instance_ids": EXPECTED_IDS, "streams": streams, "edits": edits,
            "units": manifest.get("units"), "normalization": manifest.get("normalization"),
            "observed_metrics": manifest.get("observed_metrics"), "human_quality_verdict": None,
            "engineering_checks": {"evaluation_contracts": manifest["contracts"], "edit_contracts": summary["engineering_contract"]}}
    updated = build_patch(original, data, script_url)
    if args.allow_synthetic_fixture:
        updated = updated.replace("ACTUAL 30K</div>", "SYNTHETIC FIXTURE · 非结果</div>", 1)
    receipt = {"schema": "drivingeditor_GMC_unified_viewer_merge_receipt_v1",
               "checked_utc": datetime.now(timezone.utc).isoformat(),
               "status": "VALIDATED_NOT_APPLIED", "synthetic_fixture": args.allow_synthetic_fixture,
               "target": str(target), "evaluation_manifest": str(args.evaluation_manifest.resolve()),
               "edit_summary": str(args.edit_summary.resolve()), "edit_root": str(args.edit_root.resolve()),
               "checkpoint_step": 30000, "observed_frames": 49, "fixed_camera_frames": 49,
               "edit_operations": 6, "edit_frame_pairs": 294, "browser_PNGs_verified": len(assets.checked),
               "PNG_pair_checks": assets.pair_checks, "legacy_three_scenes_data_script_buttons_and_evidence_preserved": True,
               "baseline_preview_upgraded": baseline_preview_upgraded,
               "GUI_QA_performed": False, "human_quality_verdict": None, "photo_quality_admitted": False,
               "no_training_or_rendering": True, "GPU_used": False, "new_production_HTML_entry_created": False,
               "edited_per_ID_visible_RGB_exported": False,
               "limitations": ["编辑分支当前真实出口仅有每ID before/after visible alpha",
                               "visible RGB 分支只显示未编辑的全场景加权贡献", "固定相机没有GT/PSNR或编辑后图",
                               "这里只验证结构、绑定、真实PNG路径与配对；不是GUI或照片质量验收"]}
    if args.validation_output:
        require(not args.validation_output.exists(), "保留已有validation receipt，不覆盖")
    if args.apply:
        backup_directory = HERE / "viewer_merge_backups"
        backup_directory.mkdir(parents=True, exist_ok=True)
        backup = backup_directory / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".original")
        backup.write_bytes(original_bytes)
        script_destination.parent.mkdir(parents=True, exist_ok=True)
        with script_destination.open("xb") as stream:
            stream.write(js_source.read_bytes())
        temporary = target.with_name(target.name + ".instance_merge.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(updated.encode("utf-8"))
            require(target.read_bytes() == original_bytes, "统一入口在校验期间发生并发修改，保留新JS与备份，拒绝覆盖HTML")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        receipt.update(status="APPLIED_TO_EXISTING_UNIFIED_ENTRY", backup=str(backup), script=str(script_destination))
    if args.validation_output:
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--edit-summary", required=True, type=Path)
    parser.add_argument("--edit-root", required=True, type=Path)
    parser.add_argument("--target", type=Path, default=OUTPUTS / "DrivingEditor_final_report.html")
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--apply", action="store_true", help="所有实际资产通过后修改唯一既有HTML入口")
    parser.add_argument("--allow-synthetic-fixture", action="store_true", help="只在work测试，明确标为合成非结果")
    args = parser.parse_args()
    try:
        receipt = prepare(args)
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as error:
        print(f"未接入，保留原统一入口: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""只用已完整取回的 CPU / 原生短探针证据生成独立报告；不打开 SSH/CUDA。"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image

CTX = "14705303724557273004_3105_000_3125_000"
RAWS = [98, 113, 114]
STATES = ["intact", "actor_removed", "r2_removed"]
CONTEXTS = ["actor_present", "actor_removed"]
MODES = ["both_retained", "whole_parent_remove", "gt_target_hypothesis_child_remove", "leq_road_hypothesis_child_remove"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


class Inputs:
    """检查读取期间文件未改变；未完成 JSON/ZIP/PNG 直接失败，不生成完成页。"""
    def __init__(self, base):
        self.base = base
        self.reads = {}
        self.png_checks = []

    def load(self, path, reader):
        path = Path(path)
        before = path.stat()
        require(before.st_size > 0, "Empty/unfinished input: " + str(path))
        value = reader(path)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Input changed while reading: " + str(path))
        self.reads[str(path)] = (after.st_size, after.st_mtime_ns)
        return value

    def json(self, path, complete=False):
        value = self.load(path, lambda p: json.loads(p.read_text(encoding="utf-8-sig")))
        if complete:
            require(value.get("status") == "complete", "Run has no complete summary: " + str(path))
        return value

    def npz(self, path):
        def reader(p):
            with np.load(p, allow_pickle=False) as source:
                return {key: source[key] for key in source.files}
        return self.load(path, reader)

    def png(self, path, rgb):
        require(rgb.shape == (3, 320, 480) and np.isfinite(rgb).all(), "Invalid float RGB: " + str(path))
        expected = np.round(np.clip(rgb.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        def reader(p):
            with Image.open(p) as source:
                return np.asarray(source.convert("RGB")).copy()
        actual = self.load(path, reader)
        require(np.array_equal(actual, expected), "PNG != native float display encoding: " + str(path))
        self.png_checks.append(dict(path=Path(path).relative_to(self.base).as_posix(),
                                    shape=list(actual.shape), max_abs_byte_error=0))

    def unchanged(self):
        for path, original in self.reads.items():
            state = Path(path).stat()
            require((state.st_size, state.st_mtime_ns) == original, "Input changed before publication: " + path)


def pixel_metrics(package, reference, masks):
    difference = np.abs(package["rgb"].astype(np.float64) - reference["rgb"].astype(np.float64))
    delta = difference.mean(0)
    ad = np.abs(package["alpha"].astype(np.float64).reshape(320, 480) - reference["alpha"].astype(np.float64).reshape(320, 480))
    result = {}
    for name, mask in {"full_frame": np.ones((320, 480), bool), **masks}.items():
        count = int(mask.sum())
        result[name] = dict(pixels=count, mean_abs_rgb_change=float(delta[mask].mean()) if count else None,
            max_abs_rgb_change=float(difference[:, mask].max()) if count else None,
            rgb_mse=float((difference[:, mask] ** 2).mean()) if count else None,
            mean_abs_alpha_change=float(ad[mask].mean()) if count else None,
            changed_pixels_above_1_255=int((delta[mask] > 1/255).sum()) if count else 0)
    return result


def check_metrics(actual, expected, label):
    require(actual.keys() == expected.keys(), "Metric domains differ: " + label)
    for domain in actual:
        for key, value in actual[domain].items():
            old = expected[domain][key]
            ok = value == old if value is None or isinstance(value, int) else np.isclose(value, old, rtol=1e-9, atol=1e-12)
            require(ok, f"Metric replay differs: {label}/{domain}/{key}: {value} vs {old}")


def roi(body):
    yy, xx = np.nonzero(body)
    require(len(xx) > 0, "Empty crop body")
    x0, y0 = max(0, int(xx.min()) - 24), max(0, int(yy.min()) - 20)
    x1, y1 = min(480, int(xx.max()) + 25), min(320, int(yy.max()) + 39)
    return [x0, y0, x1 - x0, y1 - y0]


def f(value, digits=4):
    return "缺测" if value is None else f"{value:.{digits}f}"


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("src", "href") and value:
                parsed = urlsplit(value)
                if not parsed.scheme and parsed.path:
                    self.paths.append(unquote(parsed.path))


def run(root, base):
    root, base = root.resolve(), base.resolve()
    inputs = Inputs(base)
    cpu = inputs.json(base / "cpu_audit.json")
    probe = inputs.json(base / "probe_anchors_r1/summary.json", complete=True)
    geometry = inputs.json(base / "support_geometry_r1/summary.json", complete=True)
    refit = inputs.json(base / "refit_raw113_r2/summary.json", complete=True)
    manifest = inputs.json(base / "refit_raw113_r2/manifest.json")
    restore = inputs.json(base / "refit_raw113_r2/restore_verification.json")
    before = inputs.json(base / "refit_raw113_r1/before_metrics.json")
    after = inputs.json(base / "refit_raw113_r2/after_metrics.json")
    require(all(x["context"] == CTX for x in [cpu, probe, geometry, refit]), "Scene mismatch")
    require(probe["training_updates"] == 0 and probe["source_checkpoints_unchanged"], "Probe source changed")
    require(geometry["source_checkpoint_unchanged"] and not geometry["thresholds"]["thresholds_adjusted"], "Geometry contract differs")
    require(refit["children"] == 14 and refit["parent_count"] == 7 and refit["children_both_sides_retained"], "Fixed split contract differs")
    require(refit["child_refit_updates_total"] == 200 and refit["child_refit_updates_this_run"] == 0 and refit["resume_acceptance_only"], "Refit accounting differs")
    require(refit["classifier_training_updates"] == refit["original_model_training_updates"] == 0 and refit["source_checkpoints_unchanged"], "Original/classifier model changed")
    require(refit["before"] == before and refit["after"] == after, "Completed summary and stage metric files differ")
    require(restore == refit["restoration"] and restore["all_passed"] and all(restore["checks"].values()), "Restore JSON is not all-pass")
    require(restore["RGB_max_abs_error"] == 0 and restore["optimizer_replayed_updates"] == 0 and restore["checks"]["alpha_exact"], "Restore RGB/alpha or update check failed")
    require(refit["resume_provenance"]["source_bindings_verified"] and refit["resume_provenance"]["failed_scene_preserved"], "Resume lineage is not verified")
    require(refit["source_bindings"]["static_namespace"] == cpu["static_namespace"] and
            refit["source_bindings"]["actor_checkpoint_sha256"] == cpu["actor_checkpoint_sha256_receipt"], "Checkpoint namespace differs from CPU audit")
    require(geometry["static_checkpoint_ref"] == cpu["static_checkpoint_ref"], "Geometry checkpoint differs")
    require(not refit["quality_accepted"] and not refit["edit_benefit_proven"], "Unexpected acceptance state; revise report language")

    p_records = {r["raw"]: r for r in probe["records"]}
    g_records = {r["raw"]: r for r in geometry["records"]}
    require(sorted(p_records) == sorted(g_records) == RAWS, "Three-anchor coverage differs")
    native_packages, crop_rois, image_paths = {}, {}, []
    max_decomposition_error = 0.0
    for raw in RAWS:
        body_source = root / f"work/v43_r2_regression_20261029/r2_delivery/r2_inherited49/{raw:03d}/evidence.npz"
        crop_rois[str(raw)] = roi(inputs.npz(body_source)["body"])
        environments = []
        for state in STATES:
            prefix = f"probe_anchors_r1/{raw:03d}/{state}"
            package = inputs.npz(base / (prefix + "_evidence.npz"))
            native_packages[raw, state] = package
            environment_contribution = (1 - package["alpha"]) * package["environment_rgb"]
            error = float(np.max(np.abs(package["rgb"] - package["gaussian_rgb"] - environment_contribution)))
            require(error <= 1e-6, "Native environment decomposition differs")
            max_decomposition_error = max(max_decomposition_error, error)
            for suffix, pixels in [("", package["rgb"]), ("_gaussian", package["gaussian_rgb"]), ("_environment", environment_contribution)]:
                path = prefix + suffix + ".png"
                inputs.png(base / path, pixels)
                image_paths.append(path)
            environments.append(package["environment_rgb"])
            require(p_records[raw]["states"][state]["adjoint_closure"] < 1e-4, "Native adjoint closure failed")
        require(all(np.array_equal(environments[0], value) for value in environments[1:]), "Environment changed between native states")
        if p_records[raw]["split"]["applied_rows"]:
            split = inputs.npz(base / f"probe_anchors_r1/{raw:03d}/split_float.npz")
            path = f"probe_anchors_r1/{raw:03d}/split_both_retained.png"
            inputs.png(base / path, split["rgb"])
            image_paths.append(path)
        require(g_records[raw]["candidates"] == p_records[raw]["dual_measured_support_rows"] and
                g_records[raw]["both_centroids_within_3sigma_rows"] == 0, "Geometry denominator differs")

    masks = inputs.npz(base / "refit_raw113_r2/domains.npz")
    references, packages = {}, {}
    metric_replays, purity_replays = 0, []
    for stage, run_name in [("before", "refit_raw113_r1"), ("after", "refit_raw113_r2")]:
        for context in CONTEXTS:
            prefix = f"{run_name}/references/{context}"
            reference = inputs.npz(base / (prefix + ".npz"))
            inputs.png(base / (prefix + ".png"), reference["rgb"])
            image_paths.append(prefix + ".png")
            references[stage, context] = reference
            for mode in MODES:
                prefix = f"{run_name}/{stage}/{context}/{mode}"
                package = inputs.npz(base / (prefix + ".npz"))
                packages[stage, context, mode] = package
                inputs.png(base / (prefix + ".png"), package["rgb"])
                image_paths.append(prefix + ".png")
            expected = (before if stage == "before" else after)[context]
            both = packages[stage, context, "both_retained"]
            check_metrics(pixel_metrics(both, reference, masks), expected["reconstruction"], stage + "/" + context)
            metric_replays += 1
            for mode in MODES[1:]:
                package = packages[stage, context, mode]
                for label, baseline in [("relative_to_stage_both_retained", both), ("relative_to_frozen_original", reference)]:
                    check_metrics(pixel_metrics(package, baseline, masks), expected["controls"][mode][label], f"{stage}/{context}/{mode}/{label}")
                    metric_replays += 1
            contributions = inputs.npz(base / f"{run_name}/{stage}_{context}_child_contributions.npz")
            weights, sides = contributions["weights"], contributions["child_sides"]
            require(weights.shape == (14, 2) and set(sides) == {"gt", "leq"} and np.isfinite(weights).all(), "Contribution layout differs")
            numerator = weights[sides == "gt", 0].sum() + weights[sides == "leq", 1].sum()
            match = float(numerator / weights.sum())
            recorded = expected["contribution_purity"]["hypothesized_side_match_fraction"]
            require(abs(match - recorded) <= 1e-12, "Two-domain side match does not replay")
            purity_replays.append(dict(stage=stage, context=context, match_fraction=match, absolute_error=abs(match-recorded)))

    refit_roi = roi(masks["body"])
    scene_link = f"../../../GS_Scene_Diagnostics_20261027/{CTX}/index.html"
    data = dict(raws=RAWS, rois=crop_rois, refitRoi=refit_roi, contextLabels={"actor_present": "Actor 保留", "actor_removed": "Actor 已移除"})
    state_titles = {"intact": "完整 static + Actor10K", "actor_removed": "仅移除 Actor10K", "r2_removed": "再移除 R2 行集 · 原环境"}
    def visual(prefix, title, group, slot, crop_only=False):
        full = "" if crop_only else f'<img class="full" data-image="{group}:{slot}" src="{html.escape(prefix)}" alt="{html.escape(title)}，原生完整图">'
        return f'''<article class="visual"><h3>{title}</h3>{full}<div class="crop" data-crop="{group}:{slot}"><img data-image="{group}:{slot}" src="{html.escape(prefix)}" alt="{html.escape(title)}，同源局部裁切"></div><a class="original" data-original="{group}:{slot}" href="{html.escape(prefix)}" target="_blank">打开完整原图 ↗</a></article>'''
    state_images = "".join(visual(f"probe_anchors_r1/113/{state}.png", state_titles[state], "state", state) for state in STATES)
    refit_images = "".join([
        visual("refit_raw113_r2/references/actor_removed.png", "冻结原模型参考", "refit", "reference"),
        visual("refit_raw113_r1/before/actor_removed/both_retained.png", "拆分初始化 · 两侧保留", "refit", "before"),
        visual("refit_raw113_r2/after/actor_removed/both_retained.png", "200步后 · 两侧保留", "refit", "after")])
    control_titles = {"whole_parent_remove": "移除7个完整父高斯", "gt_target_hypothesis_child_remove": "仅移除 target 假设侧", "leq_road_hypothesis_child_remove": "仅移除 road 假设侧"}
    control_images = "".join(visual(f"refit_raw113_r2/after/actor_removed/{mode}.png", title, "control", mode, crop_only=True) for mode, title in control_titles.items())

    geometry_rows = "".join(f'''<tr><td>{raw}</td><td>{g_records[raw]['candidates']:,}</td><td><strong>0 / {g_records[raw]['candidates']:,}</strong></td><td>{g_records[raw]['target_centroid_within_3sigma_rows']} / {g_records[raw]['road_centroid_within_3sigma_rows']}</td><td>{g_records[raw]['tiny_tail_descriptive_rows']:,}</td><td>{p_records[raw]['split']['applied_rows']}</td></tr>''' for raw in RAWS)
    match_rows = "".join(f'''<tr><td>{data['contextLabels'][context]}</td><td>{f(refit['contribution_separation'][context]['before'])}</td><td>{f(refit['contribution_separation'][context]['after'])}</td><td class="{'positive' if refit['contribution_separation'][context]['change'] > 0 else 'negative'}">{refit['contribution_separation'][context]['change']:+.4f}</td></tr>''' for context in CONTEXTS)
    local_rows = []
    for context in CONTEXTS:
        for domain, label in [("body", "目标 body"), ("measured_safe_road", "有实测深度的安全道路"), ("outside_allowed", "查询范围外")]:
            old, new = before[context]["reconstruction"][domain], after[context]["reconstruction"][domain]
            local_rows.append(f'<tr><td>{data["contextLabels"][context]}</td><td>{label} · {old["pixels"]:,}px</td><td>{f(old["mean_abs_rgb_change"],6)} → {f(new["mean_abs_rgb_change"],6)}</td><td>{old["changed_pixels_above_1_255"]:,} → {new["changed_pixels_above_1_255"]:,}</td></tr>')
    routing_rows = []
    data["routing"] = {}
    for context in CONTEXTS:
        data["routing"][context] = []
        for mode, title in control_titles.items():
            values = after[context]["controls"][mode]["relative_to_stage_both_retained"]
            data["routing"][context].append([title, values["body"]["mean_abs_rgb_change"],
                values["measured_safe_road"]["mean_abs_rgb_change"], values["measured_safe_road"]["changed_pixels_above_1_255"]])
    for mode, title in control_titles.items():
        metrics = after["actor_removed"]["controls"][mode]["relative_to_stage_both_retained"]
        routing_rows.append(f'<tr><td>{title}</td><td>{f(metrics["body"]["mean_abs_rgb_change"],6)}</td><td>{f(metrics["measured_safe_road"]["mean_abs_rgb_change"],6)}</td><td>{metrics["measured_safe_road"]["changed_pixels_above_1_255"]:,}</td></tr>')

    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>场景一 · 可编辑归属证据与双侧拆分短探针</title><style>
:root{color-scheme:dark;--bg:#09121b;--panel:#101f2b;--line:#294052;--text:#e6eef4;--muted:#a4b6c5;--cyan:#7bd7de;--orange:#ffc185;--red:#ffb0a8}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.75 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1320px;margin:auto;padding:36px 28px 70px}header{padding-bottom:26px;border-bottom:1px solid var(--line)}.eyebrow{font:12px/1.5 ui-monospace,monospace;letter-spacing:.12em;color:var(--cyan)}h1{font-size:clamp(25px,3.3vw,39px);line-height:1.3;max-width:1000px;margin:14px 0}h2{font-size:23px;line-height:1.4;margin:0 0 15px}h3{font-size:14px;font-weight:600;margin:0 0 11px}.lead{font-size:18px;max-width:1080px;color:#cfdee8}.muted,.caption{color:var(--muted)}a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}.badges,.links{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}.badge{border:1px solid var(--line);border-radius:5px;padding:4px 10px;font-size:12px}.badge.warn{color:var(--orange);border-color:#78583d}.links a{font-size:13px;padding-right:10px}nav{display:flex;gap:20px;flex-wrap:wrap;margin:22px 0;font-size:13px}section{margin-top:36px;padding-top:4px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:22px;margin-top:18px}.callout{border-left:3px solid var(--orange);padding:12px 18px;background:#221f1a;border-radius:0 7px 7px 0}.callout p{margin:0}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:18px}.stat b{display:block;font-size:26px;line-height:1.35;color:var(--cyan)}.stat span{display:block;font-size:13px;color:var(--muted);margin-top:5px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}th{color:#a9c5d7;background:#0c1a25;font-weight:500}td strong{color:var(--orange)}.table-wrap{overflow-x:auto}.positive{color:var(--cyan)}.negative{color:var(--orange)}.compare{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px;margin-top:17px}.visual{min-width:0;background:#0a1520;border:1px solid var(--line);border-radius:7px;padding:12px}.visual img.full{width:100%;display:block;border-radius:3px}.crop{position:relative;overflow:hidden;margin-top:12px;background:#04090e;border:1px solid #3f5362;border-radius:3px}.crop img{position:absolute;height:auto;max-width:none;display:block;image-rendering:auto}.original{display:inline-block;font-size:12px;margin-top:8px}.toolbar{display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;margin-top:18px}.toolbar label{font-size:13px}select{font:inherit;background:#142b3b;color:var(--text);border:1px solid #416076;padding:7px 14px;border-radius:5px}code{font:12px/1.7 ui-monospace,monospace;color:#b5d9e8;overflow-wrap:anywhere}.architecture{width:100%;display:block;background:white;border-radius:6px}.caption{font-size:12px;margin-top:9px}details{border-top:1px solid var(--line);margin-top:18px;padding-top:13px}summary{cursor:pointer;color:var(--cyan);font-size:13px}p{margin:12px 0}.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}ol{padding-left:22px}li{margin:8px 0}footer{margin-top:42px;border-top:1px solid var(--line);padding-top:20px;font-size:12px;color:var(--muted)}@media(max-width:850px){main{padding:24px 16px}.compare,.two{grid-template-columns:1fr}.stats{grid-template-columns:1fr}h2{font-size:21px}.visual img.full{max-width:600px}.crop{max-width:600px}.panel{padding:16px}th,td{padding:9px}}
</style></head><body><main><header><div class="eyebrow">V43 / SCENE 01 / 2026-10-30</div><h1>归属证据与双侧子高斯：工程短探针完成，编辑质量未通过</h1><p class="lead">先验证旧高斯是否覆盖目标与道路的真实支撑，再谈拆分后的稳定归属。当前双质心几何覆盖为零；仅调颜色和 opacity 后，遮挡消失时的两域匹配没有改善。</p><div class="badges"><span class="badge">原生三态 GPU 已执行</span><span class="badge">子高斯 refit 累计200步</span><span class="badge">恢复验收本次0步</span><span class="badge">分类器0步 · 原模型未改</span><span class="badge warn">quality_accepted=false</span></div><nav><a href="#cpu">49帧证据</a><a href="#native">同帧原生三态</a><a href="#geometry">几何支撑</a><a href="#refit">固定拆分 refit</a><a href="#next">边界与下一步</a></nav><div class="links"><a href="../r2_recovery_20261029/index.html">既有 R2 交付 ↗</a><a href="../unknown_white_20261030/index.html">既有 unknown 白材质 ↗</a><a href="@@SCENE@@#workspace?group=v43_r2_inherited49&amp;raw=113&amp;channel=rgb">共享 R2 工作台 ↗</a></div></header>
<section id="cpu"><h2>帧证据不能直接变成永久 owner</h2><div class="stats"><div class="stat"><b>1,855 / 6,241</b><span>Actor 行在全部49帧满足已有 core 条件</span></div><div class="stat"><b>56,082</b><span>static 持久行跨帧曾分别获得 road / vehicle 预测</span></div><div class="stat"><b>65,563 / 73,102</b><span>R2 曾选行中，也曾出现 road 预测的行</span></div></div><p>static 共918,850行，其中371,406行被49帧导出观察，547,444行缺少保存的可见贡献证据。未观察、已观察但语义弃权、跨帧冲突分别保留；R2 membership 只是历史选择。上述数字不构成分类准确率或owner真值。</p><p class="caption">默认 Actor core：本帧可见，已有类别14–19，5px轮廓带贡献比例&lt;0.2。<a href="cpu_audit.json">CPU审计</a> · <a href="review_candidates.json">28个复核候选</a> · <a href="row_evidence_partitions.npz">轻量行证据</a> · <a href="cpu_findings.md">CPU科学审计正文</a></p></section>
<section><details open><summary>既有 architecture components 图</summary><div class="panel"><img class="architecture" src="../../architecture.svg" alt="既有V43组件：原始观测、证据、资产与编辑数据流"><p class="caption">沿用既有组件图。本轮新增的是证据与固定拆分短探针，图中既有分支不代表本轮全部重跑。</p></div></details></section>
<section id="native"><h2>同一相机，三种移除状态，各自重算 alpha×T</h2><p>完整模型 → 仅去 Actor10K → 再去本帧 R2 static 行集。每态重新排序、计算透射率和观测矩；原环境保持不变。第三列没有局部白底替换或蓝框门控，操作与既有 R2 显示交付不同。</p><div class="toolbar"><label>原生锚点 <select id="rawSelect"><option value="98">raw98</option><option value="113" selected>raw113</option><option value="114">raw114</option></select></label><span class="caption" id="stateMeta"></span></div><div class="compare">@@STATE_IMAGES@@</div><p class="caption">上：完整480×320原生PNG。下：同一PNG的CSS局部视窗；由body bbox统一扩展左24 / 上20 / 右24 / 下38px，没有另造裁片。浮点以原公式显示裁剪至[0,1]；亮色缺口不表示背景已恢复。</p><details><summary>原生分量与机器证据</summary><div class="links" id="stateEvidence"></div><p class="caption">环境图显示实际 (1−alpha)×Env；PNG只作显示，未裁剪的RGB、alpha和原环境在NPZ中保留。</p></details></section>
<section id="geometry"><h2>双域投影存在，两个深度质心却未同时落在父高斯3σ内</h2><div class="table-wrap panel"><table><thead><tr><th>raw</th><th>实测双域候选</th><th>双质心均≤3σ</th><th>仅target / road≤3σ</th><th>小尾质量&lt;1e−8</th><th>初始化保留父行</th></tr></thead><tbody>@@GEOMETRY_ROWS@@</tbody></table></div><p>候选来自完整模型下的实测深度像素回投及 alpha×T 加权质心；它们不是两个真实表面标签。3σ计数只是描述性几何检查，阈值未调整。452 / 777 / 810条候选中，没有父高斯同时覆盖两侧观测质心，因此不能把屏幕上的双域贡献直接解释成适合沿中间面切开的空间混合。</p><p>每帧只试原排序前8个父行；数值可保留的初始化为0 / 7 / 8条父行。其双侧截断高斯矩匹配保留两侧谱系，但不保证逐像素alpha守恒、语义边界正确或编辑收益。raw98因没有安全初始化而跳过，不能填入另一帧结果。</p><p class="caption"><a href="support_geometry_r1/summary.json">几何定义与全部统计</a> · <a href="probe_anchors_r1/summary.json">三态与初始化记录</a></p></section>
<section id="refit"><h2>raw113：固定7父 / 14子，只重拟合外观与opacity</h2><p>保持父行清单、切面、子均值、协方差、旋转、尺度与高阶SH固定。只用同帧冻结原模型RGB做200步MSE蒸馏，更新子SH DC与opacity；观测mask仅用于评价，没有训练分类器。r1完成200步后在验收路径失败，r2精确恢复该检查点并补齐验收，本次更新0步，旧现场保留。</p><div class="two"><div class="panel"><h3>两域假设匹配比例</h3><table><thead><tr><th>透射上下文</th><th>初始化</th><th>200步后</th><th>变化</th></tr></thead><tbody>@@MATCH_ROWS@@</tbody></table><p class="caption">分子：target假设侧在body的贡献＋road假设侧在安全观测道路的贡献；分母：14个子高斯在这两个域的总 alpha×T。它是条件贡献比例，<strong>不是语义accuracy</strong>，也不覆盖全图。</p></div><div class="callout"><p>Actor 保留时比例提高；Actor 移除、旧static支撑暴露后比例下降。这次固定几何的辐射参数优化，没有证明遮挡消失下的可分离性改善。两侧命名始终是几何假设；代码的 <code>gt</code> 表示切平面“大于”侧，不是ground truth。</p></div></div><div class="toolbar"><label>对照上下文 <select id="contextSelect"><option value="actor_removed" selected>Actor 已移除</option><option value="actor_present">Actor 保留</option></select></label><span class="caption">原模型参考 / 初始化 / 200步后，同一帧同一区域</span></div><div class="compare">@@REFIT_IMAGES@@</div><details open><summary>局部保真：相对各自冻结原模型参考的RGB变化</summary><div class="table-wrap"><table><thead><tr><th>上下文</th><th>固定观测域</th><th>平均通道绝对变化：前→后</th><th>变化&gt;1/255像素：前→后</th></tr></thead><tbody>@@LOCAL_ROWS@@</tbody></table></div><p class="caption">这是对冻结模型的复现误差，不是对真实删除背景GT的误差。道路局部更接近教师，并不抵消Actor移除后body域的变化增大。</p></details><h3 style="margin-top:25px">200步后条件删除：每次仍做原生重渲染</h3><div class="compare">@@CONTROL_IMAGES@@</div><div class="table-wrap panel"><table><thead><tr><th>Actor移除后，再做条件删除</th><th>body平均RGB变化</th><th>实测安全道路平均RGB变化</th><th>道路变化&gt;1/255像素</th></tr></thead><tbody>@@ROUTING_ROWS@@</tbody></table><p class="caption">相对本阶段“两侧保留”结果。这里检验路由副作用，不能把删除假设侧后的颜色变化解释为正确去车或干净道路。</p></div><p class="caption">保存/恢复：DC、opacity、optimizer、几何、高阶SH、四类RNG、RGB、alpha共11项通过；恢复RGB最大差0，alpha逐位相同。<a href="refit_raw113_r2/restore_verification.json">恢复原始JSON</a> · <a href="refit_raw113_r2/summary.json">完整refit记录</a> · <a href="refit_raw113_r2/manifest.json">配置与方法合同</a></p></section>
<section id="next"><h2>本轮改变的是表示检验，不是分类器调参</h2><div class="two"><div class="panel"><h3>实际变化</h3><p>从整行选择/改色，推进到实测双域支撑检查、双侧截断初始化、固定子几何的外观蒸馏和条件删除。7父/14子、200步、学习率与候选顺序固定；没有搜索3σ阈值、放宽尾质量门限或扩大科研范围。</p><p>原static、Actor10K与环境检查点没有写回；两个子侧均保留。分类器0步，原模型0步，shadow解决方案未执行。</p></div><div class="panel"><h3>证据边界</h3><p>49帧是已有训练时刻，三锚点不是独立未见测试集。预测语义、body、道路与阴影证据没有完整人工GT。帧状态、持久行状态、颜色冲突、双域贡献和shadow effect分别保存，缺测不记负例。</p><p>阴影应作为road receiver上的目标关联effect，保留road语义。它与目标子高斯归属分开；本轮没有用该探针声称阴影已经修复。</p></div></div><ol><li><strong>先做几何校正或重表面建模：</strong>用真实深度与可见表面建立目标canonical和道路支撑，追踪旧父行到新子行的谱系。</li><li><strong>几何适合时才双侧split：</strong>不能仅凭两个屏幕域都有贡献就执行切分；保留两侧与未知状态。</li><li><strong>再验保真与编辑后的路由：</strong>无编辑局部RGB/alpha/深度，Actor移除后的新透射率，以及car侧编辑对道路的损伤分别通过。</li><li><strong>shadow4独立验证：</strong>道路receiver保留语义，目标effect单列证据与干预，不与car分类合并。</li></ol></section>
<footer><p>本页生成器只做本地CPU复核：@@PNG_COUNT@@张PNG与对应浮点显示编码逐像素一致，@@METRIC_COUNT@@组局部指标从浮点重放通过，4组两域比例重放通过，恢复JSON全部检查为true。没有在生成报告时调用GPU或SSH；页面浏览器人工验收尚未执行。</p><p><a href="report_validation.json">本页验证记录</a> · 场景 @@CTX@@ · human_verdict=null · failure_ledger_delta=none</p></footer></main>
<script type="application/json" id="reportData">@@DATA@@</script><script>
(()=>{'use strict';
const D=JSON.parse(document.getElementById('reportData').textContent);
const set=(group,slot,path,roi)=>{
 document.querySelectorAll(`[data-image="${group}:${slot}"]`).forEach(i=>{i.src=path});
 document.querySelectorAll(`[data-original="${group}:${slot}"]`).forEach(a=>{a.href=path});
 document.querySelectorAll(`[data-crop="${group}:${slot}"]`).forEach(box=>{
  const[x,y,w,h]=roi;box.style.aspectRatio=`${w} / ${h}`;const i=box.querySelector('img');
  i.style.width=(480/w*100)+'%';i.style.left=(-x/w*100)+'%';i.style.top=(-y/h*100)+'%';
 });
};
function updateRaw(){
 const raw=document.getElementById('rawSelect').value,p=raw.padStart(3,'0');
 for(const s of ['intact','actor_removed','r2_removed'])set('state',s,`probe_anchors_r1/${p}/${s}.png`,D.rois[raw]);
 document.getElementById('stateMeta').textContent=`raw${raw} · 相同相机与原环境 · 原图和局部同步`;
 document.getElementById('stateEvidence').replaceChildren();
 for(const s of ['intact','actor_removed','r2_removed'])for(const[n,f]of[['浮点',`${s}_evidence.npz`],['高斯分量',`${s}_gaussian.png`],['环境分量',`${s}_environment.png`]]){
  const a=document.createElement('a');a.href=`probe_anchors_r1/${p}/${f}`;a.textContent=s+' / '+n;a.target='_blank';document.getElementById('stateEvidence').append(a);
 }
}
function updateContext(){
 const c=document.getElementById('contextSelect').value;
 set('refit','reference',`refit_raw113_r2/references/${c}.png`,D.refitRoi);
 set('refit','before',`refit_raw113_r1/before/${c}/both_retained.png`,D.refitRoi);
 set('refit','after',`refit_raw113_r2/after/${c}/both_retained.png`,D.refitRoi);
 for(const mode of ['whole_parent_remove','gt_target_hypothesis_child_remove','leq_road_hypothesis_child_remove'])set('control',mode,`refit_raw113_r2/after/${c}/${mode}.png`,D.refitRoi);
 document.getElementById('routingContext').textContent=D.contextLabels[c];
 const tbody=document.getElementById('routingBody');tbody.replaceChildren();
 for(const row of D.routing[c]){const tr=document.createElement('tr');row.forEach((value,i)=>{const td=document.createElement('td');td.textContent=(i===1||i===2)?value.toFixed(6):String(value);tr.append(td)});tbody.append(tr)}
}
document.getElementById('rawSelect').addEventListener('change',updateRaw);
document.getElementById('contextSelect').addEventListener('change',updateContext);updateRaw();updateContext();})();
</script></body></html>'''
    page = page.replace("<th>Actor移除后，再做条件删除</th>", '<th><span id="routingContext">Actor 已移除</span>后，再做条件删除</th>')
    page = page.replace("<tbody>@@ROUTING_ROWS@@</tbody>", '<tbody id="routingBody">@@ROUTING_ROWS@@</tbody>')
    replacements = {"SCENE": scene_link, "STATE_IMAGES": state_images, "GEOMETRY_ROWS": geometry_rows,
        "MATCH_ROWS": match_rows, "REFIT_IMAGES": refit_images, "LOCAL_ROWS": "".join(local_rows),
        "CONTROL_IMAGES": control_images, "ROUTING_ROWS": "".join(routing_rows), "PNG_COUNT": str(len(inputs.png_checks)),
        "METRIC_COUNT": str(metric_replays), "CTX": CTX, "DATA": json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")}
    for key, value in replacements.items():
        page = page.replace("@@" + key + "@@", value)
    require("@@" not in page, "Unfilled report template token")
    parser = References()
    parser.feed(page)
    link_paths = sorted(set(parser.paths + image_paths + ["report_validation.json"]))
    for relative in link_paths:
        if relative != "report_validation.json":
            require((base / relative).resolve().is_file(), "Report reference missing: " + relative)
    javascript_checked = False
    node = shutil.which("node")
    if node:
        script = re.findall(r"<script>([\s\S]*?)</script>", page)[0]
        check = subprocess.run([node, "--check"], input=script, text=True, encoding="utf8", capture_output=True)
        require(check.returncode == 0, "Report JavaScript syntax failed: " + check.stderr[:500])
        javascript_checked = True
    inputs.unchanged()
    validation = dict(schema="v43-editable-ownership-report-validation-v1", context=CTX, passed=True,
        completed_summaries=["probe_anchors_r1/summary.json", "support_geometry_r1/summary.json", "refit_raw113_r2/summary.json"],
        incomplete_r1_summary_read=False, r1_before_accepted_via_completed_r2_source_provenance=True,
        stable_read_file_count=len(inputs.reads), png_float_checks=inputs.png_checks, png_float_count=len(inputs.png_checks),
        display_equation="round(clip(float_RGB,0,1)*255), RGB uint8, exact pixels",
        native_environment_decomposition_max_abs_error=max_decomposition_error,
        original_environment_identical_across_three_states=True, native_three_states_fresh_adjoint_closure_passed=True,
        local_metric_replay_groups=metric_replays, local_metric_replay_passed=True, side_match_replays=purity_replays,
        restore_checks=restore["checks"], restore_json_all_passed=True, restored_RGB_max_abs_error=restore["RGB_max_abs_error"],
        child_refit_updates_total=200, child_refit_updates_this_restoration=0, classifier_training_updates=0,
        original_model_training_updates=0, quality_accepted=False, human_verdict=None,
        old_R2_and_unknown_white_preserved=True, linked_files_checked=link_paths,
        architecture_reference="../../architecture.svg", crop_method="CSS views of the exact source PNG; no new image generation",
        javascript_syntax_checked=javascript_checked,
        report_execution=dict(device="CPU", GPU_calls=0, SSH_calls=0, source_files_modified=False, browserQA=False))
    (base / "index.html").write_text(page, encoding="utf8")
    (base / "report_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return dict(report=str(base / "index.html"), passed=True, PNG_float_count=len(inputs.png_checks),
                local_metric_replay_groups=metric_replays, restore_check_count=len(restore["checks"]), browserQA=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = args.output or args.root / "outputs/V43_Ownership_20261027/gpu_v43/editable_ownership_20261030"
    print(json.dumps(run(args.root, base), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

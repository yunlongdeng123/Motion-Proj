"""生成离线深色交互报告与不依赖推理图像包的轻量摘要。"""
from __future__ import annotations
import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import shutil

ARMS = [
    {"id": "full768_T0", "label": "整帧 768 · 注意力 LoRA"},
    {"id": "full768_base", "label": "整帧 768 · 基础模型"},
    {"id": "crop768_T0", "label": "下部裁片 768 · 注意力 LoRA"},
    {"id": "full1152_T0", "label": "整帧 1152 · 注意力 LoRA"},
]
LABELS = {
    "visible_scene081_raw000_v0": "081 / 000 · 沥青孔洞",
    "visible_scene081_raw000_v1": "081 / 000 · 标线孔洞",
    "visible_scene081_raw050_v0": "081 / 050 · 沥青孔洞",
    "visible_scene081_raw050_v1": "081 / 050 · 标线孔洞",
    "scene065_raw086": "065 / 086 · 真实遮挡",
    "scene065_raw110": "065 / 110 · 真实遮挡",
}
DOMAINS = {"synthetic_certified_hidden_target": "合成隐藏道路 · 证书内统计",
           "real_known_visible_control": "真实场景 · 已知可见道路控制"}
esc = html.escape


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pin(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def f(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def table(headers, rows, caption=None):
    h = "".join("<th>" + esc(str(value)) + "</th>" for value in headers)
    r = "".join("<tr>" + "".join("<td>" + esc(str(value)) + "</td>" for value in row) + "</tr>" for row in rows)
    return '<div class="scroll"><table>' + ("<caption>" + esc(caption) + "</caption>" if caption else "") + "<thead><tr>" + h + "</tr></thead><tbody>" + r + "</tbody></table></div>"


def outline_assets(out, cases):
    """仅绘制明确标注的范围图层，不改实验PNG。"""
    import numpy as np
    from PIL import Image, ImageFilter
    for case in cases:
        if not case.get("masks"):
            continue
        result = None
        for name, color in [("H", (78, 165, 255)), ("W", (91, 225, 181)), ("K", (244, 190, 99))]:
            mask = Image.open(out / case["masks"][name]).convert("L")
            edge = np.asarray(mask.filter(ImageFilter.MaxFilter(3))) != np.asarray(mask.filter(ImageFilter.MinFilter(3)))
            rgba = np.zeros((mask.height, mask.width, 4), np.uint8)
            rgba[..., :3] = color
            rgba[..., 3][edge] = 230
            layer = Image.fromarray(rgba)
            result = layer if result is None else Image.alpha_composite(result, layer)
        target = out / "assets/inputs" / (case["id"] + "_outline_roi.png")
        result.save(target)
        case["outline_roi"] = target.relative_to(out).as_posix()


def normalize_assets(value):
    cases = copy.deepcopy(value["cases"])
    for case in cases:
        case["label"] = LABELS[case["id"]]
        case["reference"] = {key: "assets/" + path for key, path in case["reference"].items()}
        case["masks"] = {key: "assets/" + path for key, path in case["masks"].items()}
        for job in case["jobs"]:
            for key in ["raw_model", "full_preview", "raw_roi", "final_roi"]:
                job[key] = "assets/" + job[key]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--vae-precision", type=Path)
    parser.add_argument("--narrative", type=Path)
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--supervision-validation", type=Path)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--accounting", type=Path)
    parser.add_argument("--prototype", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    for name in ["architecture.svg", "report.css", "report.js"]:
        shutil.copyfile(source / name, out / name)
    predictions = read(args.predictions) if args.predictions else None
    comparison = read(args.comparison) if args.comparison else None
    precision = read(args.vae_precision) if args.vae_precision else None
    narrative = read(args.narrative) if args.narrative else {}
    scope = read(args.scope) if args.scope else None
    transfer = read(args.transfer) if args.transfer else None
    supervision_validation = read(args.supervision_validation) if args.supervision_validation else None
    accounting = read(args.accounting) if args.accounting else None
    if transfer:
        assert transfer["complete"] and transfer["CPU_only"] and transfer["GPU_calls"] == 0
        assert not transfer["hidden_GT_RGB_opened"]
    if scope:
        assert scope["complete"] and scope["package_count"] == 20
    if not args.prototype:
        assert predictions and predictions["complete"] and predictions["completed_diffusion_calls"] == 48
        assert comparison and comparison["complete"] and comparison.get("mode") == "FROZEN_PREDICTION_CPU_SCORING"
        assert (out / "assets/asset_manifest.json").is_file()
    assets = read(out / "assets/asset_manifest.json") if (out / "assets/asset_manifest.json").exists() else None
    cases = normalize_assets(assets) if assets else [dict(id=sid, label=label, domain="real_unlabelled" if sid.startswith("scene065") else "synthetic_visible_2d", jobs=[]) for sid, label in LABELS.items()]
    outline_assets(out, cases)
    done = predictions["completed_diffusion_calls"] if predictions else None
    replays = [job for job in (predictions or {}).get("jobs", []) if job["legacy_exact_replay_expected"]]
    replay_passed = sum(all(job["legacy_replay_exact"].values()) for job in replays)
    data = dict(schema="road_2d_teacher_GPU_HTML.v1", complete=not args.prototype,
                title=narrative.get("title", "二维道路教师：颜色、细节与输入视野"),
                lead=narrative.get("lead", "固定四组教师、两个 seed、六个任务。原始候选、写回合成与两种统计网格分别呈现，评价不自动选择默认方法。"),
                findings=narrative.get("findings", []), arms=ARMS, cases=cases,
                metrics=(comparison or {}).get("macro_equal_base_frames", []),
                source_evidence={key: pin(path) for key, path in {"predictions": args.predictions,
                                 "comparison": args.comparison, "vae_precision": args.vae_precision,
                                 "scope": args.scope, "transfer": args.transfer,
                                 "supervision_validation": args.supervision_validation,
                                 "execution_accounting": args.accounting}.items() if path},
                execution=dict(diffusion_calls=done, historical_replays_passed=replay_passed,
                               main_VAE_controls=len((predictions or {}).get("VAE_controls", [])),
                               main_wall_seconds=(predictions or {}).get("wall_seconds"),
                               peak_allocated_GiB=(predictions or {}).get("peak_GPU_allocated_bytes", 0) / 1024**3 if predictions else None,
                               training_updates=0, three_dimensional_changes=0,
                               extra_VAE_roundtrips=(precision or {}).get("completed_VAE_roundtrips", 0),
                               extra_VAE_wall_seconds=(precision or {}).get("wall_seconds")),
                human_verdict=None, problem_solved=None, automatic_winner_selected=False)
    if transfer and scope:
        data["supervision"] = dict(package_count=scope["package_count"],
            source_level_identity_checks=scope["source_level_identity_checks"],
            baseline_prior_exact_checks=scope["baseline_seed101_prior_exact_checks"],
            cpu_validation=supervision_validation,
            transfer_aggregate=transfer["aggregate"],
            fixed_tile_support=[{key: row[key] for key in ["raw", "role", "fixed_common_tile_count", "status"]}
                                for row in transfer["tile_support"]],
            permission_counts=scope["comparisons"],
            q_is_high_frequency_permission=False, camera_renderer_qualified=False,
            candidate_only=True, training_ready=False, native_3d_calls=0)
    write(out / "report_data.json", data)
    e = data["execution"]
    counts = [(str(done) + "/48" if done is not None else "待接入", "固定扩散推理"),
              (str(replay_passed) + "/6" if predictions else "待接入", "历史重放 · 三种PNG逐位一致"),
              ("0", "训练更新"), ("0", "三维渲染修改")]
    counts_html = '<div class="counts">' + "".join('<div class="count"><strong>' + esc(v) + '</strong><span>' + esc(label) + '</span></div>' for v, label in counts) + "</div>"
    findings = '<div class="findings">' + "".join('<article class="finding"><h3>' + esc(x["title"]) + '</h3><p>' + esc(x["text"]) + '</p></article>' for x in data["findings"]) + "</div>" if data["findings"] else ""
    viewer = '''<section id="inspection"><h2>固定窗口并列审阅</h2>
<div class="toolbar"><label>任务 <select id="case-select" aria-label="选择固定任务"></select></label><div class="segment" id="seed-buttons"><button data-value="101" aria-pressed="true">Seed 101</button><button data-value="202" aria-pressed="false">Seed 202</button></div><label>图像 <select id="view-select"><option value="raw_roi">原始候选 · 固定区域</option><option value="final_roi">写回合成 · 固定区域</option><option value="full_preview">原始候选 · 全帧预览</option></select></label><span class="spacer"></span><label><input id="scope-toggle" type="checkbox">显示范围</label><label>缩放 <input id="zoom" type="range" min="1" max="4" step="0.1" value="1"><span id="zoom-label">1.0×</span></label><button id="reset-view">重置</button></div>
<p class="case-meta" id="case-meta"></p><div class="image-grid" id="image-grid"></div>
<div class="reference-grid"><article class="image-card"><div class="image-heading"><h3 id="reference-title">原始观察 · 同一固定窗口</h3></div><div class="canvas-wrap"><canvas id="observation-canvas" aria-label="原始观察参考"></canvas><div class="placeholder">图像待接入</div></div></article><article class="image-card" id="clean-reference"><div class="image-heading"><h3>合成 clean 参考 · 仅证书内道路作真值</h3></div><div class="canvas-wrap"><canvas id="clean-canvas" aria-label="合成干净参考"></canvas><div class="placeholder">图像待接入</div></div></article></div>
<p class="note" id="view-note"></p><div class="legend"><span><i class="dot" style="background:var(--green)"></i>W · 固定写回许可</span><span><i class="dot" style="background:var(--blue)"></i>H · 输入隐藏范围</span><span><i class="dot" style="background:var(--yellow)"></i>K · 保留对象</span><span>拖动任一图像可同步平移</span></div>
<p class="note">全帧 WebP90 仅供预览；固定区域 PNG 是原图坐标的无损裁片，仍可能来自上游插值。每组的“原始生成 PNG”保留各自生成分辨率。科学指标读取远端原始预测 PNG 与冻结真值源；报告预览不参与度量。</p></section>'''
    headers = ["教师对照", "ΔE76 ↓", "RGB MAE（0–255） ↓", "高通 RMS 比值 · 目标1", "统一3频带偏差 ↓", "Lab a / b 偏置"]
    metrics_interactive = '''<section id="metrics"><h2>颜色与纹理频谱</h2><div class="toolbar"><div class="segment" id="grid-buttons"><button data-value="native" aria-pressed="true">原图同域</button><button data-value="same_view" aria-pressed="false">同采样网格</button></div><span class="spacer"></span><div class="segment" id="domain-buttons"><button data-value="synthetic_certified_hidden_target" aria-pressed="true">合成隐藏道路</button><button data-value="real_known_visible_control" aria-pressed="false">真实可见道路控制</button></div></div><p class="metric-caption" id="metric-caption"></p><div class="scroll"><table><thead><tr>''' + "".join("<th>" + esc(h) + "</th>" for h in headers) + '''</tr></thead><tbody id="metric-body"></tbody></table></div>'''
    metric_notes = '''<p class="note">先汇总两个 seed，再按底图分层等权。4 个合成任务来自 2 张底图；两 seed 的范围不是置信区间。真实两帧仅评已知可见道路，不能给遮挡道路恢复误差。表中指标均评价原始候选；真实控制不是写回合成的复制像素，W外合成仍精确复制。</p><p class="note">高通 σ=2.5 原图像素；RMS 比值接近 1 仍不能证明生成了正确颗粒。频谱主分数是共同三频带的能量比绝对 log₂ 偏差，目标为 0；不跨组比较各自可用带宽的单一总分。三带为 0.025–0.0625、0.0625–0.125、0.125–0.2 周期/原图像素。</p>'''
    chart = '<figure class="chart"><img src="assets/summary/synthetic_color_spectrum.png" alt="合成任务颜色和共同频带偏差，分原图同域与同采样网格"><figcaption class="note">逐任务两 seed 均值与范围；无自动排序或默认方法晋升。</figcaption></figure>' if (out / "assets/summary/synthetic_color_spectrum.png").exists() else ""
    metrics_interactive += metric_notes + chart + "</section>"
    metrics_static = '<section id="metrics"><h2>固定统计结果</h2>'
    for domain, label in DOMAINS.items():
        for grid, grid_label in [("native", "原图同域"), ("same_view", "同采样网格")]:
            selected = [r for r in data["metrics"] if r["domain"] == domain and r["grid"] == grid]
            selected.sort(key=lambda r: [a["id"] for a in ARMS].index(r["arm"]))
            rows = [[next(a["label"] for a in ARMS if a["id"] == r["arm"]), f(r.get("deltaE76_mean")),
                     f(r.get("RGB_MAE_0_255")), f(r.get("HP_RMS_ratio_sigma_2p5_native")),
                     f(r.get("PSD_common3_mean_abs_log2_energy_ratio")),
                     f(r.get("mean_Lab_bias_a")) + " / " + f(r.get("mean_Lab_bias_b"))] for r in selected]
            metrics_static += "<h3>" + esc(label + " · " + grid_label) + "</h3>" + table(headers, rows or [["待接入"] * 6]) + '<div style="height:16px"></div>'
    metrics_static += metric_notes + chart + "</section>"
    supervision = ""
    if transfer and scope:
        def transfer_group(arm, output, path, band_index):
            values = [row for row in transfer["rows"] if row["arm"] == arm and row["role"] == "C"
                      and row["output_wh"] == output and row["path"] == path]
            assert len(values) == 4
            low = sum(row["shared_0p025_to_0p125_energy_retention"] for row in values) / len(values)
            band = sum(row["per_band_energy_retention"][band_index] for row in values) / len(values)
            return f(low * 100, 2) + "%", f(band * 100, 2) + "%"
        area = "exact_single_area"
        history = "historical_PIL_bilinear_uint8_native_then_float_block4mean"
        transfer_rows = []
        for label, arm, output, path, band, band_label in [
            ("1152源 → 768", "full1152_T0", [768, 512], area, 2, "0.125–0.2"),
            ("1152源 → 480", "full1152_T0", [480, 320], area, 1, "0.0625–0.125"),
            ("1152源 → 旧链480", "full1152_T0", [480, 320], history, 1, "0.0625–0.125"),
            ("768源 → 480", "full768_T0", [480, 320], area, 1, "0.0625–0.125"),
            ("768源 → 旧链480", "full768_T0", [480, 320], history, 1, "0.0625–0.125")]:
            low, high = transfer_group(arm, output, path, band)
            transfer_rows.append([label, "单次面积采样" if path == area else "双线性原图＋4×4均值",
                                  low, high + "（" + band_label + "）"])
        support_rows = []
        for row in scope["comparisons"]:
            if row["seed"] != 101:
                continue
            count = row["two_dimensional_permission_counts"]
            support_rows.append([str(row["raw"]) + " / " + "×".join(map(str, row["output_wh"])),
                " / ".join(str(count["C"][key]) for key in ["baseline", "highres", "common"]),
                " / ".join(str(count["S"][key]) for key in ["baseline", "highres", "common"])])
        supervision = '<section id="supervision"><h2>二维监督：保留源级，再独立采样</h2>'
        supervision += '<p>已构建20个固定配对候选包：raw86/110、两个seed、整帧768与1152教师。A取真实可见道路，C为孔洞伪教师，S为目标阴影伪教师。源级8项RGB逐值检查通过；1152原值档保留生成源带宽，768与480各自从原源图直接面积采样，不经中间档串联。</p>'
        supervision += '<p class="metric-caption">下表只度量已有生成信号的功率传递，不度量真实沥青细节或跨教师优劣。C域使用raw86/110各22/59个固定80×80原图窗口，步长40、可重叠；所有方法、seed、输出档与旧链共用支持。S有合法像素，但无完整80×80共同窗口，本项不量化。</p>'
        supervision += table(["生成源与输出", "RGB路径", "共同0.025–0.125频带能量保留", "较高频带能量保留（频带）"], transfer_rows)
        supervision += '<p class="note">频率单位为周期/原图像素，保留率相对各自教师源，不是RMS幅度或真实细节百分比。1152→768在较高0.125–0.2频带仅保留59.49%，不能由低频主导的95.82%共同频带均值声称完整保真。源级1152、768和480网格的轴向Nyquist分别为0.3、0.2和0.125；超出目标Nyquist的源能量不能作为独立目标频率保留，混叠不等于细节保存。</p>'
        supervision += '<details><summary>C / S共同支持域分账</summary>' + table(["原帧 / 输出尺寸", "C：基线 / 高分辨率 / 共同", "S：基线 / 高分辨率 / 共同"], support_rows)
        supervision += '<p class="note">A域不变；两seed的此项资格计数相同。C/S既有新增也有剔除，不能把分母变化写成质量改善。共同域目前只是二维来源资格，未来训练还须与同一固定三维资格取交集；完整单侧计数保留在构建receipt。</p></details>'
        if (out / "assets/summary/generated_signal_transfer.png").exists():
            supervision += '<details><summary>信号传递总图</summary><figure class="chart"><img src="assets/summary/generated_signal_transfer.png" alt="固定共同支持窗口上的生成源到二维监督采样的信号传递"><figcaption class="note">不同可用总带宽的能量不等同于相同频带比较；主要结论使用上表共同物理频带。</figcaption></figure></details>'
        supervision += '<p class="callout">q_2d仅表示来源与全滤波支持的二值资格，不是正确率或世界高频材质许可。相机K仍为调用方声明，未通过原生栅格化资格。20包均为candidate_only、training_ready=false；不接入三维训练，不扩张父资产高频许可。</p>'
        supervision += '<p class="note">本地/远端各8项合同测试、6个W×3视图及42项输出支持检查通过；8项源级逐值一致、4项旧基线包exact。包保留原PNG、model valid与显式视图映射；这些CPU验证不证明真实遮挡道路正确，也不证明三维渲染能力。</p></section>'
    vae_rows = []
    for row in (predictions or {}).get("VAE_controls", []):
        metric = row["VAE_increment_over_resampling"]
        vae_rows.append([row["id"].replace("scene065_", ""), f(metric["rgb_mae"] * 255),
                         f(metric["signed_rgb_bias"][1] * 255), f(metric["highpass_rms_ratio"])])
    precision_rows = []
    if precision:
        for dtype in ["float32", "float16"]:
            rows = [r for r in precision["rows"] if r["dtype"] == dtype]
            green = [r["increment_over_resampling"]["signed_rgb_bias"][1] * 255 for r in rows]
            mae = [r["increment_over_resampling"]["rgb_mae"] * 255 for r in rows]
            precision_rows.append([dtype.upper(), len(rows), f(min(green)) + " ～ " + f(max(green)),
                                   f(min(mae)) + " ～ " + f(max(mae))])
    vae = '<section id="vae"><h2>VAE 重建与数值精度</h2><p class="metric-caption">只用合法实拍观察与已知可见道路，latent mode 重建不运行扩散 UNet；实际扩散条件采用 latent sample，二者不等价。下表为相对同尺度重采样参考的增量，RGB与绿色通道均以 0–255 表示。</p>'
    vae += table(["主合同控制", "RGB MAE（0–255）", "绿色通道偏置（0–255）", "高通 RMS 比值"], vae_rows or [["待接入"] * 4])
    if precision:
        vae += '<h3 style="margin-top:20px">独立追加：运算精度诊断</h3><p class="note">同一 fp16 预训练权重源提升为 FP32 或使用 FP16 运算，不等于取得原始全精度模型。4 张合法观察 × 2 精度，另有 1 次 BF16 精确重放门；不加入四组教师均值。</p>'
        vae += table(["运算精度", "观察数", "绿色偏置范围（0–255）", "RGB MAE范围（0–255）"], precision_rows)
        vae += '<p class="callout">FP32 / FP16 仍保留约 −2.76～−3.20 /255 的绿色偏置，未降低原 BF16 控制误差；当前证据不支持把 BF16 作为主要原因。该诊断不等价于完整扩散链路的精度消融。</p>'
    vae += "</section>"
    interface_text = narrative.get("interface", "二维监督接口保持为独立验收项；本轮没有将新教师图接入三维训练，也没有修改三维资产、投影或渲染。")
    protocol = '<section id="protocol"><h2>方法边界与执行账</h2><div class="two-col"><article class="box"><h3>固定生成条件</h3><p>SD1.5 Inpainting；注意力 LoRA 为既有第600步权重。DDIM 20步、CFG 3、eta 0、strength 1；seed 101 / 202；正向提示词与空负向提示词固定。</p><p>训练视野是原图下部 [0,600,1920,1280]；768×272 裁片与整帧768×512均为每模型像素2.5个原图像素。视野控制没有改变道路像素倍率。</p></article><article class="box"><h3>二维接口与范围</h3><p>' + esc(interface_text) + '</p><p>W/K许可保持冻结。更大输入隐藏范围没有扩大写回区；保留对象和许可外像素保持精确复制。没有默认资产替换或自动选优。</p></article></div>'
    ledger_rows = [["主合同扩散", str(done) + " 次" if done is not None else "待接入"],
                   ["主合同独立VAE", str(e["main_VAE_controls"]) + " 次 encode/decode"],
                   ["追加VAE精度诊断", str(e["extra_VAE_roundtrips"]) + " 次（1工程重放＋8诊断）" if precision else "未登记"],
                   ["主队列墙钟", f(e["main_wall_seconds"], 2) + " 秒；包含加载与主VAE控制"],
                   ["主队列模型加载", f((predictions or {}).get("model_load_seconds"), 3) + " 秒；包含在主队列墙钟内"],
                   ["主队列峰值 PyTorch allocated", f(e["peak_allocated_GiB"], 3) + " GiB；不使用 nvidia-smi reserved 代替"],
                   ["追加精度诊断墙钟", f(e["extra_VAE_wall_seconds"], 3) + " 秒" if precision else "—"],
                   ["训练 / 三维修改 / 自动选优", "0 / 0 / 否"],
                   ["历史重放", f"{replay_passed}/6；raw_model、raw_native、final_native逐位一致" if predictions else "待接入"]]
    if scope:
        ledger_rows.append(["二维配对监督包", f"20包；CPU构建{scope['elapsed_seconds']:.3f}秒，GPU 0，三维前向0"])
    if accounting:
        total = accounting["total"]
        ledger_rows.append(["独立GPU账合计", f"{total['diffusion_calls']}次扩散 / {total['VAE_only_roundtrips']}次独立VAE；失败{total['failed_calls']}，训练{total['training_updates']}"])
    protocol += table(["项目", "实际记录"], ledger_rows)
    protocol += '<p class="note">小时级新训练仍需先约十分钟真实 smoke，覆盖保存、评价、完整恢复及下一阶段入口；本轮没有启动新训练。不同画幅计算量不同，同数字seed也不构成逐像素相同噪声。</p></section>'
    evidence = []
    for path in sorted((out / "evidence").glob("*")) if (out / "evidence").exists() else []:
        if path.is_file():
            evidence.append('<li><a href="' + esc(path.relative_to(out).as_posix()) + '">' + esc(path.name) + "</a></li>")
    evidence_html = '<details><summary>证据与复核入口</summary><ul class="evidence-list">' + "".join(evidence) + '</ul><p class="note">完整原始raw_native / final_native PNG保留远端 GPU_pilot_r1。报告中的生成PNG为原始分辨率无损副本；其他图像为明确标注的预览或固定窗口裁片。</p></details>'
    shell = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title><link rel="stylesheet" href="report.css"></head><body><main><div class="eyebrow">2D TEACHER · 2026-11-19 · FIXED GPU PILOT</div><h1>__TITLE__</h1><p class="lead">__LEAD__</p>__COUNTS__<nav><a href="#__FIRST_ANCHOR__">__FIRST_LABEL__</a><a href="#metrics">定量结果</a>__SUPERVISION_NAV__<a href="#vae">VAE控制</a><a href="#protocol">方法与执行账</a></nav>__FINDINGS__<figure class="architecture"><img src="architecture.svg" alt="二维教师四组固定对照、源级到监督采样、独立VAE控制、固定域评价与写回合成的数据流"></figure>__VIEWER____METRICS____SUPERVISION____VAE____PROTOCOL____EVIDENCE__<footer>ROAD-2D-TEACHER-HF-GPU-20261119 · 4个合成任务 / 2张底图 · 真实隐藏道路无GT · human_verdict / problem_solved = null__FOOTER__</footer></main>__SCRIPT__</body></html>'''
    for filename, full in [("index.html", True), ("publication.html", False)]:
        substitutions = dict(TITLE=esc(data["title"]), LEAD=esc(data["lead"]), COUNTS=counts_html, FINDINGS=findings,
                             FIRST_ANCHOR="inspection" if full else "metrics", FIRST_LABEL="原图对照" if full else "固定统计", VIEWER=viewer if full else "",
                             METRICS=metrics_interactive if full else metrics_static, VAE=vae, PROTOCOL=protocol,
                             SUPERVISION=supervision, SUPERVISION_NAV='<a href="#supervision">二维监督</a>' if supervision else "",
                             EVIDENCE=evidence_html if full else '<p class="note">此为轻量静态摘要。完整交互图、原始生成PNG和固定窗口图保留在交付文件包，推理原PNG保留远端。</p>',
                             FOOTER=' · <a href="publication.html">轻量静态摘要</a>' if full else "",
                             SCRIPT='<script>window.REPORT_DATA=' + json.dumps(data, ensure_ascii=False).replace("</", "<\\/") + ';</script><script src="report.js"></script>' if full else "")
        text = shell
        for key, value in substitutions.items():
            text = text.replace("__" + key + "__", value)
        if not full:
            text = text.replace('<a href="#metrics">固定统计</a>', '')
        (out / filename).write_text(text, encoding="utf-8")
    write(out / "report_build_receipt.json", dict(schema="road_2d_teacher_GPU_report_build.v1", complete=True,
          prototype=args.prototype, source_files={p.name: pin(p) for p in [Path(__file__), source / "report.js", source / "report.css", source / "architecture.svg"]},
          scientific_inputs=data["source_evidence"], image_cases=len(cases), image_jobs=sum(len(c.get("jobs", [])) for c in cases),
          preview_images_are_metric_sources=False, automatic_winner_selected=False, old_CPU_report_modified=False))
    print(json.dumps(dict(output=str(out), prototype=args.prototype, jobs=sum(len(c.get("jobs", [])) for c in cases)), ensure_ascii=False))


if __name__ == "__main__":
    main()

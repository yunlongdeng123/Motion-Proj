"""生成完整本地图像报告与不依赖60原图的轻量Git出版页。"""
from pathlib import Path
import argparse
import csv
import html
import json
import math
import shutil

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def num(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def metric_rows(path):
    if path is None or not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        for key in list(row):
            if key not in ("case_id", "base_frame", "domain", "prompt_id", "seed", "grid"):
                row[key] = num(row[key])
    return rows


def aggregate_html(summary):
    if not summary:
        return '<p class="muted">固定60候选的独立统计待合并；没有新提示词胜出结论。</p>'
    rows = []
    for r in summary["prompt_distributions"]:
        if r["grid"] != "native":
            continue
        metric = r["metrics"]
        label = "历史提示词" if r["prompt_id"] == "legacy" else "材质匹配提示词"
        domain = "合成同域" if r["domain"] == "synthetic_certified_hidden_target" else "真实raw可见保持"
        values = [domain, label]
        for field in ("deltaE76_mean", "RGB_MAE_0_255", "PSD_common3_abs_log2_energy_ratio"):
            d = metric[field]
            values.append("—" if d["mean"] is None else f'{d["mean"]:.3f} [{d["minimum"]:.3f}, {d["maximum"]:.3f}]')
        rows.append("<tr>" + "".join("<td>"+html.escape(v)+"</td>" for v in values) + "</tr>")
    return '<div class="tablewrap"><table><thead><tr><th>评价域</th><th>提示词</th><th>ΔE76 均值 [范围]</th><th>RGB MAE（0–255）</th><th>PSD三带偏差</th></tr></thead><tbody>'+"".join(rows)+'</tbody></table></div>'


def findings_html(summary):
    if not summary:
        return '<p class="notice">统计待合并，不能判断新提示词效果。</p>'
    distribution = {r["prompt_id"]: r["metrics"] for r in summary["prompt_distributions"]
                    if r["grid"] == "native" and r["domain"] == "synthetic_certified_hidden_target"}
    paired = [r for r in summary["paired_prompt_deltas"] if r["grid"] == "native" and r["domain"] == "synthetic_certified_hidden_target"]
    assert len(paired) == 20 and set(distribution) == {"legacy", "material_match_v1"}
    old, new = distribution["legacy"], distribution["material_match_v1"]
    color, spectrum = "deltaE76_mean", "PSD_common3_abs_log2_energy_ratio"
    cwin = sum(r[color+"_delta"] < 0 for r in paired)
    swin = sum(r[spectrum+"_delta"] < 0 for r in paired)
    cards = [
        ("颜色均值基本持平", f'合成同域ΔE76 {old[color]["mean"]:.4f} → {new[color]["mean"]:.4f}；新词仅{cwin}/20个任务-seed配对更低，没有稳定整体颜色收益。'),
        ("频谱均值收益局限于050_v1", f'三带偏差 {old[spectrum]["mean"]:.4f} → {new[spectrum]["mean"]:.4f}，仅{swin}/20配对更低。四任务均值中只有050_v1改善，不能概括为颗粒全面恢复。'),
        ("Seed散度没有降低", f'五seed底图等权均值的总体标准差：ΔE76 {old[color]["population_std"]:.4f} → {new[color]["population_std"]:.4f}；PSD {old[spectrum]["population_std"]:.4f} → {new[spectrum]["population_std"]:.4f}。'),
        ("继续保留历史提示词", 'P1不默认晋升，本轮不再追加改词。raw ROI仍需逐例核对新增白段、标线延伸和断裂；结合W与final复核前，不把观察当成错误率或真实隐藏GT。'),
    ]
    return '<h2>本轮实质结论</h2><div class="summarycards">'+''.join('<article class="card"><h3>'+html.escape(title)+'</h3><p>'+html.escape(text)+'</p></article>' for title,text in cards)+'</div>'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--summary", type=Path)
    p.add_argument("--per-seed", type=Path)
    p.add_argument("--asset-base", default="assets")
    p.add_argument("--architecture", type=Path, default=HERE.parent/"architecture.svg")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assets = read(a.assets/"asset_manifest.json")
    assert assets["schema"] == "road_2d_teacher_prompt_report_assets.v1" and assets["complete"]
    assert assets["raw_model_original_count"] == assets["candidate_count"] == 60
    assert len(assets["cases"]) == 6 and all(len(c["jobs"]) == 10 for c in assets["cases"])
    summary = read(a.summary) if a.summary else None
    if summary:
        assert summary["schema"] == "road_2d_teacher_prompt_seed_review.v1" and summary["complete"]
        assert not summary["automatic_winner_selected"] and not summary["automatic_seed_selection"]
        assert summary["mode"] == "FROZEN_PREDICTION_CPU_SCORING"
    per_seed = metric_rows(a.per_seed or (a.summary.parent/"per_seed.csv" if a.summary else None))
    css, js = (HERE/"report.css").read_text(encoding="utf-8"), (HERE/"report.js").read_text(encoding="utf-8")
    css += '\n.architecture svg{display:block;width:100%;height:auto;max-height:300px}@media(max-width:800px){.architecture svg{max-height:none}}'
    architecture = a.architecture.read_text(encoding="utf-8")
    prompts = assets["prompts"]
    data = dict(asset_base=a.asset_base, cases=assets["cases"], seeds=assets["seeds"], prompts=prompts, summary=summary, per_seed=per_seed)
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    escape = html.escape
    prompt_cards = ''.join('<article class="card"><h3>'+('历史提示词' if r['id']=='legacy' else '材质匹配提示词 · 待人工配对审阅')+'</h3><p class="prompttext" lang="en">'+escape(r['text'])+'</p></article>' for r in prompts)
    accounting = '<div class="badges"><span class="badge good">默认方法已采纳 · 当前审阅样例</span><span class="badge">60 正式 + 4 smoke = 64 calls</span><span class="badge">GPU 32GB · 4 workers</span><span class="badge">12 次历史1152三图精确重放</span><span class="badge warm">实例按用户要求保持开机</span></div>'
    execution_path, smoke_path = HERE.parent/"evidence/parallel_receipt.json", HERE.parent/"evidence/smoke_receipt.json"
    if execution_path.exists() and smoke_path.exists():
        execution, smoke = read(execution_path), read(smoke_path)
        assert execution["complete"] and execution["formal_completed_diffusion_calls"] == 60 and execution["total_smoke_plus_formal_diffusion_calls"] == 64
        accounting += f'<p class="sub">正式4×15并行墙钟 {execution["wall_seconds"]:.2f}s；真实4次smoke {smoke["wall_seconds"]:.2f}s。四次smoke开销单列，不进入正式60统计。</p>'
    limits = '<p class="notice">本轮同时调整颜色、纹理和标线三个词段，只能评价整体prompt，不能因果归因gray或extend某一个词。五个seed不增加场景或独立真值；四个合成任务来自两张底图，真实隐藏道路无GT。没有best-of-N选图或三维渲染修改。</p>'
    head = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>1152教师 · 提示词与五seed对照</title><style>'+css+'</style></head><body>'
    intro = ('<h1>1152教师：新提示词未带来稳定整体收益</h1><p class="lead">整帧1152＋注意力LoRA T0保持默认，继续使用历史提示词。本轮60个正式候选全部保留；P1未默认晋升，没有最佳seed选择，也不再追加改词。</p>' if summary else '<h1>1152教师：提示词与五seed配对</h1><p class="lead">60个正式候选已冻结；统计待合并，暂不判断新提示词效果。</p>')+accounting+'<div class="architecture">'+architecture+'</div>'
    viewer = '''<h2>全部候选 · 同case、同seed左右配对</h2><div class="controls"><label for="case">任务</label><select id="case"></select><label>Seed</label><div class="seedbar" id="seeds"></div><span class="spacer"></span><label for="view">视图</label><select id="view"><option value="raw_roi">原始生成 · 固定ROI</option><option value="final_roi">硬回填 · 固定ROI</option><option value="raw_model">1152原始模型PNG</option><option value="full_final_preview">全帧回填预览 · WebP</option></select><button id="zoom" type="button">1:1 原像素</button></div><p id="case-note" class="sub"></p><p id="view-note" class="sub"></p><div class="pair" id="pair">'''
    for i in range(2):
        viewer += f'<article class="candidate"><header><h3 id="candidate{i}"></h3><small id="time{i}"></small></header><div class="imagepane" id="pane{i}"><a id="imglink{i}" target="_blank" rel="noopener"><img id="img{i}" alt=""></a></div><div class="metrics" id="metric{i}"></div><footer><a id="rawlink{i}" target="_blank" rel="noopener">打开1152原始PNG</a></footer></article>'
    viewer += '''</div><div id="case-seed-distribution"></div><details><summary>原始观察与可见角色</summary><div class="referencebox"><div class="referencegrid"><figure><img id="observation" alt="原始观察固定ROI"><figcaption>原始观察</figcaption></figure><figure><img id="roles" alt="冻结角色边界"><figcaption>蓝：H隐藏 · 绿：W写入 · 金：K保护</figcaption></figure><figure id="cleanbox" hidden><img id="clean" alt="冻结后独立clean参照"><figcaption>独立clean参照；仅合法道路证书域用于评价</figcaption></figure></div><div class="links" id="masklinks"></div></div></details>'''
    aggregate = '''<h2>五seed分布 · 不选择最佳图片</h2><div class="aggregate-controls"><select id="domain-select"><option value="synthetic_certified_hidden_target">合成同域</option><option value="real_known_visible_control">真实已知可见道路</option></select><select id="metric-select"></select></div><div class="tablewrap"><table><thead><tr><th>提示词</th><th>均值</th><th>最小–最大</th><th>最差参照误差</th><th>总体标准差</th><th>有效seed</th></tr></thead><tbody id="aggregate-body"></tbody></table></div><p class="sub" id="aggregate-note"></p><p class="sub">PSD三带为0.025–0.2 cycles/native pixel的平均绝对log2功率比；四带另含0.2–0.3。HP以原生像素指定σ。未审阅不等于没有假标线，高频能量不以越高越好。</p>'''
    detail = '<h2>固定提示词</h2><div class="summarycards">'+prompt_cards+'</div>'+limits+'<details><summary>执行与来源</summary><p>full1152_T0 · T0 step600 · DDIM20 · CFG3 · eta0 · strength1 · BF16 / fp16权重。CLIP实际token数：历史23，新词40，均未截断。本轮只完成提示词对照第1轮，不自动开启第2轮。</p><p>source PNG保持原字节；raw/final ROI为无损固定原生窗口；WebP90仅为全帧预览，指标读取远端原PNG。真实已知可见指标评价raw教师解码的保持性，final则在W外/K逐位复制原始观测；二者不能混为隐藏道路颜色改善。</p><p>12个历史legacy重放的两网格指标最大差为0；它们仍包含在正式60预算中，4个smoke另列。</p><div class="links"><a href="assets/asset_manifest.json">全部图像来源</a><a href="report_summary.json">完整统计JSON</a><a href="per_seed.csv">逐seed表</a><a href="publication.html">轻量出版页</a><a href="next_round_config.json">冻结合同</a></div></details><footer class="footer">2026-11-20 · human_verdict=accepted_method_current_cases；prompt human_verdict=null。原报告保留，三维与高频材质许可不扩张。</footer>'
    full = head+'<main>'+intro+findings_html(summary)+viewer+aggregate+detail+'</main><script>window.REPORT_DATA='+payload+';</script><script>'+js+'</script></body></html>'
    publication = head+'<main class="publication">'+intro+findings_html(summary)+'<h2>完整配对统计</h2>'+aggregate_html(summary)+'<p class="sub">表内每项为五seed均值及[最小,最大]。先在同底图内汇总任务变体，再对两张底图等权。真实可见控制读取raw教师解码图；final在W外逐位复制原始观测，不能将二者混成颜色改善。</p><h2>本轮提示词</h2><div class="summarycards">'+prompt_cards+'</div>'+limits+'<p>全部60个原始生成与固定ROI已在本地完整图像报告保留；本轻量页不依赖这些图像文件。方法采纳仅覆盖当前用户审阅样例，新提示词未人工晋升。</p><footer class="footer">64 diffusion calls = 4工程smoke + 60正式；4workers；实例按用户要求保持开机。未作自动seed/图片选择，无三维渲染修改。</footer></main></body></html>'
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/"index.html").write_text(full,encoding="utf-8")
    (a.out/"publication.html").write_text(publication,encoding="utf-8")
    (a.out/"report_summary.json").write_text(json.dumps(summary or dict(status="statistics_pending",automatic_winner_selected=False),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    csv_source = a.per_seed or (a.summary.parent/"per_seed.csv" if a.summary else None)
    if csv_source and csv_source.exists() and csv_source.resolve() != (a.out/"per_seed.csv").resolve():
        shutil.copy2(csv_source,a.out/"per_seed.csv")
    print(json.dumps(dict(full=str(a.out/"index.html"),publication=str(a.out/"publication.html"),cases=6,candidates=60,summary_merged=summary is not None)))


if __name__=="__main__":
    main()

"""汇总真实case的可离线HTML报告；缺分显示弃权/待评，不用0占位。"""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import statistics

DIMS = ["A", "P", "E", "O", "T", "OP"]
CSS = """
:root{font-family:system-ui,'Microsoft YaHei',sans-serif;color:#18283b;background:#f4f7fb;line-height:1.6}body{margin:0}main{max-width:1320px;margin:auto;padding:28px}h1{font-size:30px;margin-bottom:8px}h2{font-size:22px}.muted{color:#52657b}.banner{background:#fff1d9;border-left:4px solid #c37f13;padding:15px}.stats,.pair{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}.stats{grid-template-columns:repeat(4,1fr)}.card,article{background:white;border:1px solid #dce5ef;border-radius:12px;padding:18px;margin:20px 0}.stats .card{margin:12px 0}.value{font-size:30px;color:#176b83}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #e4eaf1;vertical-align:top}th{background:#edf3f8;position:sticky;top:0}a{color:#086684}video,img{width:100%;border-radius:8px;background:#132035}button{background:#176b83;color:white;border:0;border-radius:5px;padding:9px 15px;cursor:pointer}input[type=range]{width:60%}details{margin-top:12px}summary{cursor:pointer;font-weight:600}.pill{display:inline-block;padding:2px 8px;background:#e9f4f2;border-radius:5px;font-size:12px}.note{font-size:13px;color:#566d82}svg{width:100%;height:auto}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.dim{font-weight:600}.sectionnav{display:flex;gap:15px;flex-wrap:wrap}@media(max-width:800px){.pair,.stats{grid-template-columns:1fr}main{padding:14px}}
"""
CSS += "\n.pair{grid-template-columns:repeat(3,minmax(0,1fr))}.pair strong{display:block;min-height:3em}th{white-space:nowrap}.architecture{overflow-x:auto}.architecture svg{min-width:900px}@media(max-width:800px){.pair{grid-template-columns:1fr}.pair strong{min-height:0}}\n"
CSS += "\n.task-intent{background:#edf7fa;border-left:5px solid #176b83;border-radius:8px;padding:16px 20px;margin:16px 0 22px}.task-intent h3{margin:0 0 12px;font-size:22px;color:#12566b}.task-intent p{margin:8px 0}.task-intent .task-warning{background:#fff1d9;padding:10px}.task-summary{min-width:180px;max-width:260px}.task-intent .note{overflow-wrap:anywhere}\n"
SVG = """<svg viewBox="0 0 1240 150" role="img" aria-label="输入、条件适配、成对生成、独立读出与六维评价架构"><defs><marker id="arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0 0L9 3.5L0 7" fill="#176b83"/></marker></defs><g fill="#edf5f8" stroke="#80a6b4"><rect x="5" y="25" width="260" height="85" rx="10"/><rect x="325" y="25" width="250" height="85" rx="10"/><rect x="635" y="25" width="250" height="85" rx="10"/><rect x="945" y="25" width="290" height="85" rx="10"/></g><g stroke="#176b83" stroke-width="3" marker-end="url(#arrow)"><path d="M268 65H315"/><path d="M578 65H625"/><path d="M888 65H935"/></g><g text-anchor="middle" fill="#183b50" font-family="sans-serif" font-size="18"><text x="135" y="56">固定 24-case · nuScenes</text><text x="135" y="86">RGB / 标定 / GT状态 / 地图</text><text x="450" y="56">单变量干预 + 输入适配</text><text x="450" y="86">其余状态 / seed 保持一致</text><text x="760" y="56">OmniDreams 2B</text><text x="760" y="86">factual ↔ counterfactual</text><text x="1090" y="56">独立图像读出 + AI 初评</text><text x="1090" y="86">A / P / E / O / T / OP</text></g><text x="620" y="142" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#596e80">本报告是 GT 控制适配基线，不是高斯重建方法比较；人工 verdict 保持空白。</text></svg>"""
JS = """
document.querySelectorAll('article[data-case]').forEach(card=>{
 const videos=[...card.querySelectorAll('video')], slider=card.querySelector('input[type=range]');
 const button=card.querySelector('button');card.dataset.playerReady='true';
 button?.addEventListener('click',async()=>{const t=videos[0].currentTime>=2.3?0:videos[0].currentTime;button.textContent='正在同步播放';try{await Promise.all(videos.map(v=>{v.muted=true;v.currentTime=t;return v.play()}))}catch(e){button.textContent='浏览器未启动同步，请用视频控件播放';}});
 slider?.addEventListener('input',()=>videos.forEach(v=>{v.pause();v.currentTime=+slider.value}));
 videos[0]?.addEventListener('timeupdate',()=>{slider.value=Math.min(2.3,videos[0].currentTime);if(videos[0].currentTime>=2.3){videos.forEach(v=>{v.pause();if(v.currentTime>2.3)v.currentTime=2.3});button.textContent='三列重新播放（仅评分窗口）'}});
});
"""


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def fmt(value):
    return "—" if value is None else f"{value:.3f}" if isinstance(value, float) else html.escape(str(value))


def describe_case(row):
    """由冻结输入生成中文任务说明，不根据输出或评分倒推编辑意图。"""
    case = row["case"]
    target, intervention = case["target"], case["intervention"]
    family, spec = intervention["family"], intervention["counterfactual"]
    ego = target["role"] == "ego"
    classes = {"vehicle.car": "汽车", "vehicle.truck": "卡车", "vehicle.bus.rigid": "巴士",
               "vehicle.trailer": "挂车", "vehicle.construction": "施工车辆",
               "vehicle.bicycle": "自行车/骑行者", "vehicle.motorcycle": "摩托车/骑行者"}
    noun = classes.get(target.get("class_name"), target.get("class_name", "对象"))
    key = target.get("actor_key", target.get("source_asset_actor_key"))
    name = "自车（ego，相机随自车运动）" if ego else f"{noun} #{key}"
    begin = row["event_output_frame"] / row["fps"]
    end = (row["score_frames"]-1) / row["fps"]
    if family == "actor_speed_change":
        scale = float(spec["speed_scale"])
        action = "减速" if scale < 1 else "加速" if scale > 1 else "保持原速"
        title = f"{'自车' if ego else name}{action} · {scale:g}×"
        factual = f"{name}按原轨迹正常推进（1.0×）。"
        cf = f"将{name}的轨迹时间推进倍率改为{scale:g}×（原倍率的{scale*100:g}%），沿原路线推进。不是直接指定某个km/h值。"
        expected = "观察同一时刻沿路线的推进量是否减少。" if scale < 1 else "观察同一时刻沿路线的推进量是否增加。" if scale > 1 else "观察原轨迹是否保持。"
    elif family == "actor_lateral_relocation":
        offset = float(spec["lateral_offset_m"])
        title = f"{'自车' if ego else name}横移 · {offset:+g} m"
        factual = f"{name}保持原轨迹，额外横向偏移为0米。"
        cf = f"在原轨迹上叠加局部Y方向{offset:+g}米偏移；从{begin:g}秒起平滑增加，至{end:g}秒达到全幅（过渡{end-begin:g}秒）。"
        expected = "观察目标位置/自车视角向新轨迹偏移，同时原位置不应残留第二个目标；不把几何横移直接当作合法换道。"
    elif family == "actor_removal":
        title = f"移除{name}"
        factual = f"保留{name}，按原状态继续存在。"
        cf = f"从{begin:g}秒起删除{name}的状态轨迹；之前的公共前缀仍保留它。"
        expected = "目标应消失，原遮挡区域应合理显露；不能只驶出画面，也不能留下目标残影。"
    elif family == "actor_insertion":
        dx, dy, dz = map(float, target["proposal_offset_actor_frame_m"])
        title = f"插入1辆{noun} · 供体#{key}"
        name = f"新增{noun}（尺寸/朝向/轨迹供体：{noun} #{key}）"
        factual = "不新增对象；供体原车和其他对象保留。"
        cf = f"从{begin:g}秒起新增1辆{noun}，相对供体每帧局部坐标偏移 X {dx:+g} / Y {dy:+g} / Z {dz:+g} 米；供体原车继续保留。"
        expected = "观察指定新位置是否多出一个独立对象，且供体仍存在；不是把供体搬走。当前OmniDreams输入只提供几何/轨迹，不保证复制供体外观。"
    else:
        raise ValueError(f"未定义的反事实任务: {family}")
    unchanged = "同初始图、同文本、同seed42；地图及非目标状态条件保持不变。"
    unchanged += "自车编辑会改变相机轨迹，不要求背景像素逐点相同。" if ego else "自车/相机轨迹保持不变。"
    return {"title": title, "target": name, "family": family, "factual": factual, "counterfactual": cf,
            "timing": f"视频0–{begin:g}秒为公共前缀；{begin:g}秒开始干预（生成帧{row['event_output_frame']}，从0计数；源帧{case['anchor']['event_frame']}）。观察至{end:g}秒。",
            "unchanged": unchanged, "expected_visible_change": expected,
            "coordinate_note": "偏移使用输入位姿的局部坐标，不是屏幕左右；ego以LiDAR位姿、其他对象以actor位姿为基准。" if family in ["actor_lateral_relocation", "actor_insertion"] else None}


def task_intent_html(task, audit):
    parts = ['<section class="task-intent" aria-label="本case反事实任务"><p class="note">本 case 要做的反事实（目标，不是执行结果）</p>',
             '<h3>'+html.escape(task["title"])+"</h3>"]
    for label, key in [("编辑对象", "target"), ("事实分支", "factual"), ("反事实要求", "counterfactual"),
                       ("生效时间", "timing"), ("其余保持", "unchanged"), ("看什么变化", "expected_visible_change")]:
        parts.append(f'<p><strong>{label}：</strong>{html.escape(task[key])}</p>')
    if task["coordinate_note"]:
        parts.append('<p class="note">'+html.escape(task["coordinate_note"])+"</p>")
    delta = audit.get("target_endpoint_delta_m")
    if task["family"] == "actor_speed_change" and delta is not None and delta < .01:
        parts.append(f'<p class="task-warning">输入提醒：虽然设定为速度变化，本case实际目标终点差仅{delta:.6f}米，编辑不易辨认；不能把无明显变化直接算成模型未执行。</p>')
    parts.append('</section>')
    return ''.join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--rules", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load(args.inputs / "index.json")["cases"]
    rules = load(args.rules)
    ai = load(args.run / "ai-preliminary.json", {})
    items = []
    for row in rows:
        cid = row["case_id"]
        path = args.run / cid
        result, evaluation = load(path / "result.json", {}), load(path / "evaluation.json", {})
        record = {"case_id": cid, "input": row, "counterfactual_task": describe_case(row), "generation": result, "automatic": evaluation, "input_audit": load(path / "input-audit.json", {}), "ai_preliminary": ai.get(cid), "human_verdict": None}
        items.append(record)
        dest = args.output / "cases" / cid
        dest.mkdir(parents=True, exist_ok=True)
        for name in ["original-nuscenes.mp4", "original-nuscenes.json", "factual.mp4", "counterfactual.mp4", "evaluation-review.jpg", "condition-review.jpg", "result.json", "evaluation.json", "render-result.json", "input-audit.json"]:
            if (path / name).exists():
                shutil.copy2(path / name, dest / name)
    generated = sum(x["generation"].get("status") == "generation_complete" for x in items)
    evaluated = sum(bool(x["automatic"]) for x in items)
    reviewed = sum(bool(x["ai_preliminary"]) for x in items)
    completion = "24-case 推理与首轮诊断完成" if generated == evaluated == reviewed == 24 else "进行中：推理／自动读出／AI初评分开计数"
    body = [f'<h1>OmniDreams · 自建反事实 benchmark</h1><p class="muted">{completion} · {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p>',
        '<nav class="sectionnav"><a href="#overview">24-case 对照表</a><a href="#rubric">评分规则</a><a href="report-data.json">结构化结果</a><a href="scoring-rules-v1.json">完整规则</a></nav>',
        '<p class="banner">固定我们设计的 case，不使用官方 demo 替代。输入为 nuScenes GT 地图／轨迹适配；同 seed 42、同初始图、同文本、单变量干预。77 帧原生输出中只评分前 70 帧（0–2.3 秒），尾部 7 帧填充不计分。仅 3 个源场景、开发集探索，不能推断总体发生率。</p>',
        '<div class="stats">'+''.join(f'<div class="card">{k}<div class="value">{v}</div></div>' for k,v in [("请求 / 支持", "24 / 24"),("成对推理完成", f"{generated} / 24"),("自动读出完成", f"{evaluated} / 24"),("AI 初评已审阅", f"{reviewed} / 24")])+'</div>',
        '<section class="card architecture"><h2>评测链路</h2>'+SVG+'</section>',
        '<section class="card"><h2>首轮发现与证据边界</h2><p>已逐 case 审阅固定五帧（0/15/33/51/69），不是完整视频动态或闭环驾驶评测。AI 初评与人工 verdict 分开；人工仍未填写。</p><ul><li><a href="#CFB-LATERAL-ACTOR-02">LATERAL-ACTOR-02</a>：可见单车横移，2D读出支持，保留正例。</li><li><a href="#CFB-REMOVE-05">REMOVE-05</a>：白色目标车仍保留，删除未执行；<a href="#CFB-REMOVE-01">REMOVE-01</a>也有挂车残留。</li><li><a href="#CFB-LATERAL-ACTOR-03">LATERAL-ACTOR-03</a>：原位置与新位置同时出现骑车人，是复制残留而非干净移动。</li><li><a href="#CFB-SPEED-EGO-02">SPEED-EGO-02</a>：干预终点差仅约0.00005米，A/O/T弃权；保留该case，不事后换case。</li><li>输入的横移/插入尚未通过完整道路/碰撞审阅。输入审计为事后诊断，不伪装成事前门控；高背景保持分也可能来自完全没有执行编辑。</li></ul></section>',
        '<section class="card" id="overview"><h2>逐 case 六维对照</h2><p class="note">反事实任务列说明“要求做什么”，不是“已经做到什么”。0–4 为 AI 序数诊断，非官方论文分数。— 表示未评／弃权／不适用，不等于 0。每项分母不同；不合成总分。</p><div class="scroll"><table><thead><tr><th>Case</th><th>反事实任务</th><th>状态</th>'+''.join(f'<th>{d}</th>' for d in DIMS)+'<th>背景 MAE↓</th><th>事实 PSNR↑</th></tr></thead><tbody>']
    for item in items:
        cid = item["case_id"]
        scores = (item["ai_preliminary"] or {}).get("dimensions", {})
        summary = item["automatic"].get("summary", {})
        body.append(f'<tr><td><a href="#{cid}">{cid}</a></td><td class="task-summary">{html.escape(item["counterfactual_task"]["title"])}</td><td>{"已生成" if item["generation"].get("status") == "generation_complete" else "运行中"}</td>'+''.join(f'<td>{fmt(scores.get(d, {}).get("pair"))}</td>' for d in DIMS)+f'<td>{fmt(summary.get("outside_edit_mae_01_mean"))}</td><td>{fmt(summary.get("factual_psnr_db_mean"))}</td></tr>')
    body.append('</tbody></table></div><p class="note">背景 MAE 是同相机、编辑框外的半分辨率近似；ego 干预不计算。PSNR 只与事实观测比，不能代表反事实正确性。</p></section>')
    body.append('<section class="card"><h2>六维分布 · 各报分母</h2><p class="note">序数分仅描述本轮审阅范围，不合成总分、不作模型排名。P仅可见形状/遮挡代理；P成对列采用两支较低值。E的事实列对照原图、成对列比较分支；不能混作同一统计。</p><table><tr><th>维度</th><th>可评分 n / 24</th><th>中位数</th><th>0 / 1 / 2 / 3 / 4 的数量</th><th>弃权或N/A</th></tr>')
    for d in DIMS:
        values = [(item['ai_preliminary'] or {}).get('dimensions', {}).get(d, {}).get('pair') for item in items]
        scores = [v for v in values if v is not None]
        body.append(f'<tr><td>{d}</td><td>{len(scores)} / 24</td><td>{fmt(statistics.median(scores) if scores else None)}</td><td>'+ ' / '.join(str(scores.count(i)) for i in range(5))+f'</td><td>{24-len(scores)}</td></tr>')
    body.append('</table></section>')
    body.append('<section class="card" id="rubric"><h2>评分规则 v1</h2><p>自动原单位指标 / AI 初评 / 人工复核分层保存。AI 初评依据指定静帧和独立读出；不能证明未观察到的帧间动态。所有人工 verdict 均为 null。</p><table><thead><tr><th>维度</th><th>评价问题</th><th>限制</th></tr></thead><tbody>')
    for key, dim in rules["dimensions"].items():
        body.append(f'<tr><td>{key} — {html.escape(dim["name"])}</td><td>{html.escape(dim["question"])}</td><td>{html.escape(dim["guard"])}</td></tr>')
    body.append('</tbody></table><p>'+html.escape('；'.join(k+'：'+v for k,v in rules["ordinal_anchors"].items()))+'</p><p class="note">参考 <a href="https://arxiv.org/html/2605.27589v1">What-If World</a> 的配对与 A/P/E/O 思路；该文采用严格二元评价，本地 0–4 与 T/OP 是扩展，不冒充原文复现。</p></section>')
    for item in items:
        cid, row = item["case_id"], item["input"]
        base = f"cases/{cid}"
        summary = item["automatic"].get("summary", {})
        review = item["ai_preliminary"] or {}
        body.append(f'<article id="{cid}" data-case="{cid}"><h2>{cid}</h2><p><span class="pill">{row["scene_name"]} · camera {row["camera_index"]}</span></p>')
        body.append(task_intent_html(item["counterfactual_task"], item["input_audit"]))
        body.append('<details><summary>查看原始任务参数（供核对）</summary><pre>'+html.escape(json.dumps({"target":row["case"]["target"], "intervention":row["case"]["intervention"]}, ensure_ascii=False, indent=2))+'</pre></details>')
        if item["generation"].get("status") == "generation_complete":
            body.append('<div class="pair">'+''.join(f'<div><strong>{label}</strong><video controls preload="metadata" playsinline src="{base}/{b}.mp4"></video></div>' for b,label in [("original-nuscenes", "原始 nuScenes · 10Hz"),("factual", "factual · 30Hz"),("counterfactual", "counterfactual · 30Hz")])+'</div><p><button>三列同步播放（仅评分窗口）</button> <input aria-label="同步视频时间" type="range" min="0" max="2.3" step="0.033333" value="0"></p><p class="note">原始参考保留1600×900、24帧、不插帧；生成1280×704。按相同秒数同步，评分窗口0–2.3秒。<a href="'+base+'/original-nuscenes.json">参考视频来源</a></p>')
        body.append('<table><thead><tr><th>维度</th><th>事实</th><th>反事实</th><th>成对</th><th>证据 / 弃权理由</th></tr></thead><tbody>')
        for d in DIMS:
            score = review.get("dimensions", {}).get(d, {})
            body.append(f'<tr><td>{d}</td><td>{fmt(score.get("factual"))}</td><td>{fmt(score.get("counterfactual"))}</td><td>{fmt(score.get("pair"))}</td><td>{html.escape(score.get("evidence", "AI 初评待完成"))}</td></tr>')
        body.append('</tbody></table>')
        if review:
            body.append('<p>'+html.escape(review.get("finding", ""))+'</p>')
        if summary:
            body.append('<details><summary>自动指标：真实事实 / 生成事实 / 反事实</summary><pre>'+html.escape(json.dumps(summary, ensure_ascii=False, indent=2))+'</pre></details>')
        if item["automatic"]:
            body.append(f'<details><summary>指定帧三列对照（真实 / 事实生成 / 反事实生成）</summary><img loading="lazy" src="{base}/evaluation-review.jpg" alt="{cid} 固定帧对照"><p class="note">红框原目标投影、绿框目标控制投影、黄框独立检测。框是诊断锚点，不是生成物体真值。</p></details>')
        body.append(f'<details><summary>输入条件与事后可行性审计</summary><img loading="lazy" src="{base}/condition-review.jpg" alt="{cid} 条件审计"><pre>'+html.escape(json.dumps(item['input_audit'], ensure_ascii=False, indent=2))+f'</pre></details><p><a href="{base}/result.json">推理证据</a> · <a href="{base}/evaluation.json">逐帧读出</a> · <a href="{base}/input-audit.json">输入审计</a></p></article>')
    body.append('<footer class="note">版本：cfbench-six-dimension-v1。单张 RTX 3090。公开单视图 2B 权重，非多视图版。6 native ego + 18 adapted actor。训练数据重叠未知；不作总体排名。人工判定未代填。本轮检测存档仅保留已匹配检测，不包含全部未匹配候选；移除CF没有目标track，0匹配是结构性结果，绝不是删除成功率。</footer>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OmniDreams 24-case 评测</title><style>'+CSS+'</style><main>'+''.join(body)+'</main><script>'+JS+'</script></html>'
    (args.output / "index.html").write_text(page)
    shutil.copy2(args.rules, args.output / "scoring-rules-v1.json")
    (args.output / "report-data.json").write_text(json.dumps({"generated": generated, "automatic_readout": evaluated, "ai_reviewed": reviewed, "items": items}, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"generated": generated, "automatic": evaluated, "ai_reviewed": reviewed, "output": str(args.output)}))


if __name__ == "__main__":
    main()

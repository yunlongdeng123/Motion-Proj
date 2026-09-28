"""Build the offline, human-scoring nuScenes DELETE audit page."""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

ISSUE_LABELS = {
    "background blur/artifact": "背景模糊／块状伪影",
    "no obvious issue in review frame": "本帧未见明显问题",
    "uncertain": "无法确定",
    "vehicle-like instance (unresolved)": "仍见车体，身份待核实",
    "hidden actor structure artifact": "真实后车结构异常",
    "actor regeneration (suspected)": "疑似车辆再生",
    "mask failure": "输入遮罩缺陷",
}
QUAL_LABELS = {
    "review_frame_no_obvious_mask_error": "本帧未见明显mask错误",
    "review_frame_ambiguous": "本帧实例边界／身份难确认",
    "review_frame_mask_failure": "本帧mask有明显缺陷",
}


def esc(value):
    return html.escape(str(value), quote=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.report.read_text(encoding="utf-8"))
    clips = data["clips"]
    assert len(clips) == 70
    assert len({r["scene"] for r in clips}) == 45
    issues = Counter(r["assistant_one_frame_issue"] for r in clips)
    cards = []
    for row in clips:
        cid = row["clip_id"]
        info = f"{row['scene']} · {row['category']} · {row['camera']} · keyframe {row['start_keyframe']} · target {row['instance_token']}"
        factors = ", ".join(row["difficulty_factors"]) or "无列出的 GT 风险项"
        difficulty = row.get("assistant_input_difficulty", row["input_difficulty_proxy"])
        issue_label = ISSUE_LABELS.get(row["assistant_one_frame_issue"], row["assistant_one_frame_issue"])
        qualification = row.get("input_qualification", "pending_review")
        input_note = (f"输入检查：{QUAL_LABELS.get(qualification, qualification)}；"
                      f"GT可见但mask为空的帧：{row.get('empty_core_with_visible_GT', [])}；"
                      f"写回与其他GT投影交叠：{len(row.get('frames_write_other_GT_overlap', []))}/26帧。"
                      "投影框交叠仅作风险提示，需看实际像素归属，不能直接判误删。")
        video_html = []
        for kind, title, note in [
            ("original", "原视频 · 黄框目标", "框来自 GT 投影；目标不在当前视野的帧会标明。"),
            ("model_input", "删除输入范围", "蓝色是模型洞，橙色是最终写回，绿色是邻车保护区。"),
            ("native", "DriveEditor 原生输出", "模型生成的整帧，用于看洞内内容和洞外改动。"),
            ("delete", "最终 DELETE 补景", "仅按固定 mask/保护规则写回原视频；这是实际使用的补景。"),
        ]:
            path = row["videos"][kind]["path"]
            poster_path = row["videos"][kind]["poster"]
            video_html.append(f'<section class="video-col"><h4>{esc(title)}</h4><video controls muted playsinline preload="none" poster="{esc(poster_path)}" data-src="{esc(path)}"></video><p>{esc(note)}</p></section>')
        cards.append(f'''
<article id="{esc(cid)}" class="case" data-difficulty="{esc(difficulty)}" data-issue="{esc(row['assistant_one_frame_issue'])}" data-camera="{esc(row['camera'])}" data-scene="{esc(row['scene'])}">
 <div class="case-head"><div><span class="id">{esc(cid)}</span> <strong>{esc(row['scene'])}</strong> <span class="tag">{esc(row['camera'])}</span> <span class="tag">输入难度 {esc(difficulty)}</span> <span class="tag issue">单帧粗分类 {esc(issue_label)}</span></div><div class="score">人工 DELETE 评分 <select data-score="{esc(cid)}"><option value="">待评</option><option value="0">0 · 失败</option><option value="1">1 · 不佳</option><option value="2">2 · 可接受</option></select></div></div>
 <p class="identity">{esc(info)}</p>
 <p class="meta">GT 目标可见 {row['GT_target_visible_frames']}/26 帧；SAM core 非空 {row['core_nonempty_frames']}/26 帧；模型洞非空 {row['model_nonempty_frames']}/26 帧；目标大小 {esc(row['size_bucket'])}；GT 可见性 {esc(row['occlusion_proxy'])}；后方车辆投影 {('有' if row['behind_vehicle_proxy'] else '无')}</p>
 <p class="meta">输入难度依据：{esc(factors)}。DriveEditor factual reconstruction：未运行；原视频是事实输入控制。</p>
 <p class="meta">冻结难度代理：{esc(row['input_difficulty_proxy'])}；助手抽帧补充：{esc(row.get('assistant_difficulty_note', '待复核'))}。抽帧为 f{row['prompt_frame']:02d}，未据输出重选。</p>
 <p class="meta">{esc(input_note)}</p>
 <p class="note">单帧观察：{esc(row['assistant_review_note'] or '待单帧复核；请结合完整视频人工评分。')}</p>
 <img class="oneframe" src="{esc(row['one_frame_review'])}" loading="lazy" alt="{esc(cid)} 同区域原图与删除单帧对照">
 <p class="meta"><a href="{esc(row.get('one_frame_components', row['one_frame_review']))}" target="_blank">放大查看同一帧：原图／mask／原生／最终四组件</a></p>
 <details><summary>打开四列同步视频</summary><div class="controls"><button type="button" class="play">同步播放</button><button type="button" class="pause">暂停</button><button type="button" class="restart">回到开始</button><label>速度 <select class="speed"><option value="1">1×</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option></select></label></div><div class="videos">{''.join(video_html)}</div></details>
</article>''')
    count_text = " · ".join(f"{ISSUE_LABELS.get(k,k)}: {v}" for k, v in sorted(issues.items()))
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 · nuScenes val DELETE audit</title><style>
:root{color-scheme:dark;font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:#101722;color:#e8eef6}*{box-sizing:border-box}body{margin:0}main{max-width:1720px;margin:auto;padding:24px}h1{font-size:30px;margin:0 0 8px}h2{font-size:20px;margin:25px 0 12px}p{line-height:1.55}.sub{color:#aebdce;max-width:1000px}.hero,.panel,.case{border:1px solid #344257;border-radius:16px;background:#172231;padding:20px;margin:0 0 18px}.hero{background:linear-gradient(115deg,#16283f,#102239)}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.stat{background:#23344a;padding:12px 18px;border-radius:12px;min-width:140px}.stat b{font-size:25px;display:block}.architecture{display:block;width:100%;max-width:1000px;height:auto;margin:16px 0}.filters{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:14px 0}.filters select,.filters input,.score select,.speed{background:#23344a;color:#fff;border:1px solid #52657d;padding:8px;border-radius:8px}.filters input{min-width:220px}.tag{display:inline-block;padding:3px 8px;border-radius:6px;background:#34475c;color:#d7eaff;font-size:13px;margin:2px}.issue{background:#314539}.case-head{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}.id{font-weight:800;color:#ffcf47;margin-right:8px}.identity{margin:8px 0;color:#cbd8e8}.meta{font-size:14px;color:#aebdce;margin:6px 0}.note{border-left:3px solid #8dbdff;padding-left:10px;color:#d3e6ff}.oneframe{display:block;width:min(100%,1280px);height:auto;border-radius:8px;margin:12px 0}.controls{display:flex;gap:10px;align-items:center;margin:12px 0;flex-wrap:wrap}.controls button,#export{background:#2c629e;border:0;color:white;border-radius:7px;padding:8px 12px;cursor:pointer}.videos{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.video-col{background:#101a28;padding:8px;border-radius:10px}.video-col h4{margin:4px 0 8px}.video-col video{width:100%;height:auto;background:#000}.video-col p{font-size:12px;color:#aec0d1;margin:7px 0}details summary{cursor:pointer;color:#91cbff;font-weight:700;padding:8px 0}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #40516b;padding:7px 10px;text-align:left}.hidden{display:none!important}.foot{color:#aebdce;font-size:14px}@media(max-width:1100px){.videos{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:620px){main{padding:12px}.videos{grid-template-columns:1fr}h1{font-size:23px}}
</style></head><body><main><header class="hero"><h1>v77 · nuScenes val DELETE 补景审计</h1><p class="sub">45 个新 scene、70 个单目标 clip，逐例看原视频、模型洞、DriveEditor 原生输出与最终补景。每个 clip 约 2.6 秒；四列保持同一相机与时间顺序。70/70推理已完成，280段视频共7280帧已实际解码。旧九例仅作为 DEV 用户评分，不参与本轮比例。</p><div class="stats"><div class="stat"><b>45</b>audit scene</div><div class="stat"><b>70</b>单车 DELETE</div><div class="stat"><b>25</b>隔离的 final scene</div><div class="stat"><b>0</b>训练步数</div></div><svg class="architecture" viewBox="0 0 1010 126" role="img" aria-label="视频与GT框经过SAM2 mask、DriveEditor deletion、固定写回到四列视频审核"><defs><marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#8ec9ff"/></marker></defs><g fill="#27405d" stroke="#639cc6"><rect x="5" y="25" width="170" height="70" rx="12"/><rect x="212" y="25" width="170" height="70" rx="12"/><rect x="419" y="25" width="170" height="70" rx="12"/><rect x="626" y="25" width="170" height="70" rx="12"/><rect x="833" y="25" width="170" height="70" rx="12"/></g><g fill="#eef5ff" text-anchor="middle" font-size="15"><text x="90" y="56">nuScenes RGB</text><text x="90" y="77">+ GT 目标框</text><text x="297" y="56">SAM2</text><text x="297" y="77">单车 mask</text><text x="504" y="56">DriveEditor</text><text x="504" y="77">deletion</text><text x="711" y="56">保护邻车</text><text x="711" y="77">固定写回</text><text x="918" y="56">原图/洞/原生</text><text x="918" y="77">/最终视频</text></g><g stroke="#8ec9ff" stroke-width="3" marker-end="url(#arr)"><path d="M177 60 H207"/><path d="M384 60 H414"/><path d="M591 60 H621"/><path d="M798 60 H828"/></g></svg></header>
<section class="panel"><h2>怎么评</h2><p>先看黄框目标是否是要删的那辆车，再看蓝色模型洞与橙色写回是否误伤邻车，最后看“原生输出”和“最终 DELETE”的道路、栏杆、后车身份与连续性。单帧粗分类由助手给出，用户 0/1/2 分请用每张卡的下拉框记录并导出。视频中的事实控制是原始 RGB；DriveEditor 官方 deletion 接口没有原位 factual reconstruction，因此该项保持“未运行/不可判”，不会把原视频复制成模型成绩。</p><p>70例均只看固定提示帧，计数是单帧粗分类，不是视频成功率或模型失效率。“本帧未见明显问题”不等于人工2分；“仍见车体”可能是真实后车，不能自动算幻觉。背景伪影项包含输入不确定的例子，须结合每卡的输入检查。时序漂移未从静帧评分。单帧计数：__COUNTS__。</p><p>这批是官方 nuScenes val scene；预训练 checkpoint 的具体训练 scene 重叠尚未核实。输入使用 GT 框和 SAM2；四个相机时间点最近帧偏差 70–85ms，个别 10Hz 位置复用了同一 12Hz 曝光，均在来源记录中保留。缺 mask 的帧输入原图原样保留，会标明输入缺陷。</p></section>
<section class="panel"><h2>筛选与评分</h2><div class="filters"><label>scene <input id="search" placeholder="scene-0637 / A001"></label><label>输入难度 <select id="difficulty"><option value="">全部</option><option>low</option><option>medium</option><option>high</option></select></label><label>相机 <select id="camera"><option value="">全部</option><option>CAM_FRONT</option><option>CAM_FRONT_LEFT</option><option>CAM_FRONT_RIGHT</option><option>CAM_BACK_LEFT</option><option>CAM_BACK_RIGHT</option><option>CAM_BACK</option></select></label><label>粗分类 <select id="issue"><option value="">全部</option>__ISSUE_OPTIONS__</select></label><button id="export" type="button">导出人工评分 JSON</button><span id="shown"></span></div></section>
<div id="cases">__CARDS__</div>
<section class="panel"><h2>旧九例 DEV 用户评分</h2><p>0＝失败，1＝效果不佳，2＝可接受但不一定好。它们已被观察、修过多轮，不作为新 audit 或 final 测试。</p><table><thead><tr><th>scene / actor</th><th>原位重建</th><th>编辑</th><th>用户备注</th></tr></thead><tbody><tr><td>0230 / 22</td><td>1</td><td>2</td><td></td></tr><tr><td>0255 / 25</td><td>1</td><td>0</td><td></td></tr><tr><td>official_000 / 12</td><td>2</td><td>2</td><td></td></tr><tr><td>350 / 7</td><td>2</td><td>2</td><td></td></tr><tr><td>663 / 6</td><td>0</td><td>0</td><td>目标模糊、半车身，暂缓</td></tr><tr><td>191 / 12</td><td>0</td><td>0</td><td>小目标、树干遮挡，暂缓</td></tr><tr><td>425 / 1</td><td>0</td><td>0</td><td>小目标无遮挡，需解决</td></tr><tr><td>382 / 4</td><td>1</td><td>1</td><td></td></tr><tr><td>756 / 3</td><td>1</td><td>1</td><td>小目标、夜间</td></tr></tbody></table></section><p class="foot">原始元数据、固定采样、逐帧文件与 mask、模型原生帧、最终 PNG 保留在 AutoDL run：/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-AUDIT-20260928/r1。人工分数只在浏览器本机保存，可导出；不会写回模型结果或改变冻结样本。</p></main><script>
const key='v77-delete-audit-20260928-r1-ratings';let ratings={};try{ratings=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){};
document.querySelectorAll('select[data-score]').forEach(s=>{s.value=ratings[s.dataset.score]??'';s.onchange=()=>{ratings[s.dataset.score]=s.value;try{localStorage.setItem(key,JSON.stringify(ratings))}catch(e){}}});
function filter(){const q=document.querySelector('#search').value.trim().toLowerCase(),d=document.querySelector('#difficulty').value,c=document.querySelector('#camera').value,i=document.querySelector('#issue').value;let n=0;document.querySelectorAll('.case').forEach(card=>{const show=(!q||(card.dataset.scene+card.querySelector('.id').textContent).toLowerCase().includes(q))&&(!d||card.dataset.difficulty===d)&&(!c||card.dataset.camera===c)&&(!i||card.dataset.issue===i);card.classList.toggle('hidden',!show);if(show)n++});document.querySelector('#shown').textContent=n+' / 70'}
['search','difficulty','camera','issue'].forEach(id=>document.querySelector('#'+id).addEventListener(id==='search'?'input':'change',filter));filter();
document.querySelectorAll('.case details').forEach(details=>{details.addEventListener('toggle',()=>{if(details.open)details.querySelectorAll('video[data-src]').forEach(v=>{v.src=v.dataset.src;v.removeAttribute('data-src')})});const vids=()=>[...details.querySelectorAll('video')];details.querySelector('.play').onclick=()=>{const list=vids();const ref=list[0];const t=ref.ended?0:ref.currentTime;list.forEach(v=>{v.currentTime=t;v.play().catch(()=>{})})};const ref=vids()[0];ref.addEventListener('seeking',()=>vids().slice(1).forEach(v=>{if(Number.isFinite(ref.currentTime))v.currentTime=ref.currentTime}));ref.addEventListener('timeupdate',()=>{if(!ref.paused)vids().slice(1).forEach(v=>{if(Math.abs(v.currentTime-ref.currentTime)>0.15)v.currentTime=ref.currentTime})});details.querySelector('.pause').onclick=()=>vids().forEach(v=>v.pause());details.querySelector('.restart').onclick=()=>vids().forEach(v=>v.currentTime=0);details.querySelector('.speed').onchange=e=>vids().forEach(v=>v.playbackRate=Number(e.target.value))});
document.querySelector('#export').onclick=()=>{const output={task_id:'WS-V77-DELETE-AUDIT-20260928',run_id:'r1',score_meaning:{0:'failure',1:'poor',2:'acceptable'},delete_scores:ratings};const blob=new Blob([JSON.stringify(output,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='v77-delete-audit-human-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
</script></body></html>'''
    page = page.replace("__COUNTS__", esc(count_text)).replace("__CARDS__", "\n".join(cards))
    page = page.replace("__ISSUE_OPTIONS__", "".join(f'<option value="{esc(issue)}">{esc(ISSUE_LABELS.get(issue,issue))}</option>' for issue in sorted(issues)))
    args.output.write_text(page, encoding="utf-8")
    print("HTML", len(clips), args.output)


if __name__ == "__main__":
    main()

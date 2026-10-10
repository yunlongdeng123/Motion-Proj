"""构建 P1 相邻复盘点人工审核页；只引用已有媒体，不启动推理或重编码。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MEDIA = ROOT / "outputs/v81-paper-p1"
CASES = (
    ("00f88c4f0a", "side_0.125", "滑板 · 小外扩"),
    ("00f88c4f0a", "side_0.33", "滑板 · 大外扩"),
    ("7e625db8c4", "side_0.125", "海豚 · 小外扩"),
    ("7e625db8c4", "side_0.33", "海豚 · 大外扩"),
    ("ff6eb95840", "side_0.125", "白鲸 · 小外扩"),
    ("ff6eb95840", "side_0.33", "白鲸 · 大外扩"),
)
PAIR_KEYS = (
    "source_frames", "frame_selection", "generation_protocol", "preprocessing",
    "reference_indices", "flow_pairs", "mask_pixels_left", "mask_pixels_right",
    "reference_window", "reference_selection_input", "side_ratio_each", "seed",
    "steps", "num_frames", "mode", "fps", "min_guidance", "max_guidance",
    "noise_aug_strength", "motion_bucket_id", "propagation_protocol",
)
EXPECTED_PNG = {f"{i:05d}.png" for i in range(25)}


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层须为对象: {path}")
    return value


def url(page: Path, path: Path) -> str:
    return Path(os.path.relpath(path, page.parent)).as_posix()


def require_assets(folder: Path, stem: str) -> None:
    video = folder / f"{stem}.mp4"
    frames = folder / f"frames_{stem}"
    if not video.is_file() or video.stat().st_size == 0:
        raise FileNotFoundError(f"缺少非空视频: {video}")
    actual = {p.name for p in frames.glob("*.png")}
    if actual != EXPECTED_PNG or any((frames / name).stat().st_size == 0 for name in EXPECTED_PNG):
        raise ValueError(f"须有非空的 00000..00024 共25张PNG: {frames}; 实际 {len(actual)} 张")


def validate_run(run: dict, step: int, sequence: str, side: str, path: Path) -> None:
    expected = {
        "status": "complete", "checkpoint_step": step, "sequence_id": sequence,
        "side_ratio_each": float(side[5:]), "seed": 2026, "steps": 25,
        "num_frames": 25, "mode": "paper-feedforward",
        "propagation_protocol": "paper-bidirectional-m4",
    }
    bad = [key for key, value in expected.items() if run.get(key) != value]
    if bad:
        raise ValueError(f"固定窗协议不匹配 {path}: {', '.join(bad)}")
    if len(run.get("source_frames", [])) != 25:
        raise ValueError(f"source_frames 必须恰好25帧: {path}")


def paired_runs(old: dict, new: dict, old_path: Path, new_path: Path) -> None:
    for key in PAIR_KEYS:
        if key not in old or key not in new or old[key] != new[key]:
            raise ValueError(f"相邻复盘点未配对 {key}: {old_path} <> {new_path}")


def assistant_reviews(media: Path, step: int) -> dict[tuple[str, str], dict]:
    """审核文件可尚未到；若已到则严格读取，绝不补造结论。"""
    folder = media / f"paper_bidirectional_m4/validation/step{step:06d}"
    found = {}
    expected = {(seq, side) for seq, side, _ in CASES}
    for path in sorted(folder.glob("assistant_review_*.json")):
        document = read_json(path)
        if document.get("checkpoint_step") != step or document.get("human_verdict", "missing") is not None:
            raise ValueError(f"{step} 审核元数据或 human_verdict 不符: {path}")
        rows = document.get("windows", document.get("reviews"))
        if not isinstance(rows, list):
            raise ValueError(f"{step} 审核必须含 windows 或 reviews 数组: {path}")
        for row in rows:
            key = (row.get("sequence_id"), row.get("side"))
            if key not in expected or key in found:
                raise ValueError(f"{step} 审核窗口未知或重复: {path}: {key}")
            if row.get("frames_reviewed") != 25:
                raise ValueError(f"{step} 审核须覆盖全部25帧: {path}: {key}")
            for score_key in ("native_score", "comp_score"):
                if not isinstance(row.get(score_key), (int, float)):
                    raise ValueError(f"{step} 审核评分缺失: {path}: {key}/{score_key}")
            if row.get("decision") not in ("hold", "continue"):
                raise ValueError(f"{step} 审核 decision 缺失或未知: {path}: {key}")
            found[key] = {"native_score": row["native_score"],
                          "comp_score": row["comp_score"],
                          "decision": row["decision"],
                          "comparison": row.get(f"compared_{step-2500}", ""),
                          "evidence": row.get("evidence", []),
                          "limitations": row.get("limitations", "")}
    return found


def has_assets(folder: Path, stem: str) -> bool:
    video = folder / f"{stem}.mp4"
    frames = folder / f"frames_{stem}"
    return video.is_file() and frames.is_dir()


def column(page: Path, folder: Path, stem: str, label: str, badge: str, *, check: bool = True) -> dict:
    if check:
        require_assets(folder, stem)
    return {"label": label, "badge": badge,
            "video": url(page, folder / f"{stem}.mp4"),
            "frames": url(page, folder / f"frames_{stem}")}


def propagation_output_root(root: Path) -> Path:
    """接受消融输出目录本身，也接受其上一层 run root。"""
    nested = root / "propagation_ablation_step7500"
    return nested if nested.is_dir() else root


def propagation_case(page: Path, root: Path | None, sequence: str, side: str,
                     formal_dir: Path, formal_run: dict) -> dict:
    if root is None:
        return {"status": "pending", "reason": "未提供 --propagation-root；推理消融待完成。"}
    root = propagation_output_root(root)
    report_path = root / "run.json"
    case_dir = root / sequence / side
    ablated = case_dir / "no_cross_frame"
    case_path = case_dir / "run.json"
    if not report_path.is_file() or not case_path.is_file():
        return {"status": "pending", "reason": "传播消融汇总或本窗尚未落盘；待完成。"}
    report, case = read_json(report_path), read_json(case_path)
    if report.get("status") == "running":
        return {"status": "pending", "reason": "传播消融仍在运行；本窗数据待完成。"}
    if (report.get("status") != "complete"
            or report.get("kind") != "step7500_zero_update_cross_frame_propagation_off"
            or report.get("checkpoint_step") != 7500
            or report.get("optimizer_updates") != 0
            or report.get("seed") != formal_run["seed"]
            or report.get("steps") != formal_run["steps"]
            or report.get("mode") != formal_run["mode"]):
        raise ValueError(f"传播消融汇总协议不符: {report_path}")
    if (case.get("status") != "complete"
            or case.get("checkpoint_step") != 7500
            or case.get("optimizer_updates") != 0
            or case.get("seed") != formal_run["seed"]
            or case.get("steps") != formal_run["steps"]
            or case.get("mode") != formal_run["mode"]
            or case.get("sequence_id") != sequence
            or case.get("side_ratio_each") != formal_run["side_ratio_each"]
            or case.get("source_frames") != formal_run["source_frames"]
            or case.get("reference_indices") != formal_run["reference_indices"]
            or case.get("flow_pairs") != formal_run["flow_pairs"]
            or case.get("ordinary_replays_formal_all25_exact") is not True
            or not case.get("paired_exact_shared_inputs")
            or not all(value is True for value in case["paired_exact_shared_inputs"].values())):
        raise ValueError(f"传播消融配对证明不符: {case_path}")
    # 远端绝对路径在本机无效；用正式媒体目录验证普通支，不要求复制一份。
    if Path(case.get("ordinary_formal_dir", "")).as_posix().split("/")[-2:] != [sequence, side]:
        raise ValueError(f"普通支目录记录不符: {case_path}")
    branches = case.get("branches", {})
    for branch in ("ordinary", "no_cross_frame"):
        if Path(branches.get(branch, "")).as_posix().split("/")[-3:] != [sequence, side, branch]:
            raise ValueError(f"{branch} 分支目录记录不符: {case_path}")
    summaries = report.get("cases", [])
    if not any(row.get("sequence_id") == sequence and row.get("side_ratio_each") == formal_run["side_ratio_each"]
               and row.get("ordinary_replays_formal_all25_exact") is True for row in summaries):
        raise ValueError(f"传播消融汇总缺少本窗: {report_path}")
    columns = [
        column(page, formal_dir, "pred", "普通传播 · 原生", "正式7500"),
        column(page, ablated, "pred", "no_cross_frame · 原生", "推理消融"),
        column(page, formal_dir, "comp", "普通传播 · 硬写回", "中央来自GT"),
        column(page, ablated, "comp", "no_cross_frame · 硬写回", "中央来自GT"),
    ]
    conditions = []
    condition_records = case.get("intermediate_conditions", {})
    condition_keys = {"visible_vae", "condition_full", "condition_no_cross"}
    if condition_records and set(condition_records) != condition_keys:
        raise ValueError(f"VAE 解码条件键不符: {case_path}")
    for folder, stem, label in (
        (case_dir, "visible_vae", "逐帧可见 VAE · 传播前"),
        (case_dir, "condition_full", "普通传播条件 · VAE 解码"),
        (case_dir, "condition_no_cross", "无跨帧条件 · VAE 解码"),
    ):
        if has_assets(folder, stem):
            record = condition_records.get(stem, {})
            if (Path(record.get("mp4", "")).as_posix().split("/")[-3:]
                    != [sequence, side, f"{stem}.mp4"]
                    or Path(record.get("frames", "")).as_posix().split("/")[-3:]
                    != [sequence, side, f"frames_{stem}"]):
                raise ValueError(f"条件展示元数据不符: {case_path}/{stem}")
            conditions.append(column(page, folder, stem, label, "仅条件展示 · 非生成"))
    review = None
    review_path = case_dir / "assistant_review.json"
    if review_path.is_file():
        review = read_json(review_path)
        if (review.get("human_verdict", "missing") is not None
                or review.get("frames_reviewed") != 25
                or review.get("sequence_id") != sequence
                or review.get("side") != side):
            raise ValueError(f"传播助手审核元数据不符: {review_path}")
    measurement = case.get("measurement", {})
    return {"status": "ready", "columns": columns, "review": review,
            "measurement": measurement,
            "conditions": conditions,
            "conditions_note": ("三列条件可视化均已核对25帧。" if len(conditions) == 3
                                else f"条件可视化已到 {len(conditions)}/3 列；其余待完成。"),
            "reason": "普通支逐像素重放正式7500全部25帧；同权重、同输入、零优化更新。移除跨帧传播是推理时的分布外干预，不等价于重训消融。"}


def build_data(page: Path, media: Path, propagation_root: Path | None,
               before_step: int = 5000, after_step: int = 7500,
               *, check_only: bool = False) -> list[dict]:
    phase = media / "paper_bidirectional_m4/validation"
    scores = assistant_reviews(media, before_step)
    reviews = assistant_reviews(media, after_step) if not check_only else {}
    result = []
    for sequence, side, title in CASES:
        old = phase / f"step{before_step:06d}" / sequence / side
        old_path = old / "run.json"
        old_run = read_json(old_path)
        validate_run(old_run, before_step, sequence, side, old_path)
        # 静态检查在下一复盘点尚未同步时也能完成，但会核实已有媒体。
        for stem in ("gt", "visible", "pred", "comp"):
            require_assets(old, stem)
        if check_only:
            continue
        new = phase / f"step{after_step:06d}" / sequence / side
        new_path = new / "run.json"
        if not new_path.is_file():
            raise FileNotFoundError(f"{after_step} 正式六窗媒体尚未同步：缺少 {new_path}")
        new_run = read_json(new_path)
        validate_run(new_run, after_step, sequence, side, new_path)
        paired_runs(old_run, new_run, old_path, new_path)
        columns = [
            column(page, new, "gt", "真实目标 GT", "仅作对照"),
            column(page, new, "visible", "可见输入", "唯一合法输入"),
            column(page, old, "pred", f"{before_step} 步 · 原生",
                   "历史断点" if after_step == 7500 else "上一复盘点"),
            column(page, new, "pred", f"{after_step} 步 · 原生",
                   "当前断点" if after_step == 7500 else "当前复盘点"),
        ]
        if after_step != 7500:
            previous_comp = column(page, old, "comp", f"{before_step} 步 · 硬写回", "中央来自 GT")
            previous_comp["description"] = "上一复盘点的GT中央写回；比较侧区与接缝"
            columns.append(previous_comp)
        current_comp = column(page, new, "comp", f"{after_step} 步 · 硬写回", "中央来自 GT")
        if after_step != 7500:
            current_comp["description"] = "当前复盘点的GT中央写回；比较侧区与接缝"
        columns.append(current_comp)
        score = scores.get((sequence, side), {})
        result.append({"id": sequence + "_" + side, "title": title,
                       "sequence": sequence, "side": side,
                       "old_native_score": score.get("native_score"),
                       "old_comp_score": score.get("comp_score"),
                       "old_decision": score.get("decision"),
                       "review": reviews.get((sequence, side)),
                       "columns": columns,
                       "propagation": (propagation_case(page, propagation_root, sequence, side,
                                                        new, new_run) if after_step == 7500
                                       else {"status": "not_applicable", "reason": "传播推理消融仅有 7500 步诊断。"})})
    return result


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seen-to-Scene P1 · 7500 步人工审核</title>
<style>
:root{color-scheme:dark;--bg:#090e16;--card:#111b29;--soft:#172638;--line:#2c4055;--text:#e8eff8;--muted:#aabbd0;--cyan:#62d6e9;--amber:#e9c27e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,"Segoe UI",sans-serif}main{max-width:1780px;margin:auto;padding:24px clamp(14px,2.5vw,40px) 60px}h1{font-size:clamp(27px,2.4vw,40px);line-height:1.15;margin:5px 0 10px}h2{margin:0;font-size:21px}h3{margin:0;font-size:13px}p{margin:6px 0}.muted,.sub{color:var(--muted)}.eyebrow{color:var(--cyan);font-size:12px;font-weight:750;letter-spacing:.13em;text-transform:uppercase}.pill{display:inline-block;border:1px solid var(--line);border-radius:99px;padding:6px 10px;background:var(--card);font-size:12px;margin:3px}.pill.wait{color:var(--amber)}.pill.ok{color:var(--cyan)}
header{display:flex;justify-content:space-between;gap:18px;flex-wrap:wrap;align-items:flex-start}.sub{max-width:850px}.panel{border:1px solid var(--line);background:var(--card);border-radius:13px;padding:15px;margin-top:16px}.architecture{display:flex;gap:7px;align-items:center;flex-wrap:wrap}.box{padding:7px 10px;border:1px solid #416179;border-radius:8px;background:#152b3c;font-size:12px}.arrow{color:var(--cyan)}.arch-note{margin-left:auto;color:var(--muted);font-size:12px}
button,input{font:inherit}button{cursor:pointer;color:var(--text);background:var(--soft);border:1px solid #3a536b;border-radius:8px;padding:7px 11px}button:hover,button.active{border-color:var(--cyan);color:var(--cyan)}.nav,.controls,.scores{display:flex;gap:7px;flex-wrap:wrap;align-items:center}.head{display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap}.score{font-size:12px;border:1px solid var(--line);padding:6px 9px;border-radius:8px;color:var(--muted)}.score strong{color:var(--cyan)}.controls{margin:14px 0}.controls input{flex:1;min-width:140px;accent-color:var(--cyan)}.counter{min-width:72px;text-align:center;color:var(--cyan);font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px}.tile{min-width:0;background:#0b1420;border:1px solid var(--line);border-radius:10px;overflow:hidden}.tile.native{border-color:#39828d}.tile.comp{border-color:#8d754e}.tile-head{min-height:53px;padding:8px;display:flex;justify-content:space-between;gap:5px;align-items:center}.tag{font-size:10px;color:var(--muted);white-space:nowrap}video,img.frame{width:100%;aspect-ratio:1;object-fit:contain;background:#05080d;display:block}video[hidden],img.frame[hidden]{display:none}.caption{color:var(--muted);font-size:11px;padding:8px;min-height:43px}.notice{border-left:3px solid var(--amber);background:#241e1a;color:#f4dfbb;padding:9px 11px;margin-top:12px;font-size:13px}.ablation{margin-top:19px;border-top:1px solid var(--line);padding-top:15px}.ablation .grid{grid-template-columns:repeat(2,minmax(0,1fr));max-width:860px}.ablation .tile{max-width:420px}.ablation .caption{min-height:0}.foot{font-size:12px;color:var(--muted);margin-top:19px}
.ablation #conditionGrid{grid-template-columns:repeat(3,minmax(0,1fr));max-width:none}.ablation #conditionGrid .tile{max-width:none}.ablation h3{margin-top:12px}
@media(max-width:1180px){.grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:680px){main{padding:15px 11px 38px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.tile-head{display:block}.tag{display:block}.ablation .grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:440px){.grid,.ablation .grid{grid-template-columns:1fr}}
@media(max-width:680px){.ablation #conditionGrid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:440px){.ablation #conditionGrid{grid-template-columns:1fr}}
</style></head><body><main>
<header><div><div class="eyebrow">Seen-to-Scene · P1 / fixed-window review</div><h1>7500 步人工审核</h1><p class="sub">同一组六个 25 帧短窗，逐帧比较 5000→7500 的原生生成、侧区和接缝。助手结论只在实际审核记录到齐后显示，人工 verdict 保持未填写。</p></div><div><span class="pill ok">7500 媒体已核对</span><span class="pill wait" id="overallReview">助手结论：待完成</span><span class="pill wait">人工 verdict：未填写</span></div></header>
<section class="panel architecture" aria-label="architecture components：输入、关键组件、数据流与输出"><span class="box">中心可见视频</span><span class="arrow">→</span><span class="box">VAE + RAFT 光流</span><span class="arrow">→</span><span class="box">双向潜变量传播</span><span class="arrow">→</span><span class="box">SVD 去噪</span><span class="arrow">→</span><span class="box">原生 25 帧</span><span class="arrow">→</span><span class="box">可见区硬写回</span><span class="arch-note">GT 仅作目标与写回参考</span></section>
<section class="panel"><strong>结果概览</strong><p id="resultSummary" class="muted">助手审核记录待完成。</p><p class="muted">7500 完整 checkpoint 已保存；训练停止，服务器保持开机。</p></section>
<section class="panel"><strong>实验范围</strong><p class="muted">固定六窗、seed 2026、25 步、25 帧、paper-feedforward / paper-bidirectional-m4。这是局部诊断，不是正式 90+60 基准或论文四指标。传播区仅比较推理时跨帧条件，不代表重训。</p></section>
<nav class="panel nav" id="nav" aria-label="选择验证窗"></nav>
<section class="panel"><div class="head"><div><h2 id="title"></h2><p class="muted" id="windowId"></p></div><div class="scores"><span class="score" id="oldScore"></span><span class="score" id="newScore">7500 助手结论：待完成</span><span class="score">人工 verdict <strong>未填写</strong></span></div></div>
<div class="controls"><button id="play">播放全部</button><button id="prev">上一帧</button><button id="next">下一帧</button><input id="slider" type="range" min="0" max="24" value="0" aria-label="逐帧查看"><span class="counter" id="counter">00 / 24</span><button id="videoMode" class="active">视频</button><button id="pngMode">逐帧 PNG</button></div>
<div class="grid" id="grid"></div><div class="notice">硬写回列的中央可见区直接来自 GT。只从两侧生成区域和接缝判断该列；5000 旧评分仅供历史参照，不推定 7500 效果。</div>
<div class="ablation"><h2>7500 助手审核证据</h2><div class="muted" id="reviewEvidence">待完成。</div></div>
<div class="ablation"><h2>传播收益 · 推理消融</h2><p class="muted" id="propStatus"></p><div class="notice" id="propEvidence"></div><div class="grid" id="propGrid"></div><h3>条件可视化 · VAE 解码，仅用于解释条件，不是 QUERY 生成帧</h3><p class="muted" id="conditionStatus"></p><div class="grid" id="conditionGrid"></div></div></section>
<p class="foot">本页只链接已存在的本地 MP4 与每列 00000..00024 PNG；构建过程不重新编码，也不计算 FVD。</p>
</main><script>
const cases=__DATA__;const $=id=>document.getElementById(id);let active=0,frame=0,mode='video',timer=null;
const descriptions=['仅作目标对照','真正提供给模型的中心可见带','5000 完整模型输出','7500 完整模型输出','GT 中央写回；仅侧区与接缝可归功于模型'];
function allVideos(){return [...document.querySelectorAll('video')]}function stop(){if(timer){clearInterval(timer);timer=null}allVideos().forEach(v=>v.pause());$('play').textContent='播放全部'}
function path(col,n){return col.frames+'/'+String(n).padStart(5,'0')+'.png'}
function seek(n){frame=Math.max(0,Math.min(24,Number(n)));$('slider').value=frame;$('counter').textContent=String(frame).padStart(2,'0')+' / 24';document.querySelectorAll('.frame').forEach(img=>{img.src=path(img._column,frame);img.alt=img._column.label+' 第'+frame+'帧'});allVideos().forEach(v=>{if(v.readyState>0){try{v.currentTime=Math.min(frame/7+.001,Math.max(0,v.duration-.02))}catch(e){}}})}
function addTile(container,col,kind,description){let tile=document.createElement('article');tile.className='tile '+kind;let head=document.createElement('div');head.className='tile-head';let name=document.createElement('h3');name.textContent=col.label;let tag=document.createElement('span');tag.className='tag';tag.textContent=col.badge;head.append(name,tag);let video=document.createElement('video');video.src=col.video;video.muted=true;video.playsInline=true;video.preload='metadata';video.hidden=mode!=='video';video.onclick=()=>timer?stop():play();let img=document.createElement('img');img.className='frame';img._column=col;img.hidden=mode!=='png';let foot=document.createElement('div');foot.className='caption';foot.textContent=description;tile.append(head,video,img,foot);container.append(tile)}
function play(){if(mode==='png')setMode('video');let vs=allVideos();Promise.all(vs.map(v=>v.play().catch(()=>null)));$('play').textContent='暂停全部';timer=setInterval(()=>{if(!vs.length)return;let t=vs[0].currentTime;vs.slice(1).forEach(v=>{if(Math.abs(v.currentTime-t)>.12){try{v.currentTime=t}catch(e){}}});frame=Math.max(0,Math.min(24,Math.floor(t*7)));$('slider').value=frame;$('counter').textContent=String(frame).padStart(2,'0')+' / 24';if(vs[0].ended)stop()},90)}
function setMode(next){stop();mode=next;$('videoMode').classList.toggle('active',next==='video');$('pngMode').classList.toggle('active',next==='png');allVideos().forEach(v=>v.hidden=next!=='video');document.querySelectorAll('.frame').forEach(img=>img.hidden=next!=='png');seek(frame)}
function addEvidence(parent,label,value){if(!value)return;let values=Array.isArray(value)?value:[value];values.forEach(item=>{let p=document.createElement('p');p.textContent=label+String(item);parent.append(p)})}
function select(i){stop();active=i;frame=0;let c=cases[i];$('title').textContent=c.title;$('windowId').textContent=c.sequence+' / '+c.side+' · f00–f24';let old=c.old_native_score;$('oldScore').textContent=old==null?'5000 旧评分：未提供':'5000 旧评分：原生 '+Number(old).toFixed(2)+' / 写回 '+Number(c.old_comp_score).toFixed(2)+'（'+(c.old_decision||'无结论')+'）';let r=c.review;$('newScore').textContent=r?'7500 助手：'+r.decision+' · 原生 '+Number(r.native_score).toFixed(2)+' / 写回 '+Number(r.comp_score).toFixed(2):'7500 助手结论：待完成';$('reviewEvidence').replaceChildren();if(r){addEvidence($('reviewEvidence'),'相对 5000：',r.comparison);addEvidence($('reviewEvidence'),'证据：',r.evidence);addEvidence($('reviewEvidence'),'限制：',r.limitations)}else $('reviewEvidence').textContent='本窗助手审核记录尚未到；待完成。';document.querySelectorAll('#nav button').forEach((b,j)=>b.classList.toggle('active',i===j));$('grid').replaceChildren();c.columns.forEach((col,j)=>addTile($('grid'),col,j===3?'native':j>=4?'comp':'',col.description||descriptions[j]||''));$('propGrid').replaceChildren();$('conditionGrid').replaceChildren();$('propStatus').textContent=c.propagation.reason;$('conditionStatus').textContent=c.propagation.status==='ready'?c.propagation.conditions_note:'条件可视化待完成。';if(c.propagation.status==='ready'){c.propagation.columns.forEach((col,j)=>addTile($('propGrid'),col,j>1?'comp':'native',j>1?'GT 中央硬写回；比较侧区与接缝':'相同权重与输入的原生输出'));c.propagation.conditions.forEach(col=>addTile($('conditionGrid'),col,'','仅展示条件；不代表最终生成质量'))}seek(0)}
const reviewed=cases.filter(c=>c.review).length;$('overallReview').textContent=reviewed===6?'助手审核：6/6 窗已记录':'助手审核：'+reviewed+'/6 窗；其余待完成';
if(reviewed===6){let usable=cases.filter(c=>Math.min(c.review.native_score,c.review.comp_score)>=2).length;let reviewedProp=cases.filter(c=>c.propagation.review).length;let clearProp=cases.filter(c=>c.propagation.review&&c.propagation.review.decision!=='no_clear_gain').length;$('resultSummary').textContent='5000→7500 的结构和动作变化见逐窗证据；'+usable+'/6 窗达到本次 2 分可用门槛，'+(usable===6?'六窗均达到门槛。':'整体仍未通过。')+(__PROPAGATION_ENABLED__?(reviewedProp===6?(clearProp===0?'六窗传播开／关均未见明确视觉收益。':'传播开／关有 '+clearProp+' 窗需查看独立结论。'):'传播审核待完成。'):'传播推理消融仅有 7500 步诊断，本轮未重复。')}
const selectBase=select;select=function(i){selectBase(i);let p=cases[i].propagation,r=p.review,m=p.measurement||{};$('propEvidence').replaceChildren();if(r){addEvidence($('propEvidence'),'全25帧助手结论：',r.decision);addEvidence($('propEvidence'),'视觉证据：',r.evidence);addEvidence($('propEvidence'),'条件观察：',r.condition_observation)}else addEvidence($('propEvidence'),'','传播全帧审核待完成。');let g=m.final_output_gain_positive_is_better;if(g)addEvidence($('propEvidence'),'洞区MAE改善（0–255，正值有利，仅像素代理）：',Number(g.hole_rgb_mae_0_255).toFixed(3));};
cases.forEach((c,i)=>{let b=document.createElement('button');b.textContent=c.title;b.onclick=()=>select(i);$('nav').append(b)});$('play').onclick=()=>timer?stop():play();$('prev').onclick=()=>{stop();seek(frame-1)};$('next').onclick=()=>{stop();seek(frame+1)};$('slider').oninput=e=>{stop();seek(e.target.value)};$('videoMode').onclick=()=>setMode('video');$('pngMode').onclick=()=>setMode('png');select(0);
</script></body></html>'''


def render_html(cases: list[dict], before_step: int, after_step: int) -> str:
    # 模板保留7500传播诊断；后续轮次只展示本轮正式六窗。
    html = HTML.replace("5000", "__BEFORE_STEP__").replace("7500", str(after_step))
    html = html.replace("__BEFORE_STEP__", str(before_step))
    html = html.replace("__PROPAGATION_ENABLED__", "true" if after_step == 7500 else "false")
    if after_step != 7500:
        html = html.replace("grid-template-columns:repeat(5,minmax(0,1fr))",
                            "grid-template-columns:repeat(6,minmax(0,1fr))")
        html = html.replace('<div class="ablation"><h2>传播收益 · 推理消融</h2>',
                            '<div class="ablation" style="display:none"><h2>传播收益 · 推理消融</h2>')
        html = html.replace(f"传播推理消融仅有 {after_step} 步诊断，本轮未重复。",
                            "传播推理消融仅有 7500 步诊断，本轮未重复。")
        html = html.replace(f"{after_step} 完整 checkpoint 已保存；训练停止，服务器保持开机。",
                            "本页只核对已有媒体；训练与服务器状态以控制器记录为准。")
        html = html.replace("传播区仅比较推理时跨帧条件，不代表重训。",
                            "传播推理消融仅有7500步诊断，本轮未重复。")
    return html.replace("__DATA__", json.dumps(cases, ensure_ascii=False).replace("<", "\\u003c"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, default=DEFAULT_MEDIA,
                        help="本地 outputs/v81-paper-p1 媒体根目录")
    parser.add_argument("--output", type=Path, help="输出 HTML；默认 MEDIA/step<after-step>_review.html")
    parser.add_argument("--before-step", type=int, default=5000, help="上一复盘点，默认5000")
    parser.add_argument("--after-step", type=int, default=7500, help="当前复盘点，默认7500")
    parser.add_argument("--propagation-root", type=Path,
                        help="可选 run root 或 propagation_ablation_step7500 目录；普通支复用正式7500媒体")
    parser.add_argument("--check-only", action="store_true",
                        help="只验证脚本契约与上一复盘点六窗，不要求当前媒体已到")
    args = parser.parse_args()
    if args.before_step < 5000 or args.after_step != args.before_step + 2500:
        parser.error("复盘点须从5000起，每轮相隔2500步")
    media = args.media.resolve()
    page = (args.output or media / f"step{args.after_step}_review.html").resolve()
    if "__DATA__" not in HTML or len(CASES) != 6:
        raise ValueError("页面模板或固定六窗契约损坏")
    cases = build_data(page, media, args.propagation_root.resolve() if args.propagation_root else None,
                       args.before_step, args.after_step,
                       check_only=args.check_only)
    if args.check_only:
        print(json.dumps({"status": "static_ok", "before_step": args.before_step,
                          "before_windows_checked": len(CASES), "after_step": args.after_step,
                          "after_required": False, "output_written": False}, ensure_ascii=False))
        return
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(render_html(cases, args.before_step, args.after_step), encoding="utf-8")
    print(json.dumps({"output": str(page), "windows": len(cases),
                      "main_media_links": sum(len(c["columns"]) * 26 for c in cases),
                      "propagation_ready": sum(c["propagation"]["status"] == "ready" for c in cases),
                      "assistant_reviews_loaded": sum(c["review"] is not None for c in cases),
                      "human_verdict": None}, ensure_ascii=False))


if __name__ == "__main__":
    main()

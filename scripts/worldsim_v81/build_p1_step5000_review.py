"""生成 P1 step5000 深色人工审核页；只引用已有媒体，不复制或编码视频。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MEDIA = ROOT / "outputs/v81-paper-p1"
CASES = (
    ("00f88c4f0a", "side_0.125", "滑板 · 小外扩", "镜头落地和跳跃尚未跟随，侧区仍像天空。"),
    ("00f88c4f0a", "side_0.33", "滑板 · 大外扩", "黑色结构更明显，但地面与滑板动作仍缺失。"),
    ("7e625db8c4", "side_0.125", "海豚 · 小外扩", "身体与尾鳍比 2000 步更连贯；位置和数量仍偏离。"),
    ("7e625db8c4", "side_0.33", "海豚 · 大外扩", "形体和运动增强，但主体尺度与位置仍不跟随输入。"),
    ("ff6eb95840", "side_0.125", "白鲸 · 小外扩", "水面与体态变化增加，仍未跟随靠近镜头的尺度。"),
    ("ff6eb95840", "side_0.33", "白鲸 · 大外扩", "运动更强，但头部、体型与边界对齐仍失配。"),
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def media_url(page: Path, path: Path) -> str:
    return Path(os.path.relpath(path, page.parent)).as_posix()


def require_media(path: Path, *, images: bool = False) -> None:
    if images:
        expected = [f"{i:05d}.png" for i in range(25)]
        actual = sorted(p.name for p in path.glob("*.png"))
        if actual != expected:
            raise FileNotFoundError(f"须有完整25张PNG: {path}; 实际{len(actual)}")
    elif not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def build_data(page: Path, media: Path) -> tuple[list[dict], int]:
    phase = media / "paper_bidirectional_m4/validation"
    reviews_dir = phase / "step005000"
    skater = read_json(reviews_dir / "assistant_review_skater_dolphin.json")
    beluga = read_json(reviews_dir / "assistant_review_beluga_dolphin.json")
    reviews = {(row["sequence_id"], row["side"]): row
               for row in skater["windows"] + beluga["reviews"]}
    if skater["human_verdict"] is not None or beluga["human_verdict"] is not None:
        raise ValueError("本页不能代填人工结论")
    result = []
    links = 0
    for sequence, side, title, note in CASES:
        row = reviews.get((sequence, side))
        if row is None or row.get("frames_reviewed") != 25:
            raise ValueError(f"独立助手审核不完整: {sequence}/{side}")
        old = phase / "step002000" / sequence / side
        new = phase / "step005000" / sequence / side
        old_run, new_run = read_json(old / "run.json"), read_json(new / "run.json")
        for run, step in ((old_run, 2000), (new_run, 5000)):
            if (run.get("status") != "complete" or run.get("checkpoint_step") != step
                    or run.get("sequence_id") != sequence or run.get("side_ratio_each") != float(side[5:])
                    or run.get("seed") != 2026 or run.get("steps") != 25
                    or run.get("num_frames") != 25 or run.get("mode") != "paper-feedforward"
                    or run.get("propagation_protocol") != "paper-bidirectional-m4"):
                raise ValueError(f"媒体协议不是固定正式{step}窗口: {sequence}/{side}")
        for key in ("source_frames", "reference_indices", "flow_pairs", "mask_pixels_left",
                    "mask_pixels_right", "preprocessing"):
            if old_run.get(key) != new_run.get(key):
                raise ValueError(f"2000/5000输入未配对: {sequence}/{side}/{key}")
        columns = []
        for label, badge, folder, stem in (
            ("真实目标 GT", "对照", new, "gt"),
            ("可见输入", "唯一合法输入", new, "visible"),
            ("2000 步 · 原生", "模型原生", old, "pred"),
            ("5000 步 · 原生", "模型原生", new, "pred"),
            ("5000 步 · 硬写回", "中间来自 GT", new, "comp"),
        ):
            video = folder / f"{stem}.mp4"
            frames = folder / f"frames_{stem}"
            require_media(video)
            require_media(frames, images=True)
            links += 26
            columns.append({"label": label, "badge": badge,
                            "video": media_url(page, video),
                            "frames": media_url(page, frames)})
        result.append({
            "id": f"{sequence}_{side}", "title": title,
            "sequence": sequence, "side": side, "note": note,
            "decision": row["decision"], "native_score": row["native_score"],
            "comp_score": row["comp_score"], "columns": columns,
        })
    return result, links


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seen-to-Scene P1 · 5000步人工审核</title>
<style>
:root{color-scheme:dark;--bg:#0b0f17;--card:#111827;--soft:#192437;--line:#293548;--text:#e7eef8;--muted:#a5b3c8;--cyan:#55d7e9;--amber:#f4c77a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
button,input{font:inherit}button{cursor:pointer}main{max-width:1740px;margin:auto;padding:24px clamp(14px,2.5vw,40px) 56px}
header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;flex-wrap:wrap}h1{font-size:clamp(26px,2.3vw,39px);line-height:1.15;margin:4px 0 10px;letter-spacing:-.035em}
.eyebrow{color:var(--cyan);font-size:12px;font-weight:750;letter-spacing:.14em;text-transform:uppercase}.sub,.hint{color:var(--muted);margin:0;max-width:850px}
.status{display:flex;gap:8px;flex-wrap:wrap}.pill{border:1px solid var(--line);border-radius:99px;padding:7px 11px;background:var(--card);font-size:12px;font-weight:700}.pill.ok{color:var(--cyan)}.pill.hold{color:var(--amber)}
.arch{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:23px 0;padding:13px 16px;border:1px solid var(--line);border-radius:12px;background:var(--card);color:#c7d5e8;font-size:13px}.arch b{color:var(--cyan);font-weight:600}.arch .arrow{color:#6b8ba0}
.toolbar,.review{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-top:16px}
.case-nav{display:flex;gap:8px;flex-wrap:wrap}.case-nav button,.control button{background:#1a2739;color:var(--text);border:1px solid #31435a;border-radius:9px;padding:8px 12px}.case-nav button:hover,.control button:hover{border-color:var(--cyan)}.case-nav button.active{background:#143342;color:var(--cyan);border-color:var(--cyan)}
.case-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:3px 0 16px}.case-head h2{font-size:22px;margin:0}.case-head p{color:var(--muted);margin:4px 0 0}.scores{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.score{border:1px solid var(--line);background:#172235;border-radius:8px;padding:6px 9px;font-size:12px}.score strong{color:var(--cyan)}
.control{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:12px 0 17px}.control input[type=range]{flex:1;min-width:140px;accent-color:var(--cyan)}.counter{min-width:92px;text-align:center;font-variant-numeric:tabular-nums;color:var(--cyan);font-weight:700}.switch.active{border-color:var(--cyan);color:var(--cyan)}
.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.tile{min-width:0;background:#0d1522;border:1px solid var(--line);border-radius:11px;overflow:hidden}.tile.native{border-color:#286877}.tile.comp{border-color:#80643b}.tile-head{min-height:54px;padding:8px 10px;display:flex;justify-content:space-between;align-items:center;gap:7px}.tile-head h3{margin:0;font-size:13px}.tag{color:var(--muted);font-size:10px;white-space:nowrap}.tile.native .tag{color:var(--cyan)}.tile.comp .tag{color:var(--amber)}video,.frame{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#05080e}.frame[hidden],video[hidden]{display:none}.tile-foot{padding:8px 10px;min-height:49px;color:var(--muted);font-size:11px}
.caution{margin-top:13px;border-left:3px solid var(--amber);padding:9px 12px;background:#221d1a;color:#f3ddbb;font-size:13px}.foot{margin-top:22px;color:var(--muted);font-size:12px}
@media(max-width:1200px){.grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:680px){main{padding:16px 12px 40px}.grid{grid-template-columns:1fr 1fr}.tile-head{display:block}.tag{display:block}}@media(max-width:440px){.grid{grid-template-columns:1fr}}
</style></head><body><main>
<header><div><div class="eyebrow">Seen-to-Scene · P1 / assistant review</div><h1>5000 步人工审核</h1><p class="sub">同一输入对照 2000 与 5000 步。六个固定验证窗，每窗 25 帧；先看模型原生，再看边界与硬写回。</p></div><div class="status"><span class="pill ok">5000 步完成</span><span class="pill hold">整体：hold</span><span class="pill">非正式论文指标</span></div></header>
<div class="arch" aria-label="架构组件"><span>可见视频</span><span class="arrow">→</span><b>光流 / 潜变量传播</b><span class="arrow">→</span><b>SVD 去噪</b><span class="arrow">→</span><span>原生生成</span><span class="arrow">→</span><span>可见区硬写回</span></div>
<section class="toolbar"><div class="case-nav" id="caseNav" aria-label="选择验证窗"></div></section>
<section class="review"><div class="case-head"><div><h2 id="caseTitle"></h2><p id="caseNote"></p></div><div class="scores"><span class="score" id="caseDecision"></span><span class="score">助手原生 <strong id="nativeScore"></strong></span><span class="score">助手写回 <strong id="compScore"></strong></span><span class="score">人工结论 <strong>未填写</strong></span></div></div>
<div class="control"><button id="playBtn" type="button">播放全部</button><button id="prevBtn" type="button">上一帧</button><button id="nextBtn" type="button">下一帧</button><input id="frameSlider" type="range" min="0" max="24" value="0" aria-label="选择帧"><span id="counter" class="counter">00 / 24</span><button id="videoBtn" class="switch active" type="button">视频</button><button id="pngBtn" class="switch" type="button">逐帧 PNG</button></div>
<div class="grid" id="grid"></div><div class="caution">硬写回列的中央可见区域直接复制 GT；只有两侧生成区域及接缝可用于判断模型。助手分数仅是图像审核意见，不是论文四指标，也不是人工结论。</div></section>
<p class="foot">固定 seed 2026 · 25 步 · paper-feedforward · 25 帧短窗诊断。所有媒体来自本地已有输出，页面不含外部脚本或 CDN。</p>
</main><script>
const cases=__DATA__;
const labels=['真实目标','遮蔽后可见输入','2000步原生','5000步原生','5000步硬写回'];
const details=['真实画面，仅作目标对照','实际提供给模型的中心可见带','较早断点的完整模型输出','当前断点的完整模型输出','中央来自GT，侧区来自5000原生'];
const $=id=>document.getElementById(id);let active=0,frame=0,mode='video',timer=null;
function videos(){return [...document.querySelectorAll('#grid video')]}function pause(){clearInterval(timer);timer=null;videos().forEach(v=>v.pause());$('playBtn').textContent='播放全部'}
function imgPath(column,n){return column.frames+'/'+String(n).padStart(5,'0')+'.png'}
function seek(n){frame=Math.max(0,Math.min(24,Number(n)));$('frameSlider').value=frame;$('counter').textContent=String(frame).padStart(2,'0')+' / 24';const c=cases[active];document.querySelectorAll('#grid .frame').forEach((img,i)=>{img.src=imgPath(c.columns[i],frame)});videos().forEach(v=>{if(v.readyState>0){try{v.currentTime=Math.min(frame/7+.001,Math.max(0,v.duration-.02))}catch(e){}}})}
function play(){if(mode==='png')setMode('video');const vs=videos();Promise.all(vs.map(v=>v.play().catch(()=>null))).then(()=>{});$('playBtn').textContent='暂停全部';timer=setInterval(()=>{if(!vs.length)return;const master=vs[0],t=master.currentTime;vs.slice(1).forEach(v=>{if(Math.abs(v.currentTime-t)>.12)try{v.currentTime=t}catch(e){}});const n=Math.max(0,Math.min(24,Math.floor(t*7)));frame=n;$('frameSlider').value=n;$('counter').textContent=String(n).padStart(2,'0')+' / 24';if(master.ended)pause()},90)}
function setMode(next){pause();mode=next;$('videoBtn').classList.toggle('active',next==='video');$('pngBtn').classList.toggle('active',next==='png');document.querySelectorAll('#grid video').forEach(v=>v.hidden=next!=='video');document.querySelectorAll('#grid .frame').forEach(img=>img.hidden=next!=='png');seek(frame)}
function select(index){pause();active=index;frame=0;const c=cases[index];$('caseTitle').textContent=c.title+' · '+c.sequence;$('caseNote').textContent=c.note;$('nativeScore').textContent=c.native_score.toFixed(2);$('compScore').textContent=c.comp_score.toFixed(2);$('caseDecision').textContent='助手：'+(c.decision==='continue'?'局部进步':'hold');$('caseDecision').style.color=c.decision==='continue'?'var(--cyan)':'var(--amber)';document.querySelectorAll('#caseNav button').forEach((b,i)=>b.classList.toggle('active',i===index));$('grid').innerHTML='';c.columns.forEach((col,i)=>{const tile=document.createElement('article');tile.className='tile '+(i===3?'native':i===4?'comp':'');const head=document.createElement('div');head.className='tile-head';head.innerHTML='<h3>'+col.label+'</h3><span class="tag">'+col.badge+'</span>';const video=document.createElement('video');video.src=col.video;video.muted=true;video.playsInline=true;video.preload='metadata';video.loop=false;video.addEventListener('click',()=>timer?pause():play());const img=document.createElement('img');img.className='frame';img.alt=col.label+' 第0帧';img.hidden=mode!=='png';video.hidden=mode!=='video';const foot=document.createElement('div');foot.className='tile-foot';foot.textContent=details[i];tile.append(head,video,img,foot);$('grid').append(tile)});seek(0)}
cases.forEach((c,i)=>{const b=document.createElement('button');b.type='button';b.textContent=c.title;b.onclick=()=>select(i);$('caseNav').append(b)});$('playBtn').onclick=()=>timer?pause():play();$('prevBtn').onclick=()=>{pause();seek(frame-1)};$('nextBtn').onclick=()=>{pause();seek(frame+1)};$('frameSlider').oninput=e=>{pause();seek(e.target.value)};$('videoBtn').onclick=()=>setMode('video');$('pngBtn').onclick=()=>setMode('png');select(0);
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, default=MEDIA)
    parser.add_argument("--output", type=Path, default=MEDIA / "step5000_review.html")
    args = parser.parse_args()
    page = args.output.resolve()
    if page.parent != args.media_root.resolve():
        raise ValueError("审核页必须保存在 outputs/v81-paper-p1 下，确保相对媒体链接有效")
    data, links = build_data(page, args.media_root.resolve())
    page.write_text(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    print(json.dumps({"output": str(page), "cases": len(data), "checked_local_links": links,
                      "assistant_hold": 5, "assistant_continue": 1, "human_verdict": None},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

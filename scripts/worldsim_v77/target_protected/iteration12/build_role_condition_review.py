"""r31–r34集中审核：明确主B/邻车，输入与条件都不冒充扩散输出。"""
from pathlib import Path
import json,html,sys,subprocess,shutil
import numpy as np
import cv2,imageio_ffmpeg
from PIL import Image
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r34/review'
def read(p):return json.loads(p.read_text())
def esc(s):return html.escape(str(s))
def encode(path,images):
    if path.is_file():return
    with subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1024x576','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE) as proc:
        for im in images:proc.stdin.write(np.ascontiguousarray(im).tobytes())
        proc.stdin.close();assert proc.wait()==0

def panels(c,q,i):
    root=Path(c['observed_folder']);y=np.asarray(Image.open(Path(c['source_Y_quality_only'])/f'{i:05}.png').convert('RGB')).copy()
    a=np.asarray(Image.open(root/'influence'/f'{i:05}.png'))>0;h=np.asarray(Image.open(root/'proposal_H'/f'{i:05}.png'))>0
    raw=y.copy();plan=y.copy();plan[a]=np.rint(plan[a]*.3+np.array([255,195,30])*.7).astype('uint8')
    cv2.drawContours(plan,cv2.findContours(h.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(240,80,220),2)
    for k,tok in enumerate(q['protected_instances']):
        actor=next(v for v in c['retained_instances'] if v['instance_token']==tok);bb=actor['boxes'][i]
        if bb is None:continue
        x0,y0,x1,y1=np.rint(bb).astype(int);main=tok in q.get('reveal_instances',q['protected_instances'])
        color=(35,245,120) if main else (75,165,250);label=('main B ' if main else 'keep ')+tok[:8]
        for im in [raw,plan]:
            cv2.rectangle(im,(x0,y0),(x1,y1),color,2);cv2.putText(im,label,(max(1,x0),max(15,y0-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
    x=np.asarray(Image.open(root/'rgb'/f'{i:05}.png').convert('RGB')).copy();x[h]=127
    return [raw,plan,x]

def main():
    O.mkdir(exist_ok=True,parents=True);cases={c['case_id']:c for c in read(T/'r28/prepared.json')['cases']}
    quality={c['case_id']:c['quality'] for c in read(T/'r32/instance_quality.json')['cases'] if c['technical_candidate']}
    pc=read(T/'r33/prepared.json')['cases'];cases.update({c['case_id']:c for c in pc});quality.update({c['case_id']:c['quality'] for c in pc})
    qa={c['case_id']:c for c in read(T/'r32/r32_independent_quality.json')['cases']}
    if (T/'r33/r33_independent_quality.json').exists():qa.update({c['case_id']:c for c in read(T/'r33/r33_independent_quality.json')['cases']})
    condition_qa=read(T/'r34/r34_independent_quality.json') if (T/'r34/r34_independent_quality.json').exists() else {}
    cards=[];manifest=[]
    for cid in ['Q035','Q046']+[c['case_id'] for c in pc]:
        c=cases[cid];q=quality[cid];dest=O/cid;dest.mkdir(exist_ok=True)
        clips=list(zip(*(panels(c,q,i) for i in range(30))));videos=[]
        for role,images in zip(['Y','plan','input'],clips):encode(dest/f'{role}.mp4',images);videos.append(f'{cid}/{role}.mp4')
        metrics=[]
        for tok in q['protected_instances']:
            ratio=q['model_H_occlusion_fractions'][tok];p=q['temporal_process']['protected'][tok]
            metrics.append(f'<tr><td>{esc(tok[:8])}</td><td>{"主显露 B" if tok in q.get("reveal_instances",[]) else "仅保留邻车"}</td><td>{max(ratio):.1%}</td><td>{p["approx_other_frame_support_mean"]:.1%}</td></tr>')
        status=qa.get(cid,{});grade=status.get('assistant_grade','待独立复核');reason=status.get('reason','技术通过不等于数据通过。')
        media='<div class="videos">'+''.join(f'<div><b>{name}</b><video src="{cid}/{role}.mp4" controls muted playsinline preload="none"></video></div>' for role,name in [('Y','真实原图：绿框主B，蓝框邻车'),('plan','黄＝合成A；紫＝最终删除洞H'),('input','真正带洞输入；并非生成失败')])+'</div>'
        if cid=='Q046':
            media+='<p>下面是送给模型的候选条件，尚未训练或运行新的 DriveEditor 补景。SAM只读最终H遮后的RGB；没有把Y中的隐藏车身复制进来。</p><div class="videos">'
            for role,name in [('visible_sam','遮后可见实例'),('state','有车O绿／背景N蓝／未知U灰'),('projected_rgb','局部外观投影；并非生成视频')]:
                path=dest/f'{role}.mp4';encode(path,(np.asarray(Image.open(T/'r34/state/Q046'/f'{i:05}_{role}.jpg').convert('RGB')) for i in range(30)))
                videos.append(f'{cid}/{role}.mp4');media+=f'<div><b>{name}</b><video src="{cid}/{role}.mp4" controls muted playsinline preload="none"></video></div>'
            media+='</div><p>主B：洞内身份投影精度95.3%，隐藏部分覆盖54.7%；N=0，洞内94.1%仍未知。两邻车只被轻微遮挡，投影边界精度分别26.5%／57.7%，需单独审查，联合精度会掩盖这个问题。</p>'
            media+=f'<p>条件独立质检：{esc(condition_qa.get("condition_assistant_grade","待复核"))}。{esc(condition_qa.get("reason",condition_qa.get("decision","")))}</p>'
            media+='<p>逐帧图已修复宽画面被拉成正方形的显示错误。下面按每辆车分别裁剪并保持比例；先看全景定位，再看同一身份的30帧。三列依次是真实Y（仅质检）、合法遮后输入＋可见SAM、该身份的局部投影。</p>'
            media+='<details><summary>全景定位：主B和两辆保留邻车</summary><div class="videos">'
            for frame in [0,15,29]:
                name=f'Q046_overview_{frame:02}.jpg';shutil.copy2(T/'r34'/name,dest/name)
                media+=f'<a href="{cid}/{name}"><img src="{cid}/{name}" loading="lazy" alt="Q046 frame {frame} crop locations"></a>'
            media+='</div></details>'
            for slot in [2,1,3]:
                for start in [0,15]:
                    name=f'Q046_slot{slot}_{start:02}_{start+14:02}.jpg';shutil.copy2(T/'r34'/name,dest/name)
                    media+=f'<details><summary>{"主B" if slot==2 else "保留邻车"} slot {slot} · f{start}–f{start+14}（点击图片可单独放大）</summary><a href="{cid}/{name}"><img src="{cid}/{name}" loading="lazy"></a></details>'
            for name in ['r34_independent_quality.json','review_provenance_check.json','old_sheet_crop_source_match.json','actor_identity_evaluation.json']:
                if (T/'r34'/name).exists():
                    shutil.copy2(T/'r34'/name,dest/name)
                    media+=f'<a href="{cid}/{name}">{esc(name)}</a> · '
        cards.append(f'<article id="{cid}" data-case="{cid}"><h2>{cid} · {esc(c["scene"])} · 数据AI评分 {grade}</h2><p>{esc(reason)}</p><table><tr><th>实例ID</th><th>角色</th><th>最大被遮比例</th><th>其他帧几何格支持（非纹理GT）</th></tr>{"".join(metrics)}</table>{media}<button class="play">同例全部视频同步播放／暂停</button><input class="seek" type="range" min="0" max="29" value="0"><label>人工评分 <select class="score"><option value="">未评</option><option>0</option><option>1</option><option>2</option></select></label><input class="note" placeholder="你的备注"></article>')
        manifest.append({'case_id':cid,'scene':c['scene'],'videos':videos,'assistant_grade':grade,'human_verdict':None})
    comparison=read(T/'r32/comparison.json');audit=read(T/'r33/proposal_audit.json')
    diagram='<div class="flow"><span>真实30帧Y<br>仅监督与质检</span> → <span>指定主B＋保留邻车<br>空间与显露检查</span> → <span>最终H完整遮挡<br>重新SAM</span> → <span>合法局部状态<br>O / N / U / F</span> ⇢ <span>同预算条件对照<br>尚未开始</span></div>'
    js="""const key='v77-r34-human-v1';let scores={};try{scores=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){};document.querySelectorAll('article').forEach(a=>{const id=a.dataset.case,v=[...a.querySelectorAll('video')],s=a.querySelector('.score'),n=a.querySelector('.note');s.value=scores[id]?.score??'';n.value=scores[id]?.note??'';const save=()=>{scores[id]={score:s.value,note:n.value};localStorage.setItem(key,JSON.stringify(scores))};s.onchange=save;n.oninput=save;a.querySelector('.play').onclick=()=>{const play=v[0].paused,t=v[0].currentTime;v.forEach(x=>{x.currentTime=t;play?x.play().catch(()=>{}):x.pause()})};a.querySelector('.seek').oninput=e=>v.forEach(x=>{x.pause();x.currentTime=Number(e.target.value)/10})});document.querySelector('#export').onclick=()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(scores,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='v77_r34_human_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};"""
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>v77 · 主保护车与合法条件复核</title><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{{background:#121923;color:#edf2f6;font:16px/1.6 system-ui;max-width:1580px;margin:30px auto;padding:0 20px}}p{{color:#bbcbdc}}h1{{font-size:28px}}h2{{font-size:22px}}article{{margin:24px 0;padding:20px;border:1px solid #41536b;border-radius:12px}}.videos{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px}}video,img{{width:100%;background:#080c12}}.flow{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}.flow span{{background:#263b55;padding:12px;border:1px solid #587999;border-radius:8px}}table{{border-collapse:collapse}}td,th{{text-align:left;padding:6px 20px;border-bottom:1px solid #40526a}}button,input,select{{background:#283a50;color:white;border:1px solid #60738c;border-radius:5px;padding:7px;margin:8px}}.seek{{width:30%}}.note{{width:35%}}.warn{{border-left:4px solid #edb268;background:#29291e;padding:12px}}a{{color:#86c7ff}}pre{{white-space:pre-wrap}}@media(max-width:850px){{.videos{{grid-template-columns:1fr}}}}</style>
    <h1>r31–r34：明确主保护车，检查条件实际能提供什么</h1><p class="warn">本页是数据和条件检查，未生成新的DriveEditor DELETE视频。没有新增训练；surfel尚未启动。AI评分与人工评分分开，人工默认未评。</p>{diagram}
    <p>旧规则误要求每辆轻微被遮的邻车也达到30%遮挡并发生显露。r32改成至少一辆主B满足显露，其余邻车仍须通过身份、深度与其他帧证据检查。87条轨迹和全部像素不变，技术候选1→3；Q035在人眼独立检查中因主B小而模糊被拒绝，Q046数据可继续。旧Q060保留。</p>
    <p>r33只尝试一套有界位置求解：围绕指定清楚B，反解静止A的深度和遮挡边缘，最多每B六个位置。372个解仅1个技术候选P001，仍来自Q046同一场景；独立检查发现其中一辆主B只有早期部分车身可见，不能把几何格支持当作完整纹理证据，故未准入。没有形成跨场景50条配方，不继续在同批来源追加偏移或速度网格。</p>
    <p>r35复核150个投影失败：134例碰ego底部约束、133例尺度／面积不合适、72例接近或穿过相机平面（原因可重叠）；只因水平入画／出画被拒绝的为0。不能靠放宽截边要求救回这批位置，保持空间约束。</p>
    <p><a href="../v77-target-protected-r28/index.html">87例原诊断与Q060条件</a> · <a href="../v77-target-protected-r26/index.html">保留的真实DELETE对照</a> · <a href="manifest.json">本页索引</a> · <a href="#Q046">直接看Q046</a></p><button id="export">导出人工评分</button>{''.join(cards)}
    <details><summary>r32配对计数与r33失败分布</summary><pre>{esc(json.dumps({'r32':comparison,'r33':audit['counts']},ensure_ascii=False,indent=2))}</pre></details><script>{js}</script></html>'''
    (O/'index.html').write_text(body,encoding='utf-8');(O/'manifest.json').write_text(json.dumps({'cases':manifest,'new_training_steps':0,'human_verdict':None},ensure_ascii=False,indent=2)+'\n')
    print('ROLE_REVIEW',len(manifest),sum(len(c['videos']) for c in manifest),flush=True)

if __name__=='__main__':main()

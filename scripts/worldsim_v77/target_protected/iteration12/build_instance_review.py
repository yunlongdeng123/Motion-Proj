"""r28输入诊断页：全量87例抽帧，按原因固定代表视频；不伪装成训练结果。"""
from pathlib import Path
import sys,json,html,subprocess
from collections import Counter
import numpy as np
import cv2
import imageio_ffmpeg
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).parent))
from prepare_instance_audit import O

WHY={
 'instance_identity_or_visibility_uncertain':'实例身份或可见性不确定',
 'nonvehicle_or_incomplete_protection':'触及非车辆，或保护对象缺少完整几何',
 'insufficient_actual_occlusion':'实际保护车遮挡不足',
 'actual_size':'实际轮廓过小',
 'protected_depth_order':'保护车深度顺序不成立',
 'static_foreground':'与静态前景的深度顺序不成立',
 'mask_continuity':'轮廓连续性不合格',
 'no_reveal_process':'没有充分显露或扫过过程',
 'insufficient_other_frame_evidence':'其他帧可见证据不足',
 'ego_band':'洞进入ego保护区',
 'unreviewed_dynamic_envelope':'仍有未核验动态实例',
 'protected_reveal':'保护车显露：待独立质检',
 'dense_known_background':'密集背景：待独立质检',
 'ordinary_background':'普通背景：待独立质检'}

def read(p):return json.loads(p.read_text())
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def esc(v):return html.escape(str(v))

def panels(c,i):
    folder=Path(c['observed_folder']);y=np.asarray(Image.open(Path(c['source_Y_quality_only'])/f'{i:05}.png').convert('RGB'))
    mask=lambda name:np.asarray(Image.open(folder/name/f'{i:05}.png'))>0
    a=mask('influence');h=mask('proposal_H');plan=y.copy();plan[a]=np.rint(plan[a]*.25+np.array([255,205,35])*.75).astype('uint8')
    cv2.drawContours(plan,cv2.findContours(h.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(245,85,255),2)
    x=np.asarray(Image.open(folder/'rgb'/f'{i:05}.png').convert('RGB')).copy();x[h]=127
    return [y,plan,x]

def make_contact(c,dest):
    out=Image.new('RGB',(1536,978),(18,24,31));draw=ImageDraw.Draw(out)
    for row,i in enumerate([0,15,29]):
        for col,(name,im) in enumerate(zip(['real Y (supervision only)','synthetic A yellow / final H magenta','final masked RGB'],panels(c,i))):
            draw.text((512*col+6,326*row+7),f'{c["case_id"]} f{i} '+name,fill='white')
            out.paste(Image.fromarray(im).resize((512,288)),(512*col,326*row+30))
    out.save(dest,quality=94)

def videos(c,dest):
    dest.mkdir(exist_ok=True);procs=[]
    if (dest/'encoding_success.json').exists() and all((dest/f'{n}.mp4').is_file() for n in ['Y','plan','input']):return
    for name in ['Y','plan','input']:
        path=dest/f'{name}.mp4'
        proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1024x576','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
        procs.append(proc)
    try:
        for i in range(30):
            for proc,im in zip(procs,panels(c,i)):proc.stdin.write(im.tobytes())
        for proc in procs:proc.stdin.close()
        for proc in procs:assert proc.wait()==0,'ffmpeg编码失败'
        dump(dest/'encoding_success.json',{'frames':30,'fps':10,'roles':['Y','plan','input']})
    finally:
        for proc in procs:
            if proc.poll() is None:proc.terminate();proc.wait()

def main():
    cv2.setNumThreads(1)
    plan=read(O/'prepared.json');quality=read(O/'instance_quality.json');assert quality['complete']
    checks=read(O/'input_validation.json');assert checks['all_passed']
    qa=read(O/'r28_independent_quality.json') if (O/'r28_independent_quality.json').exists() else {}
    r29=O.parent/'r29'
    state_qa=read(r29/'r29_independent_quality.json') if (r29/'r29_independent_quality.json').exists() else {}
    out=O/'review';out.mkdir(exist_ok=True);(out/'contacts').mkdir(exist_ok=True)
    cases={c['case_id']:c for c in plan['cases']};results={r['case_id']:r for r in quality['cases']}
    assert set(cases)==set(results)
    reasons={cid:(r['quality']['data_family'] if r['quality'] else r['reason']) for cid,r in results.items()}
    selected=set()
    for reason in sorted(set(reasons.values())):selected.add(min(cid for cid,v in reasons.items() if v==reason))
    selected.update(cid for cid,r in results.items() if r['technical_candidate'])
    cards=[];manifest=[]
    for cid,c in cases.items():
        r=results[cid];reason=reasons[cid];contact=out/'contacts'/f'{cid}.jpg';make_contact(c,contact)
        artifact={'case_id':cid,'scene':c['scene'],'split':c['split'],'technical_candidate':r['technical_candidate'],
            'reason':reason,'assistant_grade':None,'human_verdict':None,'contact':f'contacts/{cid}.jpg','videos':[]}
        if cid==qa.get('case_id'):artifact['assistant_grade']=qa['assistant_grade']
        media=''
        if cid in selected:
            videos(c,out/cid)
            for name in ['Y','plan','input']:artifact['videos'].append(f'{cid}/{name}.mp4')
            media='<div class="videos">'+''.join(f'<div><b>{label}</b><video controls preload="none" playsinline muted src="{cid}/{name}.mp4"></video></div>' for name,label in [('Y','真实 Y · 仅监督／核验'),('plan','计划遮挡 · 黄=A，紫=H'),('input','真正带洞输入 · 无生成结果')])+'</div><button class="play">三列同步播放／暂停</button><input class="seek" type="range" min="0" max="29" step="1" value="0">'
        if cid==state_qa.get('case_id'):
            metrics=next(v for v in read(r29/'condition_evaluation.json')['cases'] if v['case_id']==cid)
            parts=[]
            for name,label in [('visible_sam','最终遮后 SAM'),('state','O绿／N蓝／U灰'),('projected_rgb','投影可见纹理 · 并非生成结果')]:
                path=out/cid/f'state_{name}.mp4'
                if not path.exists():subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-framerate','10','-i',str(r29/'state'/cid/f'%05d_{name}.jpg'),'-frames:v','30','-c:v','libx264','-threads','1','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],check=True)
                artifact['videos'].append(f'{cid}/state_{name}.mp4')
                parts.append(f'<div><b>{label}</b><video controls preload="none" muted playsinline src="{cid}/state_{name}.mp4"></video></div>')
            media+=f'<p>r29：独立复核了全30帧，单例条件质量2分。相对Y-SAM，洞内O精度 {metrics["rendered_O_precision_inside_H"]:.1%}，覆盖隐藏B {metrics["hidden_B_coverage_by_O"]:.1%}；N占洞 {metrics["N_H_fraction"]:.1%}，U占洞 {metrics["unknown_H_fraction"]:.1%}。Y-SAM不是人工像素GT；没有证明模型DELETE改善。</p><div class="videos">'+''.join(parts)+'</div>'
            import shutil
            for start in [0,15]:
                name=f'{cid}_identity_{start:02}_{start+14:02}.jpg';shutil.copy2(r29/name,out/cid/name)
                media+=f'<details><summary>条件身份复核 f{start}–f{start+14}</summary><img loading="lazy" src="{cid}/{name}"></details>'
            artifact['condition_assistant_grade']=2
        note='技术候选通过，仍待独立抽帧检查；不是模型效果通过。' if r['technical_candidate'] else '当前不进入训练；下方给出质量检查证据。'
        if cid==qa.get('case_id'):note=f'独立输入质检 {qa["assistant_grade"]} 分；{qa["decision"]}'
        flags={'identity':r['identity_flags'],'scope':r['scope_flags'],'quality':r['quality']}
        cards.append(f'<article id="{cid}" data-id="{cid}" data-search="{esc(cid+" "+c["scene"]+" "+WHY.get(reason,reason))}"><h2>{cid} · {esc(c["scene"])} · {esc(WHY.get(reason,reason))}</h2><p>{note} {esc(c["split"])}；世界静止 A={str(c["trajectory"]["speed_mps"]==0)}；速度 {c["trajectory"]["speed_mps"]:g}m/s；GT最小间距 {c["trajectory"]["min_GT_clearance_m"]:.2f}m。</p>{media}<details><summary>查看 f0／f15／f29 的三列抽帧</summary><img loading="lazy" src="contacts/{cid}.jpg" alt="{cid} input contact"></details><details><summary>自动检查原始记录</summary><pre>{esc(json.dumps(flags,ensure_ascii=False,indent=2))}</pre></details><label>你的数据质量评分 <select class="score"><option value="">未评</option><option>0</option><option>1</option><option>2</option></select></label><input class="note" placeholder="人工备注（不会自动填写）"></article>')
        manifest.append(artifact);print('REVIEW',cid,'video' if cid in selected else 'contact',flush=True)
    counts='<table><tr><th>质量判定</th><th>候选条数</th></tr>'+''.join(f'<tr><td>{esc(WHY.get(k,k))}</td><td>{v}</td></tr>' for k,v in quality['counts'].items())+'</table>'
    diagram='''<div class="flow"><span>真实 30 帧 Y</span><i>→</i><span>冻结车道轨迹<br>轮廓 A / 洞 H</span><i>→</i><span>实例与空间质检<br>Y 只用于核验</span><i>→</i><span>合格候选独立 QA</span><i>→</i><span>最终 H 遮后 RGB<br>合法状态条件 / A·B·C</span></div>'''
    script='''const KEY='v77-r28-input-review-v1';let saved={};try{saved=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){};document.querySelectorAll('article').forEach(a=>{const id=a.dataset.id,s=a.querySelector('.score'),n=a.querySelector('.note'),v=[...a.querySelectorAll('video')];s.value=saved[id]?.score??'';n.value=saved[id]?.note??'';const put=()=>{saved[id]={score:s.value,note:n.value};localStorage.setItem(KEY,JSON.stringify(saved))};s.onchange=put;n.oninput=put;if(v.length){a.querySelector('.play').onclick=()=>{const t=v[0].currentTime,p=v[0].paused;v.forEach(x=>{x.currentTime=t;p?x.play().catch(()=>{}):x.pause()})};a.querySelector('.seek').oninput=e=>v.forEach(x=>{x.pause();x.currentTime=Number(e.target.value)/10})}});document.querySelector('#filter').oninput=e=>document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.search.toLowerCase().includes(e.target.value.toLowerCase()));document.querySelector('#export').onclick=()=>{const b=new Blob([JSON.stringify(saved,null,2)],{type:'application/json'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='v77_r28_human_input_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};'''
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 r28 输入工厂诊断</title><style>body{{background:#101720;color:#e4e9ef;font:16px/1.6 system-ui;margin:30px auto;max-width:1550px;padding:0 20px}}h1{{font-size:28px}}h2{{font-size:20px}}p{{color:#b8c8d9}}article{{margin:24px 0;padding:18px;border:1px solid #334455;border-radius:12px}}.videos{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}video,img{{width:100%;background:#080c11}}button,input,select{{padding:8px;margin:7px;background:#233142;color:#fff;border:1px solid #506276;border-radius:5px}}input.note{{width:50%}}.seek{{width:45%}}pre{{max-height:350px;overflow:auto;white-space:pre-wrap}}table{{border-collapse:collapse}}td,th{{padding:6px 22px;border-bottom:1px solid #3a4958;text-align:left}}.flow{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:20px 0}}.flow span{{background:#20344b;border:1px solid #6286af;padding:12px;border-radius:8px}}a{{color:#8cc9ff}}.warn{{border-left:4px solid #eea85d;padding:10px 16px;background:#292319}}[hidden]{{display:none!important}}@media(max-width:800px){{.videos{{grid-template-columns:1fr}}}}</style><h1>r28：固定候选的输入诊断</h1><p class="warn">本页不是新的 DriveEditor DELETE 结果。87 条候选、30 个来源场景、每条 30 帧；213 个 Y 评价实例标签已补齐。2,610 帧覆盖／像素隔离合同通过，不代表数据全部合格。训练步数为 0；人工分数全部留空。</p><p>保留原 72 条图像，仅补齐同一组既定轨迹中此前被“首个拒绝原因”漏记的 15 条。没有追加位置或速度搜索。恢复离开画面但已知的 3D 位姿，没有补造缺失轨迹。原 r21 入口继续保留，r26 洞裁减不推广。</p>{diagram}<p>流程最后两步仍受质量准入控制。完整 Y 与其 SAM 仅做离线质量标签；之后建立方法条件，必须重新使用“最终 H 遮后 RGB”。没有把完整 Y 的车辆纹理作为模型条件。</p>{counts}<p>视频按每种判定取编号最小一例，并包含所有技术候选；全部87例保留固定 f0／f15／f29 抽帧。三列分别为真实监督、遮挡计划、最终带洞输入，灰洞是送入模型前的空缺，并非补景失败。单帧抽查不能证明整段时序效果。</p><p><a href="../v77-target-protected-r26/index.html">上一轮真实 DELETE 对照（回退依据）</a> · <a href="manifest.json">完整索引</a> · <a href="#Q060">Q060：通过质检的条件样例</a></p><input id="filter" placeholder="搜索编号、场景或问题"><button id="export">导出人工评分</button>{''.join(cards)}<script>{script}</script></html>'''
    (out/'index.html').write_text(body,encoding='utf-8');dump(out/'manifest.json',{'cases':manifest,'counts':quality['counts'],'representative_video_cases':sorted(selected),'training_steps':0,'human_verdict':None})
    print('REVIEW_DONE',len(manifest),sum(len(c['videos']) for c in manifest),flush=True)

if __name__=='__main__':main()

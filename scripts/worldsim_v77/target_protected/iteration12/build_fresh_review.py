"""r38真实来源、合成洞及送模输入，保比例逐帧审核；不冒充生成效果。"""
from pathlib import Path
import sys,json,html,shutil
from collections import Counter
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageOps
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parents[1])]
from build_role_condition_review import encode,panels
from geometry_factory import read,dump
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r38';OUT=O/'review'
def esc(x):return html.escape(str(x))


def select(cases,quality):
    chosen=[];used=Counter();families=set()
    pool=sorted((r for r in quality if r['technical_candidate']),key=lambda r:(
        r['quality']['data_family']!='protected_reveal',r['scene'],r['case_id']))
    for row in pool:
        key=(row['scene'],row['quality']['data_family'])
        if used[row['scene']]>=2 or key in families:continue
        chosen.append(row['case_id']);used[row['scene']]+=1;families.add(key)
    selection={'selected':chosen,'technical_candidates':len(pool),'scenes':len(used),
        'policy':'at most two per scene, one per family; reveal first, then stable case order',
        'selection_model_outputs_used':False,'training_admission':0}
    dest=O/'review_selection.json'
    if dest.exists():assert read(dest)==selection
    else:dump(dest,selection)
    return chosen


def contact(c,q,dest):
    triplets=[panels(c,q,i) for i in range(30)]
    checks=[]
    for i,triplet in enumerate(triplets):
        y=np.asarray(Image.open(Path(c['source_Y_quality_only'])/f'{i:05}.png'))
        root=Path(c['observed_folder']);x=np.asarray(Image.open(root/'rgb'/f'{i:05}.png'))
        h=np.asarray(Image.open(root/'proposal_H'/f'{i:05}.png'))>0
        a=np.asarray(Image.open(root/'influence'/f'{i:05}.png'))>0
        assert np.array_equal(x[~h],y[~h]) and np.all(triplet[2][h]==127)
        assert h[a].all() and not h[512:].any()
        checks.append({'frame':i,'Y_X_source_exact':True,'synthetic_footprint_covered':True,'model_H_pixels':int(h.sum())})
    files=[]
    for start in [0,15]:
        sheet=Image.new('RGB',(3072,1002),(18,24,32));draw=ImageDraw.Draw(sheet)
        for j,i in enumerate(range(start,start+15)):
            col,row=j%4,j//4
            # 每格按原比例显示Y/A-H/X；不把多个actor的宽crop拉成正方形。
            for k,im in enumerate(triplets[i]):
                thumb=ImageOps.contain(Image.fromarray(im),(256,144))
                sheet.paste(thumb,(col*768+k*256,row*250+38))
            draw.text((col*768+5,row*250+7),f'{c["case_id"]} f{i}: real Y | A yellow / H purple | model X',fill='white')
            draw.text((col*768+5,row*250+190),'Green: main reveal B; blue: incidental preserved B',fill='white')
        name=f'full_{start:02}_{start+14:02}.jpg';sheet.save(dest/name,quality=94);files.append(name)
    # 保护车各自的全30帧裁剪，保留纵横比；缺投影显式留空。
    for tok in q['protected_instances']:
        actor=next(a for a in c['retained_instances'] if a['instance_token']==tok)
        for start in [0,15]:
            sheet=Image.new('RGB',(2400,660),(18,24,32));draw=ImageDraw.Draw(sheet)
            for j,i in enumerate(range(start,start+15)):
                xx,yy=(j%5)*480,(j//5)*220;bb=actor['boxes'][i]
                draw.text((xx+3,yy+3),f'f{i} {tok[:8]}: Y | model X',fill='white')
                if bb is None:continue
                b=np.asarray(bb);x0,y0=np.maximum(np.floor(b[:2]-16).astype(int),[0,0]);x1,y1=np.minimum(np.ceil(b[2:]+16).astype(int),[1024,576])
                if x1<=x0 or y1<=y0:continue
                for k,im in enumerate([triplets[i][0],triplets[i][2]]):
                    crop=ImageOps.contain(Image.fromarray(im[y0:y1,x0:x1]),(240,188))
                    sheet.paste(crop,(xx+k*240+(240-crop.width)//2,yy+27+(188-crop.height)//2))
            name=f'B_{tok[:8]}_{start:02}_{start+14:02}.jpg';sheet.save(dest/name,quality=95);files.append(name)
    dump(dest/'input_contract.json',{'case_id':c['case_id'],'frames':checks,'condition_or_model_result':False})
    return triplets,files


def main():
    cv2.setNumThreads(1);OUT.mkdir(exist_ok=True)
    cases={c['case_id']:c for c in read(O/'prepared.json')['cases']};result=read(O/'instance_quality.json')
    quality={r['case_id']:r['quality'] for r in result['cases']};selection=select(cases,result['cases'])
    qa={r['case_id']:r for r in read(O/'independent_quality.json')['cases']} if (O/'independent_quality.json').exists() else {}
    cards=[];manifest=[]
    for cid in selection:
        c=cases[cid];q=quality[cid];dest=OUT/cid;dest.mkdir(exist_ok=True)
        clips,images=contact(c,q,dest);videos=[]
        for k,role in enumerate(['Y','plan','input']):
            encode(dest/f'{role}.mp4',(p[k] for p in clips));videos.append(f'{cid}/{role}.mp4')
        score=qa.get(cid,{});grade=score.get('assistant_grade','待独立QA');reason=score.get('reason','技术规则通过，不能代替逐例视觉判断。')
        metrics=[]
        for tok in q['protected_instances']:
            curve=q['model_H_occlusion_fractions'][tok];proc=q['temporal_process']['protected'][tok]
            role='主显露B' if tok in q.get('reveal_instances',[]) else '保留邻车'
            metrics.append(f'<tr><td>{tok[:8]}</td><td>{role}</td><td>{min(curve):.1%}–{max(curve):.1%}</td><td>{proc["approx_other_frame_support_mean"]:.1%}</td></tr>')
        videohtml='<div class="grid">'+''.join(f'<div><b>{title}</b><video src="{cid}/{role}.mp4" controls muted playsinline preload="none"></video></div>' for role,title in [('Y','真实Y（仅质检／监督）：标出保护车'),('plan','合成A黄色／最终洞H紫边'),('input','实际带洞输入：灰色尚未补景')])+'</div>'
        links=''.join(f'<details><summary>{esc(name)}</summary><a href="{cid}/{name}"><img src="{cid}/{name}" loading="lazy"></a></details>' for name in images)
        cards.append(f'<article id="{cid}" data-case="{cid}"><h2>{cid} · {esc(c["scene"])} · {esc(c["split"])} · AI {grade}</h2><p>{esc(reason)}</p><p>{esc(c["camera"])}；{esc(q["data_family"])}；A速度{c["trajectory"]["speed_mps"]}m/s，30帧／3秒。世界坐标静止或车道运动由实际轨迹记录；不把平滑框当作纹理显露证明。</p><table><tr><th>实例</th><th>角色</th><th>H遮挡范围</th><th>跨帧框面支持代理</th></tr>{"".join(metrics)}</table>{videohtml}<button class="play">同步播放／暂停</button><input class="seek" type="range" min="0" max="29" value="0"><span class="frame">f0</span><label>人工评分<select class="score"><option value="">未评</option><option>0</option><option>1</option><option>2</option></select></label><input class="note" placeholder="人工备注">{links}</article>')
        manifest.append({'case_id':cid,'source_id':c['source_id'],'scene':c['scene'],'split':c['split'],
            'data_family':q['data_family'],'quality':q,'videos':videos,'images':[f'{cid}/{p}' for p in images],
            'assistant_grade':grade,'human_verdict':None,'model_inference':False})
    dump(OUT/'manifest.json',{'cases':manifest,'training_steps':0,'human_verdict':None})
    for name in ['instance_quality.json','candidate_roster.json','review_selection.json','run.json','independent_quality.json']:
        if (O/name).exists():shutil.copy2(O/name,OUT/name)
    followups=''
    if (T/'r41/summary.json').exists():
        evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
        for run,name in [('r38','S016_map_diagnosis.json'),('r39','proposal_audit.json'),('r40','paired_quality.json'),('r40','reuse_provenance_validation.json'),('r41','summary.json')]:
            shutil.copy2(T/run/name,evidence/f'{run}_{name}')
        followups='''<section class="notice"><h2>后续检查：r39–r41</h2><p>S016独立AI评分为0：落点有安全岛／草地区域疑点，未准入。地图脚印约96.7%位于lane、100%位于connector，但这不能覆盖图像语义疑点。其余两条同scene技术候选尚未独立质检，不作为合格数据。</p><p>r39用过去2秒真实位姿提出A，只得到两条空间可行轨迹，仍缺完整保护几何或其他帧显露依据。r40保持同一23例输入，只调整SAM提示帧；身份疑点从7例降到6例，但未增加可用样本。复用标签的实际提示帧／JPEG来源已纠正，180帧旧mask未改。</p><p>r41在旧候选池按“静止清楚B＋运动ego”筛选，7条可解析窗口的相机三秒行程都不足1米，没有达到预定的时序过程要求；没有为凑数放宽阈值。本页仍没有新增训练或真实DELETE生成结果。</p><p><a href="evidence/r38_S016_map_diagnosis.json">空间诊断</a> · <a href="evidence/r39_proposal_audit.json">过去位姿对照</a> · <a href="evidence/r40_paired_quality.json">提示帧成对结果</a> · <a href="evidence/r40_reuse_provenance_validation.json">旧mask溯源</a> · <a href="evidence/r41_summary.json">时序来源筛选</a></p></section>'''
    js="""const key='v77-r38-human-v1';let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){};document.querySelectorAll('article').forEach(a=>{const id=a.dataset.case,v=[...a.querySelectorAll('video')],s=a.querySelector('.score'),n=a.querySelector('.note');s.value=saved[id]?.score??'';n.value=saved[id]?.note??'';const save=()=>{saved[id]={score:s.value,note:n.value};localStorage.setItem(key,JSON.stringify(saved))};s.onchange=save;n.oninput=save;a.querySelector('.play').onclick=()=>{const go=v[0].paused,t=v[0].currentTime;v.forEach(x=>{x.currentTime=t;go?x.play().catch(()=>{}):x.pause()})};a.querySelector('.seek').oninput=e=>{const f=Number(e.target.value);a.querySelector('.frame').textContent='f'+f;v.forEach(x=>{x.pause();x.currentTime=f/10})}});document.querySelector('#export').onclick=()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(saved,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='v77_r38_human_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};"""
    diagram='<div class="flow"><span>新scene真实30帧<br>Y仅监督／质检</span> → <span>相机＋当地地图＋LiDAR<br>合成A空间检查</span> → <span>最终洞H<br>主B／邻车逐实例核验</span> → <span>合法遮后输入X<br>独立QA</span> ⇢ <span>状态条件与训练<br>尚未执行</span></div>'
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 r38 · 新来源遮挡数据审核</title><style>body{{background:#121923;color:#edf2f6;font:16px/1.6 system-ui;max-width:1580px;margin:30px auto;padding:0 20px}}article{{border:1px solid #41536b;padding:20px;border-radius:10px;margin:24px 0}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}video,img{{width:100%;background:#080c12}}.flow{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}.flow span{{background:#263b55;padding:12px;border:1px solid #587999;border-radius:8px}}p{{color:#bccedf}}a{{color:#86c7ff}}button,input,select{{background:#283a50;color:white;border:1px solid #60738c;border-radius:5px;padding:7px;margin:8px}}.seek{{width:25%}}table{{border-collapse:collapse}}td,th{{padding:6px 15px;border-bottom:1px solid #40526a}}pre{{white-space:pre-wrap}}.notice{{padding:12px;background:#30301c;border-left:4px solid #e5be68}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}</style><h1>r37–r38：新增真实来源，先检查遮挡过程</h1><p class="notice">本页是数据检查：灰色表示送模洞，尚未调用DriveEditor。没有新训练或真实DELETE收益结论。人工评分保持未评。</p>{diagram}<p>预先冻结18个新场景；30曝光／完整几何检查后保留15个场景450帧。17条初始SAM轨迹中15条通过技术连续性检查。新地图按各scene的Boston／Singapore location选择，没有套用旧Boston道路。</p><p>每来源沿用最多24个车道位置和0／3／6m/s；21条轨迹通过空间检查后，对最终H触及的每个实例补齐评价标签。全量技术结果：{esc(json.dumps(result['counts'],ensure_ascii=False))}。按预先固定每scene最多两例、每家族一例选择{len(selection)}例人工审核；数量不足如实保留，不降低标准凑数。</p><p>绿色是主显露B，蓝色是仅保留邻车，黄色是合成A轮廓，紫色是与真实DELETE统一的最终洞H。洞外仍是真实RGB；完整Y不进入未来条件。跨帧框面支持只是几何代理，不能证明隐藏纹理见过。</p><p><a href="instance_quality.json">21例完整技术分母</a> · <a href="run.json">冻结规则</a> · <a href="manifest.json">视频／逐帧图索引</a> · <a href="../v77-target-protected-r36/index.html">旧条件对照</a></p><button id="export">导出人工评分</button>{''.join(cards)}<script>{js}</script></html>'''
    body=body.replace('<button id="export">',followups+'<button id="export">')
    (OUT/'index.html').write_text(body,encoding='utf-8');print('FRESH_REVIEW',len(selection),len(selection)*3,flush=True)


if __name__=='__main__':main()

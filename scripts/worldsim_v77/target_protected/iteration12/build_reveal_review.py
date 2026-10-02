"""r23失败定位与r25三例可见保护修复的四列逐帧审核。"""
from pathlib import Path
import sys, html, json
from collections import Counter
sys.path.insert(0,str(Path(__file__).parent))
from refine_visible_hole import O,T,read,dump,PROBES
from render_reveal import video
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont


def main():
    result=read(O/'method_result.json');out=O/'review';(out/'contacts').mkdir(parents=True,exist_ok=True)
    qa_path=O/'r25_independent_quality.json'
    quality={v['case_id']:v for v in read(qa_path)['cases']} if qa_path.exists() else {}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19);rows=[];validation=[]
    for cid,run,srun in PROBES:
        meta=read(O/'observed'/cid/'observations.json');old=Path(meta['source_pair']);dest=O/'refined'/cid
        panels={k:[] for k in ['gt','labels','proposal','protected']};metrics=[];previews=[]
        for i in range(30):
            y=np.asarray(Image.open(old/'Y'/f'{i:03}.png').convert('RGB')).copy()
            core=np.asarray(Image.open(old/'influence'/f'{i:03}.png'))>0
            observed=np.asarray(Image.open(dest/'protected_visible'/f'{i:03}.png'))>0
            proposal=np.asarray(Image.open(dest/'proposed_H'/f'{i:03}.png'))>0
            h=np.asarray(Image.open(dest/'model_hole'/f'{i:03}.png'))>0
            cp=np.asarray(Image.open(dest/'condition'/f'{i:03}.png')).copy()
            prev=np.asarray(Image.open(old/'X'/f'{i:03}.png').convert('RGB')).copy();prev[proposal]=127
            label=y.copy();label[core]=np.rint(label[core]*.3+np.array([255,190,20])*.7).astype('uint8')
            cv2.drawContours(label,cv2.findContours(observed.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(45,235,110),2)
            # 旧完整Y mask只在此评价，不能进入refine_visible_hole.py。
            reference=np.zeros_like(core)
            for p in (old/'protected').glob(f'{i:03}_*.png'):reference|=np.asarray(Image.open(p))>0
            visible_reference=reference&~core;hidden_reference=reference&core
            assert h[core].all() and np.array_equal(cp[~h],y[~h])
            metrics.append({'frame':i,'old_hidden_visible_B':int((proposal&visible_reference).sum()),
                            'new_hidden_visible_B':int((h&visible_reference).sum()),
                            'target_influence_pixels':int(core.sum()),'target_leak':0,
                            'actually_hidden_B_pixels':int(hidden_reference.sum()),
                            'hidden_B_revealed_to_condition':int((hidden_reference&~h).sum())})
            for role,a in [('gt',y),('labels',label),('proposal',prev),('protected',cp)]:panels[role].append(a)
        asset=out/'assets'/cid;asset.mkdir(parents=True,exist_ok=True)
        for role,frames in panels.items():
            if not (asset/f'{role}.mp4').exists():video(asset/f'{role}.mp4',frames)
        sheet=Image.new('RGB',(1600,830),(18,24,32));draw=ImageDraw.Draw(sheet)
        for n,i in enumerate([0,15,29]):
            for j,(role,frames) in enumerate(panels.items()):
                draw.text((400*j+6,n*275+8),f'{cid} f{i} | {role}',font=font,fill='white')
                sheet.paste(Image.fromarray(frames[i]).resize((400,225)),(400*j,n*275+38))
        sheet.save(out/'contacts'/f'{cid}.jpg',quality=95)
        row={'case_id':cid,'scene':meta['scene'],'origin_run':run,'frame_count':30,'quality_reference':'Y-video SAM is approximate evaluation label only',
             'metrics':metrics,'videos':{r:f'assets/{cid}/{r}.mp4' for r in panels},'contact':f'contacts/{cid}.jpg',
             'assistant_quality':quality.get(cid,{'assistant_grade':None,'status':'pending'}),'human_verdict':None}
        rows.append(row);print('REVIEW',cid,sum(m['old_hidden_visible_B'] for m in metrics),sum(m['new_hidden_visible_B'] for m in metrics),flush=True)
    dump(out/'evaluation.json',{'cases':rows,'model_inference':False,'training_steps':0})
    cards=[]
    titles=['真实视频 Y（只作监督/评价）','黄：待删遮挡 A；绿：可见保护实例','外扩矩形 H 后的模型输入','保留可见保护实例后的模型输入']
    for r in rows:
        cid=r['case_id'];old=sum(m['old_hidden_visible_B'] for m in r['metrics']);new=sum(m['new_hidden_visible_B'] for m in r['metrics'])
        videos=''.join(f'<figure><figcaption>{title}</figcaption><video muted loop playsinline preload="metadata" controls src="{r["videos"][role]}"></video></figure>' for title,role in zip(titles,r['videos']))
        cards.append(f'''<section id="{cid}"><h2>{cid} · {html.escape(r['scene'])} · 来源 {r['origin_run']}</h2>
<p>可见 B 被洞误遮的标签像素总量：{old:,} → {new:,}（30帧累计，完整视频SAM仅作近似评价）。目标影响区全部覆盖，真正被 A 遮住的 B 没有泄漏进条件。此页展示输入修复，尚未展示新模型补景。</p>
<p>独立 gpt-6-sol xhigh 抽帧质量：{r['assistant_quality'].get('assistant_grade','待审')} / 2。只评价 f0 / f15 / f29 的输入覆盖和可见性；不代表新训练集准入、全视频通过或模型收益。人工分数仍为空。</p>
<div class="videos">{videos}</div><p><button class="play">同步播放/暂停</button> <input class="seek" type="range" min="0" max="29" value="0" step="1"> <span class="frame">帧 0 / 29</span></p>
<label>人工质量：<select class="score"><option value="">未评分</option><option>0</option><option>1</option><option>2</option></select></label> <input class="note" placeholder="备注：位置、邻车保持、mask覆盖、时序…">
<details><summary>固定 0 / 15 / 29 帧与限制</summary><img src="{r['contact']}" alt="{cid}三帧对照"><p>抽帧不能认证全段时序。绿色mask来自目标影响区先擦除的 X 上新跑的SAM2。隐藏Y没有进入mask构造。来源 r16 是旧DEV；R001 是 r23 唯一新候选，不能称50条新数据。</p></details></section>''')
    body=''.join(cards)
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>v77 · 可见保护车输入修复</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#111923;color:#e5edf5;margin:24px}h1{font-size:27px}section{background:#1a2735;padding:20px;margin:22px 0;border-radius:12px}.videos{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}figure{margin:0}figcaption{min-height:52px;font-size:14px}video,img{width:100%}button,select,input{font:inherit;padding:5px}.seek{width:45%}.note{width:50%}a{color:#8fcaff}.warn{border-left:4px solid #ffc365;padding:12px;background:#302b20}.flow{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.node{border:1px solid #7594b1;border-radius:7px;padding:10px;background:#22384c}table{border-collapse:collapse}td,th{border:1px solid #52677a;padding:7px}@media(max-width:950px){.videos{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>
<h1>v77：先修“洞误遮可见邻车”，再扩训练数据</h1>
<p>r23–r25 · 原模型与 r7 权重保留 · 本页新训练 0 步 · 人工分数由你填写</p>
<div class="warn">r23固定规则尝试61个场景，只产生2个同场景背景候选，按场景/类型去重后留下R001；独立QA给1分，未进入训练。这里定位并修复其可见保护边界，再用两个旧DEV样本检查隐藏区域没有泄漏。不能把输入修复当成新微调收益。</div>
<h2>本轮组件</h2><div class="flow"><div class="node">合成 X + 完整目标影响区 M</div>→<div class="node">先擦除 M<br>SAM2 只看可见实例</div>→<div class="node">H = 外扩洞 − 可见保护区<br>完整覆盖 M</div>→<div class="node">最终遮后 RGB<br>→ 合法状态 / DriveEditor</div></div>
<p>Y 保持真实原视频，仅在训练监督与独立评价读取。模型的真实输入不会包含灰色 synthetic actor 的纹理。本页第四列是修复后的<strong>输入</strong>，不是生成结果。</p>
<p>空间条件Adapter已通过真实30帧DriveEditor工程探针：原9通道不变，新增160,192个参数；零初始化前后输出差为0，8个零连接参数张量都有有限非零梯度，峰值约14.63GiB。身份分支、同预算微调和surfel尚未执行，条件收益仍未证明。</p>
<p><a href="../v77-target-protected-r21/index.html">r21：真实DELETE入口修复（包括A022）</a> · <a href="../v77-target-protected-r22/index.html">r22：合法投影条件覆盖</a> · <button id="export">导出本页人工评分</button></p>
__CARDS__
<h2>工厂为何不足50条</h2><p>道路轨迹的主要拒绝项为地面观测支持不足、窗口内出画/接近ego、车道长度不足和未分割实体包络交叠。随机增加速度/位置不会自动解决这些问题。三个定位例中，既有真实行人，也有未纳入保护列表的车辆；GT包络交叠仍须用可见实例核对。补全可见保护标注与合理来源覆盖之后才重开扩量。</p>
<script>const key='v77-r25-human-review';let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}for(const s of document.querySelectorAll('section')){const vs=[...s.querySelectorAll('video')],seek=s.querySelector('.seek'),score=s.querySelector('.score'),note=s.querySelector('.note');if(saved[s.id]){score.value=saved[s.id].score??'';note.value=saved[s.id].note??''}s.querySelector('.play').onclick=()=>{if(vs[0].paused){for(const v of vs){v.currentTime=vs[0].currentTime;v.play().catch(()=>{})}}else vs.forEach(v=>v.pause())};seek.oninput=()=>{vs.forEach(v=>{v.pause();v.currentTime=Number(seek.value)/10});s.querySelector('.frame').textContent='帧 '+seek.value+' / 29'};vs[0].ontimeupdate=()=>{if(!vs[0].paused){seek.value=Math.min(29,Math.round(vs[0].currentTime*10));s.querySelector('.frame').textContent='帧 '+seek.value+' / 29';for(const v of vs.slice(1))if(Math.abs(v.currentTime-vs[0].currentTime)>.12)v.currentTime=vs[0].currentTime}};const save=()=>{saved[s.id]={score:score.value===''?null:Number(score.value),note:note.value};try{localStorage.setItem(key,JSON.stringify(saved))}catch(e){}};score.onchange=save;note.oninput=save}document.getElementById('export').onclick=()=>{const data={run:'r25',role:'human_review',reviews:saved};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='v77_r25_human_review.json';a.click();URL.revokeObjectURL(a.href)}</script></html>'''
    page=template.replace('__CARDS__',body)
    if (T/'r26/assistant_output_review.json').exists():
        warning='<div class="warn"><strong>后续真实DELETE对照未通过，本洞裁减规则不推广。</strong>r26中A048/A034/A061出现更大的车形或保护车结构退化，继续使用r21默认入口。此页输入QA的2分不等于模型输出2分。<a href="../v77-target-protected-r26/index.html">查看8例新旧补景视频</a>。</div>'
        page=page.replace('<h2>本轮组件</h2>',warning+'<h2>本轮组件</h2>')
    (out/'index.html').write_text(page)


if __name__=='__main__':main()

"""CPU对应审核与后续原生/写回视频共用一个轻量页面。"""
from common import *
import html, shutil, re
import numpy as np
import torch
from PIL import Image, ImageDraw
from routing import pooled_owner

ARCH='''<svg viewBox="0 0 1120 250" role="img" aria-label="参考空间绑定模块图">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#83b5d0"/></marker></defs>
<g fill="#15334b" stroke="#658aa9"><rect x="10" y="10" width="175" height="55" rx="5"/><rect x="230" y="10" width="175" height="55" rx="5"/><rect x="10" y="98" width="175" height="55" rx="5"/><rect x="230" y="98" width="175" height="55" rx="5"/><rect x="470" y="30" width="205" height="65" rx="5"/><rect x="470" y="155" width="205" height="55" rx="5"/><rect x="730" y="83" width="180" height="75" rx="5"/><rect x="955" y="93" width="155" height="55" rx="5"/></g>
<g fill="#ecf3fa" font-size="15" text-anchor="middle" font-family="Arial,Microsoft YaHei"><text x="98" y="32">已有参考 RGB</text><text x="98" y="54">已擦除待删 A</text><text x="317" y="32">冻结 VAE</text><text x="317" y="54">原参考编码器</text><text x="98" y="120">pose / tracks / LiDAR</text><text x="98" y="142">可靠来源</text><text x="317" y="120">实例 + 局部位置</text><text x="317" y="142">未知不强行绑定</text><text x="572" y="53">RGB 交叉注意力</text><text x="572" y="76">仅增加软空间偏置</text><text x="572" y="177">原 BEV / 2D 分支</text><text x="572" y="198">沿用现有编码</text><text x="820" y="108">DriveEditor</text><text x="820" y="129">目标 RGB + 固定 H</text><text x="820" y="149">主干冻结</text><text x="1032" y="116">原生 DELETE</text><text x="1032" y="138">固定 α 写回</text></g>
<g fill="none" stroke="#83b5d0" stroke-width="2" marker-end="url(#arrow)"><path d="M185 38H230"/><path d="M405 38H470"/><path d="M185 126H230"/><path d="M405 126H440V80H470"/><path d="M185 146H205V182H470"/><path d="M675 63H700V105H730"/><path d="M675 182H700V140H730"/><path d="M910 121H955"/></g></svg>'''

JS='''function frame(e){const a=e.closest('article'),f=e.value;a.querySelectorAll('[data-kind]').forEach(im=>{const p=`assets/${a.id}/${im.dataset.kind}_${f.padStart(2,'0')}.jpg`;im.src=p;im.parentElement.href=p});a.querySelector('[data-frame-summary]').textContent=JSON.parse(a.dataset.frames)[f]}
function play(b){b.closest('article').querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})})}
function pause(b){b.closest('article').querySelectorAll('video').forEach(v=>v.pause())}
const key='v77-r49-human';let scores={};try{scores=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
document.querySelectorAll('[data-score]').forEach(e=>{e.value=scores[e.dataset.score]?.score||'';e.onchange=()=>store(e)});
document.querySelectorAll('[data-note]').forEach(e=>{e.value=scores[e.dataset.note]?.note||'';e.onchange=()=>store(e)});
function store(e){const id=e.dataset.score||e.dataset.note,a=e.closest('article');scores[id]={score:a.querySelector('[data-score]').value,note:a.querySelector('[data-note]').value};localStorage.setItem(key,JSON.stringify(scores))}
function exportScores(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({run:'r49',rows:scores},null,2)],{type:'application/json'}));a.download='r49_human_review.json';a.click();URL.revokeObjectURL(a.href)}'''


def color(owner,main):
    out=np.full((*owner.shape,3),[100,110,120],np.uint8)
    out[owner==-1]=[65,155,255];out[owner>0]=[235,155,65]
    if main:out[owner==main]=[45,210,125]
    return out


def marked(image,mask):
    im=Image.fromarray(image.copy());draw=ImageDraw.Draw(im)
    yy,xx=np.where(mask)
    if len(xx):draw.rectangle((int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())),outline='#ffe000',width=3)
    return im


def figures(cid,req,meta,check,dest):
    main=meta['identities'].get(meta['main_protected_token'],0);route=req.routing_metadata
    for slot in range(6):
        Image.fromarray(req.priors['references'][slot]).save(dest/f'reference_{slot:02}.png')
        owner=route['routing_reference_owner'][slot]
        rgb=req.priors['references'][slot].copy()
        tint=np.repeat(np.repeat(color(owner,main),32,axis=0),32,axis=1)
        rgb=(rgb*.67+tint*.33).round().astype('uint8');im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
        for y in range(8):
            for x in range(8):
                draw.rectangle((x*32,y*32,(x+1)*32-1,(y+1)*32-1),outline='#7a8694')
                if owner[y,x]!=0:
                    label=str(slot*64+y*8+x)
                    draw.rectangle((x*32+1,y*32+1,x*32+23,y*32+12),fill='#14202a')
                    draw.text((x*32+2,y*32+1),label,fill='white')
        im.save(dest/f'reference_{slot:02}_patches.png')
    cond=req.branch_condition(torch.zeros(6,4,32,32),'cpu')
    coarse,q=pooled_owner(cond['routing_query_owner'],cond['routing_query_q'],(18,32))
    coarse=coarse.numpy().reshape(10,18,32)
    for f in (0,5,9):
        original=marked(req.target_rgb[f],req.edit_mask[f]);original.save(dest/f'original_{f:02}.jpg',quality=95)
        owner=route['routing_query_owner'][f,0]
        display=np.repeat(np.repeat(color(owner,main),4,axis=0),4,axis=1)
        # 条件地图同时保留RGB的部分轮廓，色块始终是proxy。
        im=marked((req.target_rgb[f]*.25+display*.75).round().astype('uint8'),req.edit_mask[f])
        im.save(dest/f'owner_{f:02}.jpg',quality=95)
        safe=req.target_rgb[f].copy();safe[req.edit_mask[f]]=127
        im=marked(safe,req.edit_mask[f]);draw=ImageDraw.Draw(im)
        Hc=torch.nn.functional.adaptive_max_pool2d(cond['hole'],(18,32))[f,0].numpy()>0
        for yy,xx in zip(*np.where(Hc)):
            key=coarse[f,yy,xx];c=tuple(color(np.array(key),main).tolist())
            draw.rectangle((xx*32,yy*32,xx*32+31,yy*32+31),outline=c,width=3)
        im.save(dest/f'grid_{f:02}.jpg',quality=95)
        # 点编号对应上方参考格编号；只画3个实际可见对应，避免满屏连线。
        im=marked(safe,req.edit_mask[f]);draw=ImageDraw.Draw(im)
        ids=np.flatnonzero((route['routing_reference_owner'].ravel()==main)&(route['routing_projected_q'][f]>0)) if main else np.array([],int)
        if len(ids):ids=ids[np.unique(np.linspace(0,len(ids)-1,min(3,len(ids))).round().astype(int))]
        for i in ids:
            p=route['routing_projected_uv'][f,i]*[1024,576]
            rad=np.maximum(route['routing_projected_radius'][f,i]*[1024,576],[32,32])
            draw.ellipse((p[0]-rad[0],p[1]-rad[1],p[0]+rad[0],p[1]+rad[1]),outline='#d68bff',width=2)
            draw.rectangle((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill='#ffffff')
            draw.text((p[0]+6,p[1]-12),str(i),fill='#f4baff',stroke_fill='black',stroke_width=1)
        im.save(dest/f'projection_{f:02}.jpg',quality=95)


def video(cid,name,label):
    return f'<figure><figcaption>{html.escape(label)}</figcaption><video controls muted playsinline preload="metadata" src="assets/{cid}/{name}.mp4"></video></figure>'


def main():
    torch.set_num_threads(1);out=O/'review';out.mkdir(exist_ok=True)
    byid=cases();checks={x['case_id']:x for x in read(O/'input_checks.json')['cases']}
    state=read(O/'controller_state.json');cards=[];new_video_count=0
    for cid in EVAL_IDS+TRAIN_IDS:
        c=byid[cid];req=request(c);dest=out/'assets'/cid;dest.mkdir(parents=True,exist_ok=True)
        meta=read(O/'inputs'/cid/'routing.json');check=checks[cid]
        figures(cid,req,meta,check,dest)
        for source,name in [(O/'inputs'/cid/'check.json','input_check.json'),(O/'inputs'/cid/'routing.json','routing.json')]:
            shutil.copy2(source,dest/name)
        summaries={str(f):f'f{f:02}：洞内 {check["attention_scales"][0]["frames"][f]["H_queries"]} 个粗query；主B {check["attention_scales"][0]["frames"][f]["main_B_queries_H"]}，背景N {check["attention_scales"][0]["frames"][f]["known_N_queries_H"]}，未知U {check["attention_scales"][0]["frames"][f]["unknown_queries_H"]}。实际空间patch {check["query_frames"][f]["main_B_spatial_patches"]}。' for f in (0,5,9)}
        body=f'<h2>{cid} · {html.escape(c["scene"])} · {c["split"]}</h2>'
        body+=f'<p>删除目标 A：<code>{c["target_token"]}</code>；主保护 B：<code>{meta["main_protected_token"] or "无主保护车参考"}</code>。沿用原10帧、原mask和原参考。</p>'
        body+='<p>显示帧 <select onchange="frame(this)"><option value="0">f00</option><option value="5" selected>f05</option><option value="9">f09</option></select> <span data-frame-summary>'+summaries['5']+'</span></p>'
        body+='<div class="four">'
        for name,label in [('original','原RGB / synthetic-X；黄框是H包络'),('owner','身份条件：绿主B / 橙其他车 / 蓝N / 灰U'),('grid','实际18×32查询格；黄框内仍有大量未知'),('projection','最多3个主B参考patch的位置对应')]:
            body+=f'<figure><figcaption>{label}</figcaption><a href="assets/{cid}/{name}_05.jpg" target="_blank"><img data-kind="{name}" src="assets/{cid}/{name}_05.jpg"></a></figure>'
        body+='</div><p>绿/橙来自GT框面代理，不是精确silhouette。灰色是未知条件，不是补景结果。紫色圈表示投影patch尺度；点编号对应参考图网格。没有可见对应的同实例patch只保留身份信息。</p>'
        body+='<details open><summary>实际6张参考与8×8 patch归属（每行：原参考 / 对应标签）</summary><div class="refs">'
        for slot,row in enumerate(meta['source_slots']):
            info=req.reference_manifest['references'][slot]
            label=f'slot{slot} · {info["camera"]} · '+('无效padding' if info['padding'] else info['role'])
            label+=f'；主声明patch {row.get("declared_actor_patches",0)} / 64；已知背景 {row["known_background_patches"]}。'
            body+=f'<figure><figcaption>{html.escape(label)}</figcaption><div class="pair"><img loading="lazy" src="assets/{cid}/reference_{slot:02}.png"><img loading="lazy" src="assets/{cid}/reference_{slot:02}_patches.png"></div></figure>'
        body+='</div></details>'
        body+='<details><summary>各注入尺度的实际绑定覆盖</summary><table><tr><th>query尺度</th><th>f00 主B/H</th><th>f05 主B/H</th><th>f09 主B/H</th></tr>'
        for scale in check['attention_scales']:
            body+='<tr><td>'+str(scale['size'])+'</td>'+''.join(f'<td>{scale["frames"][f]["main_B_queries_H"]}/{scale["frames"][f]["H_queries"]}</td>' for f in (0,5,9))+'</tr>'
        body+='</table><p>低分辨率格中身份/背景混合时为U。空间覆盖稀疏是本实验边界，不能当成所有后车部位已绑定。</p></details>'
        if not any(x['known_actor_patches'] or x['known_background_patches'] for x in meta['source_slots']):
            body+='<p class="warn">本例参考没有能可靠标记的纯patch；当前路由不施加偏置。它仍参与原任务，但不提供空间绑定的训练信号。</p>'
        if cid in EVAL_IDS:
            source=PARENT/'review/gpu_assets'/cid
            for name in ('original','baseline','baseline_native','RGB_and_geometry','RGB_and_geometry_native'):
                if not (dest/f'{name}.mp4').exists():shutil.copy2(source/f'{name}.mp4',dest/f'{name}.mp4')
            old64=CONTROL/'review/assets'/cid
            for native in (False,True):
                name='r48_64'+('_native' if native else '')
                if not (dest/f'{name}.mp4').exists():shutil.copy2(old64/f'{name}.mp4',dest/f'{name}.mp4')
            available=[]
            for stage in ('zero','step64'):
                for arm in ('correct','wrong','no_RGB','routing_off'):
                    folder=O/'evaluation'/stage/cid/arm
                    if not (folder/'result.json').exists():continue
                    from iteration15.review import encode
                    available.append((stage,arm));shutil.copy2(folder/'result.json',dest/f'{stage}_{arm}.json')
                    for native in (False,True):
                        name=stage+'_'+arm+('_native' if native else '')
                        if not (dest/f'{name}.mp4').exists():encode(folder,dest/f'{name}.mp4',native)
                        new_video_count+=1
            body+='<h3>历史视频对照 · r49未跑时仅作背景</h3><div class="three">'+video(cid,'original','原视频 · H边界')+video(cid,'baseline','r46官方原权重 + r21完整SAM')+video(cid,'RGB_and_geometry','r47已有条件分支')+'</div>'
            body+='<p><button onclick="play(this)">同步从头播放</button> <button onclick="pause(this)">暂停</button></p>'
            if not available:body+='<p class="pending">r49还未进行GPU推理，训练0步；本页的新增图片全部是CPU条件审核图。</p>'
            for stage in ('zero','step64'):
                if (stage,'correct') not in available:continue
                body+=f'<h3>r49 {stage} · 固定同权重、同输入对照</h3><div class="three">'+video(cid,'baseline','r46')+video(cid,'RGB_and_geometry','r47')+video(cid,stage+'_correct','r49 '+stage+' 空间绑定')+'</div>'
                body+='<details open><summary>原生/写回与参考控制</summary><div class="three">'+video(cid,stage+'_correct_native','r49原生')+video(cid,stage+'_correct','r49固定α写回')+'</div>'
                body+='<div class="three">'+''.join(video(cid,stage+'_'+arm,arm) for arm in ('wrong','no_RGB','routing_off') if (stage,arm) in available)+'</div></details>'
            body+='<details><summary>历史原生、r48同64步预算控制</summary><div class="three">'+video(cid,'baseline_native','r46原生')+video(cid,'RGB_and_geometry_native','r47原生')+video(cid,'r48_64','r48同64步预算控制')+'</div>'+video(cid,'r48_64_native','r48_64原生')+'</details>'
            body+=f'<p>本轮人工分 <select data-score="{cid}"><option value="">未评</option><option>0</option><option>1</option><option>2</option></select> <input data-note="{cid}" placeholder="后车轮廓 / 薄膜 / 幻觉 / 时序"></p>'
        body+=f'<p><a href="assets/{cid}/input_check.json">完整逐帧覆盖</a> · <a href="assets/{cid}/routing.json">身份与参考元数据</a></p>'
        cards.append(f'<article id="{cid}" data-frames="{html.escape(json.dumps(summaries,ensure_ascii=False),quote=True)}">{body}</article>')
    for name in ('preflight.json','input_checks.json','controller_state.json'):
        shutil.copy2(O/name,out/name)
    # Git仅保存轻量配置，页面另放删掉case大元数据的可读配置。
    compact=read(O/'manifest.json');compact['cases']=[{k:c[k] for k in ('case_id','scene','split','kind')} for c in compact['cases']]
    dump(out/'manifest.json',compact)
    nav=' · '.join(f'<a href="#{cid}">{cid}</a>' for cid in EVAL_IDS+TRAIN_IDS)
    page='''<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>r49 · 参考空间绑定准备与审核</title>
<style>body{font:15px/1.6 Arial,"Microsoft YaHei",sans-serif;background:#0a1622;color:#e5edf4;margin:0;padding:22px}main{max-width:1650px;margin:auto}a{color:#8bd5ff}article,header{background:#112333;padding:20px;margin:18px 0;border-radius:10px}h1,h2,h3{margin:4px 0 14px}p{margin:10px 0}code{overflow-wrap:anywhere;color:#b4e2ff}.three,.four,.refs{display:grid;gap:12px}.three{grid-template-columns:repeat(3,minmax(0,1fr))}.four{grid-template-columns:repeat(4,minmax(0,1fr))}.refs{grid-template-columns:repeat(3,minmax(0,1fr))}figure{margin:0}img,video,svg{width:100%;height:auto;background:#14283c}figcaption{font-size:13px;min-height:42px}.pair{display:flex;gap:4px}.pair img{width:calc(50% - 2px)}summary{cursor:pointer;color:#bce2ff;margin:10px 0}input,select,button{background:#1d3d55;color:white;padding:6px;border:1px solid #6385a0;border-radius:4px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #45617a;padding:6px;text-align:left}.warn,.pending{background:#493d19;padding:10px}.legend{color:#b2c4d3}@media(max-width:900px){.three,.four,.refs{grid-template-columns:1fr 1fr}}@media(max-width:550px){.three,.four,.refs{grid-template-columns:1fr}}</style><main>'''
    page+='<header><h1>r49 · 参考 → 对应恢复位置</h1><p><b>当前阶段：</b>'+html.escape(state['stage'])+'。新增GPU窗口 '+str(state.get('new_inference_windows',0))+'；训练 '+str(state['training_steps'])+' 步。</p>'
    page+='<p>先验证“正确车身信息到正确位置”。新增逻辑仅是RGB交叉注意力的无参数软偏置；基线权重从r47 step320加载。r46继续保留为默认。输入、mask、几何和固定α沿用原版。</p>'+ARCH
    page+='<p class="legend">黄：删除mask的包络。绿：主保护车B。橙：其他保留车。蓝：实际LiDAR支持背景N。灰：未知U。身份与位置来自3D框面proxy，不代表真实silhouette；未知区域不作无车判断。</p>'
    page+='<p>CPU页面包含9例的f00/f05/f09、6张真实输入参考、对应坐标、逐帧覆盖。先跑9个零训练窗口；直接看图后至多一次64步。A034/A061需同时保住后车并减少薄膜；A022与M003/M006检查回退。正确/错配只换互为有效的主B外观patch。</p>'
    page+='<p>'+nav+'</p><p><a href="preflight.json">CPU检查</a> · <a href="manifest.json">冻结配置</a> · <a href="input_checks.json">逐帧/逐尺度记录</a> · <button onclick="exportScores()">导出人工记录</button></p></header>'
    page+=''.join(cards)+'<script>'+JS+'</script></main></html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    links=re.findall(r'(?:src|href)="([^"#]+)"',page)
    assert all((out/p).is_file() for p in links), [p for p in links if not (out/p).is_file()]
    dump(O/'review_check.json',{'case_cards':len(cards),'links_exist':len(links),'new_GPU_videos':new_video_count,
        'all_media_exist':True,'GPU_jobs_started_by_report':0,'human_verdict':None})
    print('HTML_READY',out,flush=True)


if __name__=='__main__':main()

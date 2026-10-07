"""复用现有多列审核方式；CPU页只显示实际输入，不冒充生成结果。"""
from common import *
import argparse, html, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

LABELS = {'film_or_ghost':'薄膜 / 多余轮廓','protected_actor_deformation':'后车变形',
          'car_body_extends_onto_road':'车身向道路延伸','actor_regeneration':'重新生车',
          'mask_failure':'mask身份 / 覆盖问题','none_visible':'该帧未见明显结构问题','uncertain':'不能确定'}


def encode(source, output, ext='png'):
    if output.exists():return
    subprocess.run([ffmpeg_binary(),'-v','error','-y','-threads','1','-framerate','10','-i',str(source/f'%05d.{ext}'),
                    '-frames:v','10','-c:v','libx264','-threads','1','-filter_threads','1','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(output)],check=True)


def annotate(image, case, frame):
    image=image.copy();draw=ImageDraw.Draw(image);f=case['frames'][frame]
    b=case.get('protected_actor')
    if b:
        actor=next((a for a in f['neighbors'] if a['instance_token']==b['instance_token']),None)
        if actor:
            draw.rectangle(actor['box_xyxy'],outline=(50,235,120),width=2)
            x,y=actor['box_xyxy'][:2];draw.text((max(0,x),max(0,y-13)),'B protected proxy',fill=(50,235,120))
    box=f['target']['box_xyxy'];draw.rectangle(box,outline=(255,220,40),width=2)
    x,y=box[:2];draw.text((max(0,x),max(0,y-13)),case['case_id']+' DELETE A',fill=(255,220,40))
    return image


def generate_assets(case, cpu_only):
    cid=case['case_id'];folder=Path(case['folder']);dest=O/'review'/cid;dest.mkdir(parents=True,exist_ok=True)
    if cpu_only and (dest/'original.mp4').exists() and (dest/'inputs_f00_f05_f09.jpg').exists():return
    if not (dest/'original.mp4').exists():
        seq=dest/'boxed';seq.mkdir(exist_ok=True)
        for i in range(10):
            annotate(Image.open(folder/'rgb'/f'{i:05}.jpg').convert('RGB'),case,i).save(seq/f'{i:05}.jpg',quality=96)
        encode(seq,dest/'original.mp4',ext='jpg')
    canvas=Image.new('RGB',(1536,720),(18,24,34));d=ImageDraw.Draw(canvas)
    for k,i in enumerate([0,5,9]):
        rgb=Image.open(folder/'rgb'/f'{i:05}.jpg').convert('RGB')
        canvas.paste(annotate(rgb,case,i).resize((512,288)),(k*512,25))
        box=case['frames'][i]['target']['box_xyxy'];x0,y0,x1,y1=map(int,box)
        # 只裁真实RGB；按原比例显示，不生成或锐化目标。
        crop=rgb.crop((max(0,x0-12),max(0,y0-12),min(1024,x1+12),min(576,y1+12)))
        crop.thumbnail((500,260));canvas.paste(crop,(k*512+6,343))
        d.text((k*512+6,4),f'{cid} original f{i:02}',fill='white')
        d.text((k*512+6,321),'Target A original RGB crop',fill=(255,220,40))
    if case['best_protected_reference']:
        ref=case['best_protected_reference'];fr=ref['frame']
        im=Image.open(folder/'rgb'/f'{fr:05}.jpg').convert('RGB');box=ref['box_xyxy']
        crop=im.crop(tuple(map(int,box)));crop.save(dest/'protected_reference.jpg',quality=95)
        d.text((6,630),f"B reference f{fr:02}; A/B box overlap {ref['target_bbox_overlap_fraction']:.1%}. Geometry proxy, visually inspect RGB.",fill=(70,240,150))
    else:d.text((6,630),'No qualifying behind-car bbox proxy. This is not proof that all hidden content is background.',fill='white')
    canvas.save(dest/'inputs_f00_f05_f09.jpg',quality=93)
    if cpu_only:return
    if (folder/'mask_result.json').exists():
        seq=dest/'mask_overlay';seq.mkdir(exist_ok=True)
        for i in range(10):
            im=np.array(Image.open(folder/'rgb'/f'{i:05}.jpg').convert('RGB'))
            mask=np.array(Image.open(folder/'model_mask'/f'{i:05}.png'))>0
            core=np.array(Image.open(folder/'sam'/f'{i:05}.png'))>0
            im[mask]=np.rint(im[mask]*.6+np.array([50,120,255])*.4).astype('uint8')
            cv2.drawContours(im,cv2.findContours(core.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(255,220,40),1)
            Image.fromarray(im).save(seq/f'{i:05}.jpg',quality=96)
        encode(seq,dest/'model_input.mp4',ext='jpg')
    if (folder/'result.json').exists():
        encode(folder/'native',dest/'native.mp4');encode(folder/'compose',dest/'delete.mp4')
        rows=[]
        for i in [0,5,9]:
            panels=[Image.open(folder/'rgb'/f'{i:05}.jpg').convert('RGB'),
                    Image.open(folder/'native'/f'{i:05}.png'),Image.open(folder/'compose'/f'{i:05}.png')]
            rows.append((i,panels))
        board=Image.new('RGB',(1536,954),(18,24,34));draw=ImageDraw.Draw(board)
        for r,(i,panels) in enumerate(rows):
            for k,im in enumerate(panels):
                draw.text((k*512+6,r*318+4),f'{cid} f{i:02} '+['original','native generation','fixed writeback'][k],fill='white')
                board.paste(im.resize((512,288)),(k*512,r*318+24))
        board.save(dest/'structure_review.jpg',quality=95)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu-only',action='store_true');args=p.parse_args()
    m=read(O/'manifest.json');root=O/'review';root.mkdir(exist_ok=True);cards=[]
    assistants=read(O/'assistant_reviews.json') if (O/'assistant_reviews.json').exists() else {}
    for c in m['cases']:
        generate_assets(c,args.cpu_only);cid=c['case_id'];a=assistants.get(cid,{})
        columns=[]
        for name,title in [('original','原视频：黄框目标 A / 绿框后车 B'),('model_input','SAM实例与模型洞'),('native','官方 DriveEditor 原生补景'),('delete','r46 规则最终 DELETE')]:
            link=f'{cid}/{name}.mp4'
            pending='输入受限；本轮不推理' if c.get('structural_audit_eligible') is False else '尚未运行；等待 GPU'
            content=f'<video controls muted preload="none" src="{link}"></video>' if (root/link).exists() else f'<p class="pending">{pending}</p>'
            columns.append(f'<div><b>{title}</b>{content}</div>')
        ref=c['best_protected_reference'];reftext='未找到达到尺寸门槛的后车投影；不能据此断言隐藏区域无车。'
        if ref:
            reftext=f"本窗口 B 最少 A 框重叠的参考：f{ref['frame']:02}，框重叠 {ref['target_bbox_overlap_fraction']:.1%}。这是几何近似，RGB 是否清楚须看图。"
        known=c.get('input_visual_review','pending');gpu=Path(c['folder'])/'result.json'
        tags=' / '.join(c['difficulty_factors']) or '未触发当前尺寸、可见度、亮度代理疑点'
        options=''.join(f'<option value="{k}">{v}</option>' for k,v in LABELS.items())
        note=html.escape(a.get('note','生成尚未运行；不填写模型失败标签。' if not gpu.exists() else '待单帧结构粗分类。'))
        image_links=f'<a href="{cid}/inputs_f00_f05_f09.jpg">f00 / f05 / f09 输入与目标 crop</a>'
        if (root/cid/'structure_review.jpg').exists():image_links+=f' · <a href="{cid}/structure_review.jpg">原图 / 原生 / 写回结构对照</a>'
        if ref:image_links+=f' · <a href="{cid}/protected_reference.jpg">真实 B 参考</a>'
        cards.append(f'''<article id="{cid}" data-scene="{c['scene']}"><h2>{cid} · {c['scene']} · {c['camera']}</h2>
<p>本次只删除 <code>{c['instance_token']}</code>；同场景其他目标在独立副本运行。尺寸中位数 {c['median_width']:.0f}×{c['median_height']:.0f}px；输入难度代理：{c['input_difficulty']}。</p>
<p>{html.escape(tags)}；输入人工/助手图像检查：{html.escape(known)}。</p>
<p>本轮结构排查：{'输入限制，暂不进GPU队列' if c.get('structural_audit_eligible') is False else '输入目标可辨；SAM身份尚待GPU检查'}。用户分数留空，不能把输入限制当模型失败。</p>
<div class="columns">{''.join(columns)}</div><button onclick="playCase(this)">四列同步播放</button>
<details><summary>查看实际目标及参考证据</summary><p>{reftext} 参考 crop 只用于本页排查；r46 模型输入仍是遮后视频，不新增外部 RGB 分支。</p><p>{image_links}</p><img loading="lazy" src="{cid}/inputs_f00_f05_f09.jpg"></details>
<p>助手单帧结构观察：{note}</p><label>用户分数 <select class="human"><option value="">未评</option><option>0</option><option>1</option><option>2</option></select></label>
<label>现象 <select class="family"><option value="">未评</option>{options}</select></label><input class="note" placeholder="人工备注；时序需观看视频" /></article>''')
    generated=sum((Path(c['folder'])/'result.json').exists() for c in m['cases'])
    masked=sum((Path(c['folder'])/'mask_result.json').exists() for c in m['cases'])
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>v77 r50 · 真实 DELETE 结构开发排查</title>
<style>body{background:#101722;color:#e6edf7;font:16px/1.6 system-ui;margin:24px auto;max-width:1750px;padding:0 18px}a{color:#75c4ff}article{background:#192332;border:1px solid #34445b;border-radius:12px;padding:16px;margin:20px 0}h1{font-size:26px}h2{font-size:20px}.columns{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}video,img{width:100%;height:auto}video{margin-top:8px;min-height:100px}.pending{padding:30px;color:#a3b4cc;background:#0c121c}input,select,button{background:#0e1725;color:#e6edf7;border:1px solid #58718b;padding:6px;margin:8px}.note{width:50%}svg{max-width:100%;height:auto}code{font-size:13px;overflow-wrap:anywhere}@media(max-width:1100px){.columns{grid-template-columns:repeat(2,1fr)}}.legend{color:#afc1d9}</style>
<h1>r50：先扩大真实 DELETE 排查，再造匹配结构失败的数据</h1>'''
    quota=m.get('eligible_cases',m['case_count'])
    body+=f'<p>20 个 nuScenes train 开发场景 / {m["case_count"]} 个独立单车目标。准入队列 SAM {masked}/{quota}；DELETE {generated}/{quota}；训练0步。官方原始权重 + r21 完整 SAM；seed42，25步，10帧，1024×576。同场景每次恢复原视频再删另一目标。</p>'
    if 'eligible_cases' in m:
        body+=f'<p>原图单帧检查后，{m["eligible_cases"]}例 / {m["eligible_scenes"]}景进入本轮候选队列；6例输入受限保留原图及理由，暂不推理、不计模型失败。20景49例是采样数，不能冒充全部合格。</p>'
    body+='''<p>当前只定位单帧结构：薄膜、多余车身轮廓、后车变形、车身延伸到道路。现有视频模型仍需10帧输入，本轮不改时间层。真实隐藏区没有 GT；原图或 B 证据模糊的案例单列，不能混成确定的结构失败。</p>
<svg viewBox="0 0 1280 150" role="img" aria-label="真实 RGB，经固定 SAM、官方 DriveEditor、原生与写回对照，先定位结构失败，后用真实 Y 造遮挡再测试空间层">
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#9dcaff"/></marker></defs>
<g fill="#20344e" stroke="#5888b1"><rect x="5" y="40" width="165" height="65" rx="8"/><rect x="200" y="40" width="160" height="65" rx="8"/><rect x="390" y="40" width="200" height="65" rx="8"/><rect x="620" y="40" width="180" height="65" rx="8"/><rect x="830" y="40" width="200" height="65" rx="8"/><rect x="1060" y="40" width="210" height="65" rx="8"/></g>
<g stroke="#9dcaff" marker-end="url(#arrow)"><path d="M170 72H197"/><path d="M360 72H387"/><path d="M590 72H617"/><path d="M800 72H827"/><path d="M1030 72H1057" stroke-dasharray="5 5"/></g>
<g fill="#e6edf7" text-anchor="middle" font-size="17"><text x="87" y="68">nuScenes train</text><text x="87" y="91">真实 RGB / 单车</text><text x="280" y="68">固定 SAM2</text><text x="280" y="91">完整实例 mask</text><text x="490" y="68">官方 DriveEditor</text><text x="490" y="91">r46 / 原始权重</text><text x="710" y="68">原生 + 固定写回</text><text x="710" y="91">单帧结构排查</text><text x="930" y="68">真实 Y + 匹配遮挡</text><text x="930" y="91">未开始，失败后造</text><text x="1165" y="68">内部空间层验证</text><text x="1165" y="91">未开始，不改时间层</text></g></svg>
<p class="legend">黄框：删除目标 A；绿框：投影后方候选 B，框交叠是代理，不是精确遮挡标签。蓝色区域（GPU后）：模型洞；黄色轮廓：完整 SAM。真实 B crop 未经生成或锐化；不声称隐藏部分真值。</p>
<p>缓存来源集合受旧数据准备影响，本次不是700场景均匀总体评测；这20景只作开发/后续训练。另留5景本轮不看 RGB、不训练；原模型预训练是否见过它们尚未核实。</p>
<p><a href="manifest.json">统一配置与目标明细</a> · <a href="sampling.json">采样与隔离场景记录</a> · <a href="cpu_check.json">CPU 交付核验</a></p>
<button onclick="exportScores()">导出人工评分 CSV</button><p>用户评分默认空；助手单帧观察不代替用户判分或视频时序判断。</p>'''
    body+=''.join(cards)
    body+='''<script>
const key='v77-r50-real-structure-review';let saved=JSON.parse(localStorage.getItem(key)||'{}');
for(const c of document.querySelectorAll('article')){let v=saved[c.id]||{};for(const f of ['human','family','note']){let e=c.querySelector('.'+f);e.value=v[f]||'';e.addEventListener('change',()=>{saved[c.id]={human:c.querySelector('.human').value,family:c.querySelector('.family').value,note:c.querySelector('.note').value};localStorage.setItem(key,JSON.stringify(saved))})}}
function playCase(button){let videos=button.closest('article').querySelectorAll('video');for(const v of videos){v.currentTime=0;v.play().catch(()=>{})}}
function exportScores(){const q=x=>'"'+String(x).replaceAll('"','""')+'"';let lines=['case_id,scene,human_score,failure_family,note'];for(const c of document.querySelectorAll('article')){let v=saved[c.id]||{};lines.push([c.id,c.dataset.scene,v.human||'',v.family||'',v.note||''].map(q).join(','))}let a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+lines.join('\\n')],{type:'text/csv;charset=utf-8'}));a.download='v77_r50_structure_review.csv';a.click();URL.revokeObjectURL(a.href)}
</script></html>'''
    (root/'index.html').write_text(body,encoding='utf-8')
    for name in ['manifest.json','sampling.json']:
        dump(root/name,read(O/name))
    print('REVIEW',len(cards),generated,flush=True)


if __name__=='__main__':main()

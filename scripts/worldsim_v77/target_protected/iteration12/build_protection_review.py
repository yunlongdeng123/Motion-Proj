"""相同权重的新旧入口对照，明确不是新条件模型结果。"""
from pathlib import Path
import json,sys,subprocess,html
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r26';R=O/'review'
def read(p):return json.loads(p.read_text())
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def encode(folder,dest):
    if not dest.exists():
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-framerate','10','-i',str(folder/'%05d.png'),'-c:v','libx264','-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
    decoded=imageio_ffmpeg.read_frames(str(dest));meta=next(decoded);n=sum(1 for _ in decoded)
    count={'nb_read_frames':n,'width':meta['size'][0],'height':meta['size'][1]}
    assert n==10
    return count

def main():
    state=read(O/'baseline_state.json');complete={(r['case'],r['arm']) for r in state['completed']}
    plan=read(O/'real_input_plan.json')['cases'];R.mkdir(exist_ok=True);(R/'contacts').mkdir(exist_ok=True)
    cases=[];validation=[]
    for c in plan:
        cid=c['eval_id']
        if not all((cid,a) in complete for a in ['base','r7']):continue
        root=O/'baseline'/cid;source=Path(c['folder']);out=R/'assets'/cid;out.mkdir(parents=True,exist_ok=True)
        target_folder=O/'review_frames'/cid/'target';mask_folder=O/'review_frames'/cid/'mask'
        target_folder.mkdir(parents=True,exist_ok=True);mask_folder.mkdir(exist_ok=True)
        for j,i in enumerate(c['frames']):
            im=np.asarray(Image.open(root/'input'/f'{j:05}.png'));sam=np.asarray(Image.open(source/'target_guard'/f'{i:05}.png'))>0;h=np.asarray(Image.open(source/'model_mask'/f'{i:05}.png'))>0
            ann=Image.fromarray(im);draw=ImageDraw.Draw(ann);yy,xx=np.where(sam)
            if len(xx):draw.rectangle((int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())),outline=(255,210,20),width=3)
            draw.text((10,10),cid+' TARGET '+c['instance_token'][:8],fill=(255,220,50));ann.save(target_folder/f'{j:05}.png')
            overlay=im.copy();overlay[h]=np.rint(overlay[h]*.5+np.array([35,145,250])*.5).astype('uint8');Image.fromarray(overlay).save(mask_folder/f'{j:05}.png')
        old=T/'r21/baseline'/cid
        roles=[('target','原视频＋目标框',target_folder),('mask','新入口 H（蓝色）',mask_folder),('old_base','旧入口·原模型',old/'base'),('new_base','新入口·原模型',root/'base'),('old_r7','旧入口·有效 r7',old/'r7'),('new_r7','新入口·有效 r7',root/'r7')]
        assets=[];panels=[];idx=5
        for name,title,folder in roles:
            dest=out/(name+'.mp4')
            info=encode(folder,dest);validation.append({'case':cid,'role':name,**info,'full_decode':True})
            src=f'assets/{cid}/{name}.mp4';assets.append({'role':name,'title':title,'src':src})
            frame=np.asarray(Image.open(folder/f'{idx:05}.png').convert('RGB'));Image.fromarray(frame).save(out/(name+'.jpg'),quality=95);panels.append((title,frame))
        # 固定f5目标区域，所有臂同一crop，不用结果挑区域。
        sam=np.asarray(Image.open(source/'target_guard'/f'{c["frames"][idx]:05}.png'))>0;yy,xx=np.where(sam)
        x0=max(0,int(xx.min())-90);x1=min(1024,int(xx.max())+90);y0=max(0,int(yy.min())-70);y1=min(576,int(yy.max())+70)
        if x1-x0<320:mid=(x0+x1)//2;x0=max(0,mid-160);x1=min(1024,x0+320)
        canvas=Image.new('RGB',(1440,700),(17,22,30));d=ImageDraw.Draw(canvas)
        for k,(title,im) in enumerate(panels):
            row,col=divmod(k,3);d.text((col*480+8,row*350+8),['INPUT + target','NEW H','OLD base','NEW base','OLD r7','NEW r7'][k]+' f5',fill='white')
            tile=Image.fromarray(im[y0:y1,x0:x1]);tile.thumbnail((480,315));canvas.paste(tile,(col*480,row*350+30))
        canvas.save(R/'contacts'/(cid+'.jpg'),quality=95)
        cases.append(c|{'assets':assets,'contact':'contacts/'+cid+'.jpg','fixed_review_frame':idx,'human_verdict':None})
    dump(R/'manifest.json',{'cases':cases,'role':'input_engineering_control','new_model_trained':False,'human_verdict':None})
    dump(R/'media_validation.json',{'cases':len(cases),'videos':len(validation),'decoded_frames':sum(int(r['nb_read_frames']) for r in validation),'items':validation})
    if len(cases)<8:print('PARTIAL_CONTACTS',len(cases));return
    rows=[]
    for c in cases:
        vids=''.join(f'<figure><figcaption>{a["title"]}</figcaption><video controls muted playsinline loop preload="metadata" src="{a["src"]}" poster="assets/{c["eval_id"]}/{a["role"]}.jpg"></video></figure>' for a in c['assets'])
        input_review=read(O/'assistant_input_review.json')
        note=next(v['comment'] for v in input_review['cases'] if v['case']==c['eval_id'])
        note += ' 本轮保留的可见实例像素（10帧累计）：'+str(sum(m['observed_pixels_preserved'] for m in c['metrics']))+'。'
        result_path=O/'assistant_output_review.json'
        if result_path.exists():
            reviews=read(result_path)['cases'];entry=next((v for v in reviews if v['case']==c['eval_id']),None)
            if entry:note+=' 助手固定帧观察：'+entry['comment']
        rows.append(f'<section id="{c["eval_id"]}"><h2>{c["eval_id"]} · {c["scene"]} · {c["camera"]}</h2><p>删除目标：{c["instance_token"]}。{note}</p><button class="play">同步播放 / 暂停</button><button class="restart">回到首帧</button><label>逐帧 <input class="frame" type="range" min="0" max="9" value="0" step="1"><output>0</output></label><div class="grid">{vids}</div><details><summary>固定 f5 同区放大</summary><img src="{c["contact"]}" alt="固定帧六臂对照"></details><p>人工新入口 r7：<select class="score"><option value="">未评</option><option>0</option><option>1</option><option>2</option></select> <input class="note" placeholder="保留车／新增车／残留／时序备注"></p></section>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 r26 可见保护输入对照</title><style>body{background:#101820;color:#e7edf4;font:16px/1.6 system-ui;margin:24px auto;max-width:1600px;padding:0 18px}h1{font-size:28px}h2{font-size:21px}p{max-width:1200px}section{border-top:1px solid #425467;padding:22px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:12px}figure{margin:8px 0}video,img{max-width:100%;width:100%}figcaption{font-weight:600}.diagram{display:flex;flex-wrap:wrap;gap:12px;align-items:center;background:#1a2b3d;padding:18px}.box{padding:8px 16px;border:1px solid #6c93b9;border-radius:6px}button,select,input{font:inherit;background:#23384b;color:#fff;border:1px solid #617d96;padding:6px;margin:4px}.note{width:55%}a{color:#8ac5ff}.warning{color:#ffc473}@media(max-width:850px){.grid{grid-template-columns:1fr 1fr}}</style><h1>v77 r26：保留可见邻车后的真实 DELETE 对照</h1><p class="warning">这是统一输入规则的工程对照，新训练0步。新旧两组权重完全相同：原模型对原模型、有效r7对有效r7。蓝色是生成区域H，黄色框是目标范围。各删除视频为DriveEditor输出局部写回，没有把原生deletion称作factual重建。</p><div class="diagram"><span class="box">真实RGB<br>先擦除完整目标＋边缘保护区</span>→<span class="box">SAM2可见邻车<br>从旧H中保留这些像素</span>→<span class="box">同一原模型 / 同一r7<br>10帧 · 576×1024 · seed42 · 25步</span>→<span class="box">对应alpha写回</span>→<span class="box">旧r21 / 新r26逐帧对照</span></div><p>所有8个既有DEV窗口按同一规则运行，无逐例调参。80帧完整目标边缘保护区的洞外遗漏与alpha遗漏均为0；可见邻车共85,595像素次被保留进输入。这只是输入合同，不证明SAM实例正确或隐藏后车已恢复。A041模糊边界、A042立柱处实例仍有疑点。A022输入没有变化，是零变化控制。</p><p>这里回答“少擦掉已见邻车，是否让旧模型更会补景”。新空间/身份条件尚未完成同预算训练，不能把本页变化算成条件模型收益。每例固定f5观察只是助手粗分类，人工0/1/2留空，时序需看视频。</p><p><a href="../v77-target-protected-r25/index.html">r25：三例输入修复及独立QA</a> · <a href="../v77-target-protected-r22/index.html">r22：合法投影条件覆盖</a> · <a href="../v77-target-protected-r21/index.html">r21：完整目标覆盖对照</a></p><button id="export">导出本页人工评审 JSON</button>'''+''.join(rows)+'''<script>
const storageKey='v77-r26-human-review';let saved={};try{saved=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}
document.querySelectorAll('section').forEach(s=>{const vs=[...s.querySelectorAll('video')],r=s.querySelector('.frame'),score=s.querySelector('.score'),note=s.querySelector('.note');if(saved[s.id]){score.value=saved[s.id].score??'';note.value=saved[s.id].note??''}const save=()=>{saved[s.id]={score:score.value===''?null:Number(score.value),note:note.value};try{localStorage.setItem(storageKey,JSON.stringify(saved))}catch(e){}};score.onchange=save;note.oninput=save;s.querySelector('.play').onclick=()=>{if(vs.some(v=>!v.paused)){vs.forEach(v=>v.pause())}else{const t=vs[0].currentTime;vs.forEach(v=>{v.currentTime=t;v.play().catch(()=>{})})}};s.querySelector('.restart').onclick=()=>{vs.forEach(v=>{v.pause();v.currentTime=0});r.value=0;s.querySelector('output').value=0};r.oninput=()=>{vs.forEach(v=>{v.pause();v.currentTime=Number(r.value)/10+.025});s.querySelector('output').value=r.value};vs[0].ontimeupdate=()=>{if(!vs[0].paused){r.value=Math.min(9,Math.floor(vs[0].currentTime*10));s.querySelector('output').value=r.value;for(const v of vs.slice(1))if(Math.abs(v.currentTime-vs[0].currentTime)>.12)v.currentTime=vs[0].currentTime}};});
document.querySelector('#export').onclick=()=>{const rows=[...document.querySelectorAll('section')].map(s=>({case:s.id,run:'r26_visible_protection',human_score:s.querySelector('.score').value===''?null:Number(s.querySelector('.score').value),note:s.querySelector('.note').value}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));a.download='v77_r26_human_review.json';a.click();URL.revokeObjectURL(a.href)};
</script></html>'''
    if (O/'assistant_output_review.json').exists():
        warning='<p class="warning"><strong>本候选不推广，继续使用r21入口。</strong>助手固定f5发现A048、A034、A061明显退化；A013的保护大巴仍变形。A022两臂全部20帧与旧结果逐像素相同。本页保留完整反例，不把输入QA的2分当生成效果通过；人工分数与全视频判定留给你。</p>'
        page=page.replace('<div class="diagram">',warning+'<div class="diagram">',1)
    (R/'index.html').write_text(page)
    print('REVIEW_READY',len(cases),len(validation))

if __name__=='__main__':main()

"""导出数据准入与固定微调对照；同步视频、原生输出、分数与限制。"""
from pathlib import Path
import argparse,json,subprocess,html,shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw

def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def h(x):return html.escape(str(x))
def encode(files,out):
    if out.exists():return
    files=list(files);assert files
    proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','image2pipe','-vcodec','png','-r','10','-i','-','-an','-c:v','libx264','-threads','2','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    for f in files:proc.stdin.write(f.read_bytes())
    proc.stdin.close();assert proc.wait()==0

STYLE='''body{font:16px system-ui;background:#121820;color:#e5edf8;margin:24px}a{color:#9ccfff}.cols{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.data{grid-template-columns:repeat(3,1fr)}video,img{width:100%;background:#000}article{margin:30px 0;padding:16px;border:1px solid #4b5b70;border-radius:8px}p{line-height:1.6}table{border-collapse:collapse}td,th{border:1px solid #556477;padding:6px}.diagram{display:flex;flex-wrap:wrap;align-items:center;gap:12px;padding:16px;background:#243244}.diagram span{border:1px solid #7696bd;padding:12px}button,select{padding:8px;margin:6px;background:#283b55;color:white;border:1px solid #8194ad}@media(max-width:1000px){.cols,.data{grid-template-columns:repeat(2,1fr)}}'''
SCRIPT=r'''document.querySelectorAll('article').forEach(a=>{let vv=[...a.querySelectorAll('video')];a.querySelectorAll('[data-play]').forEach(b=>b.onclick=()=>{let t=vv[0].currentTime;vv.forEach(v=>{v.currentTime=t;v.play()})});a.querySelectorAll('[data-pause]').forEach(b=>b.onclick=()=>vv.forEach(v=>v.pause()));vv.forEach(v=>v.addEventListener('seeked',()=>{if(v.matches(':focus'))vv.forEach(o=>{if(o!==v&&Math.abs(o.currentTime-v.currentTime)>.12)o.currentTime=v.currentTime})}));let s=a.querySelector('select');if(s){s.value=localStorage.getItem('v77r6-'+a.id)||'';s.onchange=()=>localStorage.setItem('v77r6-'+a.id,s.value)}});document.querySelector('#export')?.addEventListener('click',()=>{let rows=[['eval_id','human_score']];document.querySelectorAll('article select').forEach(s=>rows.push([s.closest('article').id,s.value]));let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.map(r=>r.join(',')).join('\n')],{type:'text/csv'}));a.download='v77_r6_human_review.csv';a.click()});'''
def video(path,label):return f'<div><p>{h(label)}</p><video controls muted playsinline preload="metadata" src="{h(path)}"></video></div>'
def doc(title,content):return f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{h(title)}</title><style>{STYLE}</style><h1>{h(title)}</h1>{content}<script>{SCRIPT}</script></html>'
DIAGRAM='<div class="diagram"><span>真实 nuScenes train Y</span>→<span>几何放置 A + 精确 H<br>五项质检 / AI技术2分</span>→<span>先清零 X，再缩放<br>masked-X 条件 + 真实 Y 监督</span>→<span>原 DriveEditor<br>现有80张量微调，结构不变</span>→<span>固定留出 + 已曝光 DELETE<br>原权重 / 微调权重对照</span></div>'

def data_report(root,out):
    catalog=read(root/'dataset_catalog.json');out.mkdir(exist_ok=True);assets=out/'data_assets';assets.mkdir(exist_ok=True)
    def case_assets(c):
        cid=c['dataset_id'].replace('/','_');folder=Path(c['folder']);dest=assets/cid;dest.mkdir(exist_ok=True)
        for role in ('Y','X','condition_preview'):encode(sorted((folder/role).glob('*.png')),dest/(role+'.mp4'))
        # 一帧原始hole证据，合成外观不是训练条件。
        source=sorted((folder/'model_hole').glob('*.png'))[min(5,c['frame_count']-1)];shutil.copy2(source,dest/'hole.png')
        return cid
    with ThreadPoolExecutor(max_workers=2) as pool:ids=list(pool.map(case_assets,catalog['cases']))
    body=DIAGRAM+f'<p>保留 {len(ids)} 个case，{catalog["summary"]["receiver_scene_count"]} 个receiver scene；48训练 / 4验证。29纯背景 / 23单保护车 / 0密集多车。720帧无损对逐帧重新检查：合成影响在H内，masked-X等于masked-Y。此处2分是技术AI准入，历史人工评分未改；人工 verdict 全部为空。</p><p>4验证case共享一个receiver scene，按receiver与donor整连通分量隔离；不代表4个独立验证场景。AI视觉是既定抽帧，工程是全帧，不能宣称人工逐帧通过。</p><p><a href="index.html">返回权重效果对照</a> · <a href="dataset_catalog.json">完整数据及旧拒绝记录</a></p>'
    for c,cid in zip(catalog['cases'],ids):
        body+=f'<article id="{cid}"><h2>{h(c["dataset_id"])} · {h(c["type"])} · {h(c["split"])}</h2><p>AI技术分 2；receiver {h(c["receiver_scene"])}，donor {h(c["donor_scene"])}；{c["frame_count"]}帧。{h(c["note"])}</p><p>依据：{h(c["quality_basis"])}</p><button data-play>同步播放</button><button data-pause>全部暂停</button><div class="cols data">'+video(f'data_assets/{cid}/Y.mp4','真实Y / 恢复监督')+video(f'data_assets/{cid}/condition_preview.mp4','masked-X 条件示意（灰洞）')+video(f'data_assets/{cid}/X.mp4','synthetic-X 诊断；完整RGB不送条件')+'</div></article>'
    (out/'data_review.html').write_text(doc('v77 r6 · 技术准入数据 52 cases',body));shutil.copy2(root/'dataset_catalog.json',out/'dataset_catalog.json')

def evaluation_report(root,out):
    plan=read(root/'evaluation_plan.json');training=root/'training';ev=root/'evaluation';assets=out/'eval_assets';assets.mkdir(exist_ok=True)
    before=read(training/'validation_base.json');after=read(training/'validation_finetuned.json');state=read(training/'state.json');cfg=read(training/'config.json')
    delta=(after['mean']/before['mean']-1)*100
    body=DIAGRAM+f'<p>本轮实际完成 {state["steps"]} 步。训练48case / 68个十帧窗口，分辨率320×576；原架构、原官方损失不变，仅更新主分支空间self-attention的Q/K/V/out（80张量，{cfg["trainable_parameters"]:,}参数），其余冻结。未做LoRA或增加protected输入通道。使用单卡AdamW固定1e-5学习率；不是官方8卡完整训练配置。</p><p>固定验证去噪loss：{before["mean"]:.6f} → {after["mean"]:.6f}（{delta:+.1f}%）。这是4例×2个固定噪声的latent指标，不能替代视频效果。推理保持576×1024 / 10帧 / seed42 / 25步，原权重与微调使用同输入、同mask、同写回规则。</p><p><a href="data_review.html">52例数据准入 / 来源 / 合同</a> · <a href="summary.json">指标与边界</a> · <a href="assistant_effect_reviews.json">独立单帧粗评</a></p><p>真实A022/A041/A013/A048是已曝光nuScenes val开发失败例，不是final test；删除真值未知。合成留出有真实Y可核对，但4case共享一个receiver scene。每个视频仅1秒首窗，不从单帧粗评判定时序。完整synthetic-X不送条件；真实Y作为扩散目标，含噪Y latent按标准扩散训练进入UNet，干净Y不作条件。</p><button id="export">导出人工分数CSV</button>'
    body='<p><strong>本轮微调未取得稳定收益，保持原模型为默认。</strong>去噪loss下降，生成视频仍有模糊/残留；下方保留全部正反例。四个合成留出例的洞内真实保护车MAE均变差；不按此结果倒改数据分数，也不将单场景结论外推全数据集。</p>'+body
    correction=root/'training_checkpoint_correction.json'
    if correction.exists():
        body='<aside id="r7-encoder-correction"><strong>r7纠正：r6训练缺失106个目标encoder权重，Y由随机冻结encoder监督。推理0missing不能认证训练。旧输出保留，不能归因数据或模块。</strong></aside>'+body
    rows=[]
    reviews=read(root/'assistant_effect_reviews.json') if (root/'assistant_effect_reviews.json').exists() else {'cases':[]}
    qa={c['eval_id']:c for c in reviews['cases']}
    for c in plan['cases']:
        cid=c['eval_id'];folder=ev/cid;dest=assets/cid;dest.mkdir(exist_ok=True)
        if c['kind']!='synthetic':
            marked=folder/'input_box';marked.mkdir(exist_ok=True)
            cores=sorted((Path(c['folder'])/'core').glob('*.png'))
            for i in range(10):
                im=Image.open(folder/'input'/f'{i:05}.png').convert('RGB');d=ImageDraw.Draw(im);mask=np.asarray(Image.open(cores[c['frames'][i]]))>0;yy,xx=np.where(mask)
                if len(xx):
                    box=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())];d.rectangle(box,outline='#ffe529',width=3);d.text((box[0]+4,max(0,box[1]-16)),f'{c["eval_id"]} DELETE actor {c["actor_ordinal_in_scene"]}',fill='#ffe529')
                im.save(marked/f'{i:05}.png')
        roles=['GT','condition','base','finetuned'] if c['kind']=='synthetic' else ['input_box','condition','base','finetuned']
        for role in roles+['base_native','finetuned_native']+(['input'] if c['kind']!='synthetic' else []):encode(sorted((folder/role).glob('*.png')),dest/(role+'.mp4'))
        metrics={arm:read(folder/f'{arm}_metrics.json') for arm in ('base','finetuned')}
        def mean(arm,key):
            vals=[r[key] for r in metrics[arm]['scores'] if r[key] is not None]
            if not vals:return None
            return {v:float(np.mean([z[v] for z in vals])) for v in ('MAE','PSNR')}
        row={'eval_id':cid,'kind':c['kind'],'base_hole':mean('base','hole'),'finetuned_hole':mean('finetuned','hole'),'base_protected_in_hole':mean('base','protected_inside_hole'),'finetuned_protected_in_hole':mean('finetuned','protected_inside_hole')};rows.append(row)
        note=qa.get(cid,{}).get('note','等待独立单帧粗评；人工分数未填。')
        rows[-1].update(input_not_exercised=c.get('input_not_exercised',False),frames=c['frames'])
        if c.get('input_not_exercised'):note='原首窗0/10帧有mask，未实际执行删除。先前此帧AI2只说明画面未变，不算DELETE通过；新有效窗口另列，原始记录保留。'
        labels=['真实Y / 已知恢复目标','模型条件示意','原DriveEditor写回','微调160步写回'] if c['kind']=='synthetic' else [f'原视频 / 黄框 actor {c["actor_ordinal_in_scene"]}','模型条件示意 / 单目标洞','原DriveEditor DELETE','微调160步 DELETE']
        body+=f'<article id="{h(cid)}"><h2>{h(cid)} · {h(c["kind"])}</h2><p>{h(note)}</p><p>人工视频判定：<select><option value="">未评</option><option value="0">0 失败</option><option value="1">1 较差</option><option value="2">2 可接受</option></select></p><button data-play>同步播放</button><button data-pause>全部暂停</button><div class="cols">'+''.join(video(f'eval_assets/{cid}/{role}.mp4',label) for role,label in zip(roles,labels))+'</div>'
        body+=f'<p><a href="eval_assets/{cid}/base_native.mp4">原权重整帧原生输出</a> · <a href="eval_assets/{cid}/finetuned_native.mp4">微调整帧原生输出</a></p>'
        if c['kind']!='synthetic':body+=f'<p>{h(c["scene"])} / {h(c["camera"])} / instance {h(c["instance_token"])}。黄框是冻结实例core的可视范围包络，不是删除矩形。<a href="eval_assets/{cid}/input.mp4">未标框原视频</a></p>'
        if row['base_hole']:
            body+=f'<p>洞内 MAE {row["base_hole"]["MAE"]:.4f} → {row["finetuned_hole"]["MAE"]:.4f}；PSNR {row["base_hole"]["PSNR"]:.2f} → {row["finetuned_hole"]["PSNR"]:.2f} dB。protected指标只测H内真实B，H外固定写回保留不冒充恢复收益。</p>'
        body+='</article>'
    data=read(root/'dataset_catalog.json')['summary'].copy();data['GPU_training_steps_at_catalog_freeze']=data.pop('GPU_training_steps')
    valid=[r for r in rows if r['kind']=='synthetic']
    mean_b=float(np.mean([r['base_protected_in_hole']['MAE'] for r in valid]));mean_f=float(np.mean([r['finetuned_protected_in_hole']['MAE'] for r in valid]))
    summary={'data':data,'training':state,'model_decision':'r6 finetune not promoted; original default retained','validation_loss_base':before['mean'],'validation_loss_finetuned':after['mean'],'validation_loss_change_percent':delta,'protected_MAE_base':mean_b,'protected_MAE_finetuned':mean_f,'protected_MAE_change_percent':(mean_f/mean_b-1)*100,'protected_MAE_worse_cases':sum(r['finetuned_protected_in_hole']['MAE']>r['base_protected_in_hole']['MAE'] for r in valid),'evaluation':rows,'assistant_review':reviews,'human_verdict':None,'limits':['dense_actor training data zero','10 receiver scenes total; heldout4 cases share one receiver scene','160step partial self-attention pilot, not full-model training','train320x576 vs inference576x1024; resolution control not run, not proven causal','real deletion examples exposed development; unknown clean GT','sampled-frame QA cannot certify temporal quality']}
    if correction.exists():summary['r7_training_encoder_correction']=read(correction)
    dump(out/'summary.json',summary);dump(root/'pilot_summary.json',summary);(out/'index.html').write_text(doc('v77 r6 · 原权重 / 微调 DELETE 对照',body));dump(out/'assistant_effect_reviews.json',reviews)

def contacts(root):
    dest=root/'effect_review';dest.mkdir(exist_ok=True)
    for c in read(root/'evaluation_plan.json')['cases']:
        folder=root/'evaluation'/c['eval_id'];roles=['GT','condition','base','finetuned'] if c['kind']=='synthetic' else ['input','condition','base','finetuned']
        if not all((folder/r/'00005.png').exists() for r in roles):continue
        mask=np.asarray(Image.open(folder/'mask/00005.png'))>0;yy,xx=np.where(mask)
        if len(xx):
            box=(max(0,int(xx.min())-60),max(0,int(yy.min())-60),min(1024,int(xx.max())+61),min(576,int(yy.max())+61))
        else:box=(0,0,1024,576)
        canvas=Image.new('RGB',(4*512,2*318),'#161b22');d=ImageDraw.Draw(canvas)
        for row,i in enumerate((5,5)):
            for col,role in enumerate(roles):
                im=Image.open(folder/role/f'{i:05}.png')
                if row:im=im.crop(box)
                im.thumbnail((512,288));canvas.paste(im,(col*512+(512-im.width)//2,row*318+30+(288-im.height)//2));d.text((col*512+4,row*318+5),f'{c["eval_id"]} / {role} / frame {i}'+(' / crop' if row else ''),fill='white')
        canvas.save(dest/(c['eval_id']+'.jpg'),quality=92)

def main(a):
    out=a.root/'delivery';out.mkdir(exist_ok=True)
    if a.mode=='data':data_report(a.root,out)
    elif a.mode=='contacts':contacts(a.root)
    else:evaluation_report(a.root,out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['data','contacts','evaluation'],required=True);a=p.parse_args();main(a)

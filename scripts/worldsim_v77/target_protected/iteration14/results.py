"""GPU后完整视频对照交付；CPU阶段不调用，不填写人工verdict。"""
from common import *
import html, shutil
import numpy as np
from PIL import Image, ImageDraw
from review import ARCHITECTURE
from audit_inputs import main as audit_inputs


def encode(frames,path):
    import subprocess, imageio_ffmpeg
    height,width=frames.shape[1:3]
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{width}x{height}','-r','10','-i','-','-an','-c:v','libx264','-threads','1',
        '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    result=subprocess.run(command,input=frames.tobytes(),capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr.decode())


def main():
    plan=read(O/'manifest.json');state=read(O/'evaluation/state.json')
    assert state['stage']=='complete_pending_review' and len(state['completed'])==60
    training=read(O/'training/state.json');assert training['stage']=='complete'
    audit_inputs()
    token_audit={r['case_id']:r for r in read(O/'reference_token_audit.json')['rows']}
    user_scores={r['case_id']:r for r in read(O/'user_review/human_review.json')['cases']}
    assistant_review={r['case_id']:r for r in read(O/'assistant_visual_review.json')['cases']}
    assert all(r['reviewed'] for r in assistant_review.values())
    information={r['case_id']:r for r in read(O/'input_information_audit.json')['rows']}
    out=O/'review';(out/'gpu_assets').mkdir(exist_ok=True);cards=[];scores=[];video_paths=[]
    labels={'baseline':'官方原权重 + r21 SAM','null_priors':'训练分支 · 全未知先验',
        'RGB_only':'参考RGB + camera/time','geometry_only':'BEV + 2D几何','RGB_and_geometry':'RGB + BEV + 2D几何'}
    for c in [c for c in plan['cases'] if c['split']!='train']:
        cid=c['case_id'];dest=out/'gpu_assets'/cid;dest.mkdir(exist_ok=True)
        original=images(c,'target') if c['kind']=='synthetic' else images(c,'rgb')
        hole=images(c,'hole')>0;alpha=images(c,'alpha').astype('float32')/255 if c['kind']=='real' else hole.astype('float32')
        x=images(c,'rgb');masked=x.copy();masked[hole]=127
        import cv2
        marked=original.copy()
        for i in range(10):
            contours,_=cv2.findContours(hole[i].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(marked[i],contours,-1,(55,165,255),2)
        for name,frames in [('original',marked),('masked',masked)]:
            path=dest/f'{name}.mp4';encode(frames,path);video_paths.append(path)
        metrics={}; fixed_frames=[Image.fromarray(original[5])]
        for arm in plan['arms']:
            folder=O/'evaluation'/cid/arm
            frames=np.stack([np.asarray(Image.open(folder/f'{i:05}.png').convert('RGB')) for i in range(10)])
            raw=np.stack([np.asarray(Image.open(folder/'native'/f'{i:05}.png').convert('RGB')) for i in range(10)])
            assert np.array_equal(frames[~hole],x[~hole])
            for suffix,data in [('',frames),('_native',raw)]:
                path=dest/f'{arm}{suffix}.mp4';encode(data,path);video_paths.append(path)
            metrics[arm]={'hole_MAE':float(np.abs(frames.astype(float)-original).mean(-1)[hole].mean()/255) if c['kind']=='synthetic' else None}
            fixed_frames.append(Image.fromarray(frames[5]))
        yy,xx=np.where(hole[5]);box=(max(0,int(xx.min())-60),max(0,int(yy.min())-45),min(1024,int(xx.max())+61),min(576,int(yy.max())+46))
        contact=Image.new('RGB',(1536,646),'#0f1824');draw=ImageDraw.Draw(contact)
        for j,(name,im) in enumerate(zip(['original/GT']+plan['arms'],fixed_frames)):
            left=j%3*512;top=j//3*323;draw.text((left+10,top+6),cid+' f05 '+name,fill='white')
            crop=im.crop(box);crop.thumbnail((512,288));contact.paste(crop,(left+(512-crop.width)//2,top+30+(288-crop.height)//2))
        contact.save(dest/'f05_crop.jpg',quality=94)
        def video(name,caption):
            return f'<figure><figcaption>{caption}</figcaption><video controls muted playsinline preload="metadata" src="gpu_assets/{cid}/{name}.mp4"></video></figure>'
        body=f'<h2>{cid} · {html.escape(c["scene"])}</h2><p>{html.escape(c["split"])} · <a href="index.html#{cid}">本例CPU输入和参考来源</a></p>'
        previous=user_scores.get(cid)
        if previous and previous.get('human_score') is not None:body+=f'<p>既有 r46 原模型人工分：{previous["human_score"]}；本轮新结果尚未人工评审。</p>'
        slots=token_audit[cid]['slots']
        actor_tokens=sum(r['valid_attention_tokens'] for r in slots if r['role']=='protected_actor_appearance')
        context_tokens=sum(r['valid_attention_tokens'] for r in slots if r['role']=='background_context')
        body+=f'<p>实际参考有效token：保护车 {actor_tokens}，场景上下文 {context_tokens}。这是网络可接收范围，不代表完整车身/清楚身份；OCC仍是GT包络proxy。源mask通过不等于appearance充分。</p>'
        fractions=information[cid]['mean_H_fraction']
        body+=f'<p>10帧洞内控制格平均：保留车包络 {fractions["O_proxy_H_fraction"]:.1%}，实测LiDAR背景 {fractions["N_measured_H_fraction"]:.1%}，未知 {fractions["U_H_fraction"]:.1%}。不是精确silhouette或已认证可见面积。</p>'
        body+='<div class="row">'+video('original','原RGB / 合成真实GT')+video('baseline',labels['baseline'])+video('RGB_and_geometry',labels['RGB_and_geometry'])+'</div>'
        body+='<p><button onclick="play(this)">同步播放</button> <button onclick="pause(this)">暂停</button> 帧 <input type="range" min="0" max="9" value="0" oninput="seek(this)"> <span>f00</span></p>'
        body+='<details><summary>遮后输入、条件消融及原生输出</summary>'+video('masked','独立mask擦除后的查询RGB')+'<div class="row">'+''.join(video(a,labels[a]) for a in ['null_priors','RGB_only','geometry_only'])+'</div><div class="row">'+''.join(video(a+'_native',labels[a]+' · 未硬mask写回') for a in plan['arms'])+'</div></details>'
        note=assistant_review[cid]
        body+='<p><strong>助手固定f05粗查（不是人工分、不是时序判断）：</strong>'+html.escape(note['observation'])+'</p><details><summary>f05 六组洞区放大对照</summary><img loading="lazy" style="width:100%" src="gpu_assets/'+cid+'/f05_crop.jpg"></details>'
        body+='<p>本轮人工分 <select data-case="'+cid+'" onchange="save(this)"><option value="">未评</option><option>0</option><option>1</option><option>2</option></select> <input class="note" data-case="'+cid+'" placeholder="删除/幻觉车/后车/邻车/时序备注" onchange="save(this)"></p>'
        if c['kind']=='synthetic':body+='<p>真实GT洞内MAE（0–1；10帧洞内像素合并）：'+ '；'.join(labels[a]+f' {metrics[a]["hole_MAE"]:.5f}' for a in plan['arms'])+'</p>'
        cards.append('<article id="'+cid+'">'+body+'</article>');scores.append({'case_id':cid,'metrics':metrics,'assistant_f05_review':note,'human_verdict':None,'temporal_verdict':None})
    # 实际完整解码，视频数量不冒充独立样本数。
    import cv2
    decoded=0
    for path in video_paths:
        capture=cv2.VideoCapture(str(path));count=0
        while True:
            ok,frame=capture.read()
            if not ok:break
            count+=1
        capture.release();assert count==10,(path,count);decoded+=count
    architecture=ARCHITECTURE.replace('r47：CPU合同与输入准备；完整官方模型接入、训练和生成仍待 GPU','r47：主干冻结；多先验分支320步；固定DEV条件消融')
    synthetic=[r for r in scores if r['metrics']['baseline']['hole_MAE'] is not None]
    macro={a:float(np.mean([r['metrics'][a]['hole_MAE'] for r in synthetic])) for a in plan['arms']}
    summary='<p><strong>当前结论：</strong>4个合成DEV的洞内误差下降；真实DEV固定f05中，A034和A061_w08的保护车轮廓/前脸更完整，但A041、A048等仍有涂抹，未建立跨例稳定收益。本轮保留全部结果，默认继续r46原模型+完整SAM，不按case择优换臂。整段质量待人工评审。</p>'
    summary+='<table><caption>4个合成DEV · 10帧洞内像素合并后逐case等权平均MAE（0–1）</caption><tr>'+''.join('<th>'+labels[a]+'</th>' for a in plan['arms'])+'</tr><tr>'+''.join(f'<td>{macro[a]:.6f}</td>' for a in plan['arms'])+'</tr></table>'
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>v77 r47 · 真实DELETE完整视频对照</title><style>body{font:16px/1.6 system-ui;background:#0d1520;color:#dde7f2;padding:24px}article{border:1px solid #40556d;padding:18px;margin:24px 0}a{color:#9acaff}.row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%}figure{margin:0}.note{width:60%}svg{width:100%;max-width:1120px}select,input,button{font:inherit;margin:3px}@media(max-width:900px){.row{grid-template-columns:1fr}}</style><h1>v77 r47 · 新条件接口完整视频对照</h1><p>全部为已曝光DEV。主栏是原RGB、原模型、RGB+BEV；展开查看同一训练分支的条件消融。原生deletion不是factual重建。人工0/1/2及时间评价只由用户填写，单帧不承担时序结论。</p>'''+architecture+'''<p>实际官方模型零初始化等价、RGB/BEV梯度和主干冻结已通过。新支路固定320步，原始权重及查询r21 SAM保持基线；未根据真实结果选择checkpoint。</p><p>本轮重点边界：appearance anchor可能不充分、部分reference有效信息少、OCC是包络proxy、洞内target absence与protected preservation尚未被硬约束。洞外原像素保持，不代表洞内保护车恢复成功。</p><p><a href="index.html">输入准备页（CPU阶段快照）</a> · <a href="reference_token_audit.json">有效参考token检查</a> · <button onclick="exportReview()">导出人工评分</button></p>'''+''.join(cards)+'''
<script>const key='v77-r47-human-review';let scores={};try{scores=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function videos(el){return [...el.closest('article').querySelectorAll('video')]}
function play(el){let vv=videos(el).filter(v=>v.getClientRects().length>0),t=vv[0].currentTime;vv.forEach(v=>{v.currentTime=t;v.play().catch(()=>{})})}
function pause(el){videos(el).forEach(v=>v.pause())}
function seek(el){videos(el).forEach(v=>{v.pause();v.currentTime=Number(el.value)/10+.001});el.nextElementSibling.textContent='f'+el.value.padStart(2,'0')}
function save(el){let a=el.closest('article'),cid=el.dataset.case;scores[cid]={score:a.querySelector('select').value||null,note:a.querySelector('.note').value};localStorage.setItem(key,JSON.stringify(scores))}
document.querySelectorAll('select').forEach(el=>{let r=scores[el.dataset.case];if(r){el.value=r.score||'';el.closest('article').querySelector('.note').value=r.note||''}})
function exportReview(){let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({run:'r47',reviewer:'human',cases:scores},null,2)],{type:'application/json'}));a.download='v77-r47-human-review.json';a.click();URL.revokeObjectURL(a.href)}
</script>'''
    for name in ['reference_token_audit.json','input_information_audit.json','assistant_visual_review.json']:
        shutil.copy2(O/name,out/name)
    page=page.replace(architecture,architecture+summary,1)
    (out/'results.html').write_text(page);dump(out/'GPU_review_summary.json',{'cases':scores,'synthetic_case_macro_hole_MAE':macro,'videos':len(video_paths),'decoded_frames':decoded,'human_verdict':None})


if __name__=='__main__':main()

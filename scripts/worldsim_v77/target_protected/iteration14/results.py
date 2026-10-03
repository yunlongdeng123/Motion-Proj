"""GPU后完整视频对照交付；CPU阶段不调用，不填写人工verdict。"""
from common import *
import html, shutil
import numpy as np
from PIL import Image


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
        metrics={}
        for arm in plan['arms']:
            folder=O/'evaluation'/cid/arm
            frames=np.stack([np.asarray(Image.open(folder/f'{i:05}.png').convert('RGB')) for i in range(10)])
            raw=np.stack([np.asarray(Image.open(folder/'native'/f'{i:05}.png').convert('RGB')) for i in range(10)])
            assert np.array_equal(frames[~hole],x[~hole])
            for suffix,data in [('',frames),('_native',raw)]:
                path=dest/f'{arm}{suffix}.mp4';encode(data,path);video_paths.append(path)
            metrics[arm]={'hole_MAE':float(np.abs(frames.astype(float)-original).mean(-1)[hole].mean()/255) if c['kind']=='synthetic' else None}
        def video(name,caption):
            return f'<figure><figcaption>{caption}</figcaption><video controls muted playsinline preload="metadata" src="gpu_assets/{cid}/{name}.mp4"></video></figure>'
        body=f'<h2>{cid} · {html.escape(c["scene"])}</h2><p>{html.escape(c["split"])} · <a href="index.html#{cid}">本例CPU输入和参考来源</a></p>'
        body+='<div class="row">'+video('original','原RGB / 合成真实GT')+video('baseline',labels['baseline'])+video('RGB_and_geometry',labels['RGB_and_geometry'])+'</div>'
        body+='<p><button onclick="play(this)">同步播放</button> <button onclick="pause(this)">暂停</button> 帧 <input type="range" min="0" max="9" value="0" oninput="seek(this)"> <span>f00</span></p>'
        body+='<details><summary>遮后输入、条件消融及原生输出</summary>'+video('masked','独立mask擦除后的查询RGB')+'<div class="row">'+''.join(video(a,labels[a]) for a in ['null_priors','RGB_only','geometry_only'])+'</div><div class="row">'+''.join(video(a+'_native',labels[a]+' · 未硬mask写回') for a in plan['arms'])+'</div></details>'
        body+='<p>本轮人工分 <select data-case="'+cid+'" onchange="save(this)"><option value="">未评</option><option>0</option><option>1</option><option>2</option></select> <input class="note" data-case="'+cid+'" placeholder="删除/幻觉车/后车/邻车/时序备注" onchange="save(this)"></p>'
        cards.append('<article>'+body+'</article>');scores.append({'case_id':cid,'metrics':metrics,'human_verdict':None,'temporal_verdict':None})
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
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>v77 r47 · 真实DELETE完整视频对照</title><style>body{font:16px/1.6 system-ui;background:#0d1520;color:#dde7f2;padding:24px}article{border:1px solid #40556d;padding:18px;margin:24px 0}a{color:#9acaff}.row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%}figure{margin:0}.note{width:60%}select,input,button{font:inherit;margin:3px}@media(max-width:900px){.row{grid-template-columns:1fr}}</style><h1>v77 r47 · 新条件接口完整视频对照</h1><p>全部为已曝光DEV。主栏是原RGB、原模型、RGB+BEV；展开查看同一训练分支的条件消融。原生deletion不是factual重建。人工0/1/2及时间评价只由用户填写，单帧不承担时序结论。</p><p><a href="index.html">CPU组件图和先验准备</a> · <button onclick="exportReview()">导出人工评分</button></p>'''+''.join(cards)+'''
<script>const key='v77-r47-human-review';let scores={};try{scores=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function videos(el){return [...el.closest('article').querySelectorAll('video')]}
function play(el){let vv=videos(el),t=vv[0].currentTime;vv.forEach(v=>{v.currentTime=t;v.play().catch(()=>{})})}
function pause(el){videos(el).forEach(v=>v.pause())}
function seek(el){videos(el).forEach(v=>{v.pause();v.currentTime=Number(el.value)/10+.001});el.nextElementSibling.textContent='f'+el.value.padStart(2,'0')}
function save(el){let a=el.closest('article'),cid=el.dataset.case;scores[cid]={score:a.querySelector('select').value||null,note:a.querySelector('.note').value};localStorage.setItem(key,JSON.stringify(scores))}
document.querySelectorAll('select').forEach(el=>{let r=scores[el.dataset.case];if(r){el.value=r.score||'';el.closest('article').querySelector('.note').value=r.note||''}})
function exportReview(){let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({run:'r47',reviewer:'human',cases:scores},null,2)],{type:'application/json'}));a.download='v77-r47-human-review.json';a.click();URL.revokeObjectURL(a.href)}
</script>'''
    (out/'results.html').write_text(page);dump(out/'GPU_review_summary.json',{'cases':scores,'videos':len(video_paths),'decoded_frames':decoded,'human_verdict':None})


if __name__=='__main__':main()

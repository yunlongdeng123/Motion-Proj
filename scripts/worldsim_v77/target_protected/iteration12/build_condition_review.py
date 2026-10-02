"""输入证据与条件可用性审核页；没有冒充已训练的新模型输出。"""
from pathlib import Path
import json,subprocess,sys
import numpy as np
from PIL import Image
import imageio_ffmpeg
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r22';R=O/'review'
def read(p):return json.loads(p.read_text())
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main():
    R.mkdir(exist_ok=True);quality=read(O/'condition_quality.json')['cases'];validation=[];sections=[]
    for q in quality:
        cid=q['case_id'];s=O/'state'/cid;out=R/'assets'/cid;out.mkdir(parents=True,exist_ok=True)
        gtpaths=sorted((T/'r16/synthetic'/cid/'Y').glob('*.png'));roles=[('GT','真实Y：仅用于评价'),('input','遮后输入'),('visible_sam','遮后SAM：可见保护车'),('state','投影条件：O绿／N蓝／U灰'),('projected_rgb','合法观测颜色：未知置灰')]
        for i,gp in enumerate(gtpaths):
            Image.open(gp).convert('RGB').save(out/f'{i:05}_GT.jpg',quality=94)
            for role,_ in roles[1:]:
                import shutil
                shutil.copy2(s/f'{i:05}_{role}.jpg',out/f'{i:05}_{role}.jpg')
        figures=[]
        for role,title in roles:
            video=out/(role+'.mp4');subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-framerate','10','-i',str(out/f'%05d_{role}.jpg'),'-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
            reader=imageio_ffmpeg.read_frames(str(video));meta=next(reader);count=sum(1 for _ in reader);assert count==30
            validation.append({'case_id':cid,'role':role,'frames':count,'size':meta['size'],'actual_full_decode':True})
            figures.append(f'<figure><figcaption>{title}</figcaption><video muted controls loop playsinline preload="metadata" src="assets/{cid}/{role}.mp4" poster="assets/{cid}/00015_{role}.jpg"></video></figure>')
        m=q['summary'];state=read(s/'result.json');label='两辆车的可见部分可用；洞中背景证据不足' if cid=='L007' else '局部身份证据可用；整洞条件仍不足'
        sections.append(f'<section><h2>{cid} · {q["scene"]} · {label}</h2><p>投影到洞内的O中，{m["O_identity_precision"]:.1%}与同身份的完整视频SAM标签相符；覆盖被遮保护车的{m["B_reveal_coverage"]:.1%}。N仅覆盖全洞{m["known_background_H_fraction"]:.2%}。L007没有被遮的保护车，这两项车辆比例不适用；这些是条件指标，不是DELETE效果。背景窗口内缺失LiDAR文件{state["missing_in_window_lidar_files"]}个，未用窗口外扫描替代。</p><button class="play">同步播放 / 暂停</button><label>逐帧<input type="range" min="0" max="29" step="1" value="0"><output>0</output></label><div class="grid">'+''.join(figures)+'</div></section>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 r22 合法actor-state条件</title><style>body{background:#101820;color:#e9f0f7;font:16px/1.6 system-ui;max-width:1750px;margin:24px auto;padding:0 18px}h1{font-size:28px}h2{font-size:21px}section{padding:24px 0;border-top:1px solid #46576a}.grid{display:grid;grid-template-columns:repeat(3,minmax(250px,1fr));gap:14px}figure{margin:5px}video{width:100%}button,input{font:inherit;margin:6px;background:#20384c;color:white;border:1px solid #7395b6;padding:7px}.diagram{display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:18px;background:#1a2c3d}.box{border:1px solid #719abf;border-radius:6px;padding:10px}.caution{color:#ffca82}a{color:#8bc7ff}</style><h1>r22：模型能拿到哪些合法的时序证据</h1><p class="caution">本页是条件可用性检查，尚未训练Adapter、未启用surfel。真实Y只在评价与本页显示；构建条件的程序只读取遮后RGB、同窗SAM与明确标注的GT几何辅助。</p><div class="diagram"><span class="box">固定30帧遮后窗口<br>2.9秒真实曝光</span>→<span class="box">重跑SAM2可见实例</span>→<span class="box">actor局部框表面代理<br>同窗可见LiDAR背景</span>→<span class="box">删A后按深度投影<br>O／N／U／Q＋颜色</span>→<span class="box">检查精度、覆盖与身份</span></div><p>O：观测支持的保留车辆；N：正证据背景；U：未知或冲突。没有点不会被当作道路。三例重新分割120帧，独立gpt-6-sol xhigh复核：L001与L009 uncertain，L007 pass仅限可见部分；均未证明整洞补全能力。当前主要缺口是证据覆盖，不能据此直接扩量训练。完整视频SAM只是质量标签，也不是像素级真值。</p><p>下一步优先让数据真的包含“被遮部分在别帧显露”，并检查背景正证据覆盖；不使用隐藏Y补造条件。<a href="../v77-target-protected-r21/index.html">另看 r21 入口修复的真实DELETE对照</a>。</p>'''+''.join(sections)+'''<script>document.querySelectorAll('section').forEach(s=>{const vs=[...s.querySelectorAll('video')],r=s.querySelector('input');s.querySelector('.play').onclick=()=>{if(vs.some(v=>!v.paused))vs.forEach(v=>v.pause());else{const t=vs[0].currentTime;vs.forEach(v=>{v.currentTime=t;v.play().catch(()=>{})})}};r.oninput=()=>{vs.forEach(v=>{v.pause();v.currentTime=+r.value/10+.025});s.querySelector('output').value=r.value}})</script></html>'''
    (R/'index.html').write_text(page);dump(R/'media_validation.json',{'cases':3,'videos':15,'decoded_frames':450,'items':validation});dump(R/'quality_metrics.json',{'cases':quality})
    print('CONDITION_REVIEW_READY',15,450)
if __name__=='__main__':main()

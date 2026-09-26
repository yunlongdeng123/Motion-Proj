"""只在DriveEditor两scene实际完成后导出三栏视频；不挪用旧结果。"""
import html,json,pathlib,shutil,subprocess
import cv2,numpy as np
from PIL import Image
from v77_driveeditor_compare import ROOT,SCENES,dump

OUT=ROOT/'review'
if OUT.exists():raise RuntimeError('不覆盖既有review，请检查运行状态')
for scene in SCENES:
    assert json.loads((ROOT/scene/'C/result.json').read_text())['status']=='complete'
    for folder in ['rgb','C/native','C/composite']:
        assert len(list((ROOT/scene/folder).glob('*.png')))==10
OUT.mkdir()
ffmpeg=shutil.which('ffmpeg')
if not ffmpeg:
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
validation=[];sections=[]
for scene,(actor,cam,start) in SCENES.items():
    dst=OUT/scene;dst.mkdir()
    for slug,folder in [('original','rgb'),('native','C/native'),('composite','C/composite')]:
        inp=ROOT/scene/folder;video=dst/f'{slug}.mp4'
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-framerate','10','-i',str(inp/'%05d.png'),'-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
        cap=cv2.VideoCapture(str(video));count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(576,1024,3);count+=1
        fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert count==10 and fps==10
        validation.append({'scene':scene,'video':video.name,'decoded_frames':count,'fps':fps,'size':[1024,576]})
        shutil.copy2(inp/'00005.png',dst/f'{slug}.png')
    im=np.array(Image.open(ROOT/scene/'rgb/00005.png').convert('RGB'))
    sam=np.array(Image.open(ROOT/scene/'target_sam/00005.png'))>0
    mask=np.array(Image.open(ROOT/scene/'mask_b/00005.png'))>0
    overlay=im.copy();overlay[mask]=(.65*im[mask]+.35*np.array([255,150,35])).astype('uint8')
    contours,_=cv2.findContours(sam.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay,contours,-1,(0,240,230),2)
    Image.fromarray(overlay).save(dst/'target_mask.png')
    (dst/'frames').mkdir()
    for kind in ['native','composite']:
        shutil.copytree(ROOT/scene/'C'/kind,dst/'frames'/kind)
    shutil.copy2(ROOT/scene/'C/result.json',dst/'result.json')
    label='棕色旅行车／掀背车' if scene=='scene_0230' else '银白色SUV（前方有围栏）'
    cards=''
    for slug,title,caption in [('original','① 原视频','模型实际接收的连续10帧；没有覆盖GLB，也没有预先替换背景。'),('native','② DriveEditor 原生输出','保留模型生成的整帧。请同时检查目标、邻车、围栏与mask外区域。'),('composite','③ 仅替换删除区域','mask内用同一份DriveEditor输出，mask外逐像素保留原RGB；接缝和mask内错误均保留。')]:
        cards+=f'<article><h3>{title}</h3><video controls loop muted playsinline preload="metadata" poster="{scene}/{slug}.png" src="{scene}/{slug}.mp4"></video><p>{caption}</p><a href="{scene}/{slug}.mp4" download>下载视频</a></article>'
    sections.append(f'''<section id="{scene}"><h2>{scene} · actor {actor}</h2><p>目标：{label}。CAM{cam}，源帧 {start}–{start+9}（{start/10:.1f}–{(start+9)/10:.1f} 秒）。三段视频严格同一输入窗，10 Hz，共1秒。</p><div class="target"><img src="{scene}/target_mask.png" alt="青色目标轮廓与橙色删除范围"><div><h3>先确认删的是哪辆车</h3><p>青色线是目标SAM2轮廓，橙色是实际送入DriveEditor的删除mask。图中是源帧{start+5}。</p><p>mask来自官方扩大框函数的固定输出，会遮到比车身更大的范围；这不是精准轮廓删除。其他车辆被改变也应记录为该输入下的副作用。</p></div></div><div class="controls"><button onclick="playGroup(this)">同步从头播放</button><button onclick="pauseGroup(this)">暂停全部</button><label>播放速度 <select onchange="speedGroup(this)"><option value="1">1×</option><option value="0.25">0.25×</option></select></label><label>逐帧 <input type="range" min="0" max="9" step="1" value="0" oninput="seekGroup(this)"></label></div><div class="videos">{cards}</div><p>查看重点：原车是否消失；路面、围栏有没有糊成团；邻车是否被删除或换形；帧间结构是否跳动。清晰但凭空生成的道路仍不是实测背景真值。</p><p><a href="{scene}/native.png">原生中间帧 PNG</a> · <a href="{scene}/composite.png">合成中间帧 PNG</a> · <a href="{scene}/result.json">实际运行记录</a></p></section>''')

page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 · DriveEditor 删除补景</title><style>
body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#17283b;background:#f2f5f8;margin:0}main{max-width:1580px;margin:32px auto;padding:0 24px}h1{font-size:32px;margin-bottom:8px}h2{font-size:24px}h3{font-size:18px}p{margin:10px 0}section,.intro{background:white;border:1px solid #dce4eb;border-radius:14px;padding:24px;margin:24px 0}.badge{display:inline-block;padding:4px 12px;background:#e6effa;border-radius:20px;font-size:14px}.videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.videos article{min-width:0}video{display:block;width:100%;background:#10151b}.target{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:center}.target img{width:100%;max-width:640px}.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:18px 0}button,select{padding:8px 12px;border:1px solid #bbcbd9;border-radius:6px;background:#f3f7fb}a{color:#175bac}.notice{border-left:4px solid #d78b1e;padding-left:16px}svg{width:100%;max-height:145px}footer{font-size:13px;color:#566777}@media(max-width:950px){.videos,.target{grid-template-columns:1fr}main{padding:0 12px}section{padding:18px}}</style><main><span class="badge">官方训练后权重 · 零训练 · 两个开发场景</span><h1>DriveEditor 删除补景，直接看视频</h1><p>本页只展示这次实际生成的背景删除结果。原位factual与GLB不在本轮重做；先前的资产形状反馈不受这里的补景结果替代。</p><div class="intro"><svg viewBox="0 0 1200 120" role="img" aria-label="RGB和删除mask输入DriveEditor，输出原生和局部合成视频"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="#386992"/></marker></defs><g fill="#edf4fc" stroke="#a6bdd4"><rect x="10" y="25" width="230" height="64" rx="9"/><rect x="320" y="25" width="250" height="64" rx="9"/><rect x="650" y="25" width="220" height="64" rx="9"/><rect x="950" y="25" width="235" height="64" rx="9"/></g><g fill="#17283b" font-size="19" text-anchor="middle"><text x="125" y="53">连续RGB + 删除mask</text><text x="125" y="78" font-size="14">同一个目标 · 各10帧</text><text x="445" y="53">DriveEditor</text><text x="445" y="78" font-size="14">SVD + SV3D / trained checkpoint</text><text x="760" y="53">原生整帧输出</text><text x="760" y="78" font-size="14">保留所有改动</text><text x="1067" y="53">mask内合成</text><text x="1067" y="78" font-size="14">mask外复制原RGB</text></g><g stroke="#386992" stroke-width="2" marker-end="url(#arrow)"><path d="M240 57H315"/><path d="M570 57H645"/><path d="M870 57H945"/></g></svg><p class="notice">本轮是10帧原生短窗测试，还没有长视频、跨相机或Ω重建验收。隐藏背景没有真实GT，人工结论保持待评审；不能仅凭画面更清晰就认定几何正确。</p><p>25步 · seed 42 · 1024×576 · 单RTX3090顺序CFG · 每次解码1帧。没有输入目标外观参考，也没有执行MOVE。</p></div>'''+''.join(sections)+'''<footer>WS-V77-DRIVEEDITOR-COMPARE-20260927/r1 · <a href="validation.json">视频解码检查</a> · <a href="https://github.com/yvanliang/DriveEditor">DriveEditor 官方代码</a></footer></main><script>
function group(el){return [...el.closest('section').querySelectorAll('video')]}
function playGroup(el){group(el).forEach(v=>{v.currentTime=0;v.play()})}
function pauseGroup(el){group(el).forEach(v=>v.pause())}
function speedGroup(el){group(el).forEach(v=>v.playbackRate=Number(el.value))}
function seekGroup(el){group(el).forEach(v=>{v.pause();v.currentTime=Number(el.value)/10})}
</script></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8')
dump(OUT/'validation.json',{'videos':validation,'human_verdict':None,'browser_interaction_tested':False})
dump(ROOT/'review_export.json',{'state':'complete','review_root':str(OUT),'videos':6,'frames_per_video':10,'human_verdict':None})
print('EXPORTED',OUT,flush=True)

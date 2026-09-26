"""将R3可见性证据与明确标注来源的既有视频放到同一审核页。"""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont

p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path.cwd());args=p.parse_args();root=args.workspace
src=root/'work/v77_r3';out=root/'outputs/v77-pipeline-r3';out.mkdir(parents=True,exist_ok=True)
summary=json.loads((src/'summary.json').read_text(encoding='utf-8'));timing=json.loads((root/'outputs/v77-actor-audit/review_data.json').read_text(encoding='utf-8'))
fontpath='C:/Windows/Fonts/msyh.ttc';font=ImageFont.truetype(fontpath,24);small=ImageFont.truetype(fontpath,19)
sections=[]
for r in summary['scenes']:
 s=r['scene'];hard=s=='scene_0230';sd=out/s;sd.mkdir(exist_ok=True);z=np.load(src/f'{s}_visibility.npz');xy=z['query_local'][:,:2];core=z['core'];size=z['gt_size'];c=r['core']
 canv=Image.new('RGB',(1536,680),'#f1f6f8');d=ImageDraw.Draw(canv)
 panels=[('原 GT 框：视线通过',z['raw_all_box_clear_views']>0,f"核心 {c['raw_box_ever_all_boxes_clear']} / {c['queries']}"),('下延到地面：视线通过',z['all_box_clear_views']>0,f"核心 {c['ever_all_boxes_clear']} / {c['queries']}"),('地面 LiDAR 距离 ≤ 15cm',z['lidar_ground_support'],f"核心 {c['lidar_ground_support_within_15cm']} / {c['queries']}")]
 for k,(title,positive,count) in enumerate(panels):
  ox=k*512;scale=min(435/np.ptp(xy[:,0]),445/np.ptp(xy[:,1]));cx=ox+256;cy=330
  def screen(x,y):return cx+x*scale,cy-y*scale
  d.text((ox+20,20),title,font=font,fill='#15394b');d.text((ox+20,60),count,font=small,fill='#39556a')
  side=.2*scale*.86
  for (x,y),ok in zip(xy,positive):
   u,v=screen(x,y);d.rectangle((u-side/2,v-side/2,u+side/2,v+side/2),fill='#168b79' if ok else '#d3dce3')
  x0,y0=screen(-size[0]/2+.2,size[1]/2-.2);x1,y1=screen(size[0]/2-.2,-size[1]/2+.2);d.rectangle((x0,y0,x1,y1),outline='#c04774',width=3)
  d.text((ox+20,568),'粉框：原车底核心（向内缩20cm）',font=small,fill='#b1426b');d.text((ox+20,600),'绿色：通过本栏条件；灰色：无支持',font=small,fill='#39556a');d.text((ox+20,632),'横轴沿车长；纵轴沿车宽（米制）',font=small,fill='#39556a')
 canv.save(sd/'ground_evidence.png')
 for suffix in ['box_context.jpg','box_control.jpg','visibility.json','view_counts.json']:
  shutil.copy2(src/f'{s}_{suffix}',sd/suffix)
 # These are reviewed historical renders, explicitly labelled as such.
 for key in ['raw','corrected','delete']:
  for ext in ['.mp4','_poster.jpg']:shutil.copy2(root/f'outputs/v77-actor-audit/{s}/{key}{ext}',sd/f'{key}{ext}')
 shutil.copy2(root/f'outputs/v77-actor-audit/{s}/target.jpg',sd/'target.jpg')
 notes=('棕色旅行车。原 GT 框底面世界高度从 −0.0045m 变化到 0.5455m；该车全轨迹中心净位移仅约0.65m，不能把框高度变化当成车辆腾空。原 GT 框检查把核心168点全判为可见；贴地包络后全部被挡。' if hard else '银白 SUV。6相机搜索后仍有9个核心点通过贴地框视线，但都缺15cm内的地面LiDAR支持；原CAM3为0。它们不能作为已观测地面复制回去。')
 cards=''.join(f'<article><h4>{label}</h4><p>{note}</p><video controls playsinline preload="metadata" src="{s}/{key}.mp4" poster="{s}/{key}_poster.jpg"></video></article>' for key,label,note in [('raw','① 原视频','青色目标轮廓；右侧是对应车辆放大。'),('corrected','② 原位 factual · 已有放置修正版','同一GLB；朝向/相机修正。仍有光照与遮挡限制。'),('delete','③ DELETE · 已有原补景','独立去车背景；未执行MOVE，没有GLB。')])
 sections.append(f'''<section id="{s}" data-scene="{s}"><h2>{s} · actor {r['actor_id']} · {'棕色旅行车' if hard else '银白 SUV'}</h2>
 <p>目标未更换。<a href="../v77-actor-audit/index.html#{s}">目标、车头、尺度及旧指令审计</a> · <a href="../v77-explicit-poc/{s}/actor.glb">原始 GLB</a></p><img src="{s}/target.jpg" alt="明确本场景目标车辆与源图放大">
 <h3>原视频 / 原位 / 删除：先看同一目标</h3><p>下面沿用<strong>放置审计轮已有的10时刻视频</strong>，用于定位目标和复核，不是R3新渲染。{'2fps，源f000–045，CAM2→CAM4→CAM5' if hard else '1fps，源f000–090，全部CAM3'}。R3新增的是下方的背景证据检查。<a href="../v77-pipeline-r2/index.html#{s}">另看R2完整10Hz原视频 / 196帧输入的DELETE对照</a>。</p>
 <div class="controls"><button class="play">三栏同步播放</button><button class="pause">暂停</button><input class="scrub" type="range" min="0" max="9" value="0" aria-label="抽样时间点"><output></output></div><div class="videos">{cards}</div>
 <h3>新证据：地面点“在画面内”不等于没有遮挡</h3><p>{notes}</p><p>全序列196帧中，仅使用目标GT位姿已知的{r['target_track_frames']}帧（f{r['frame_range'][0]:03}–{r['frame_range'][1]:03}），搜索6相机。剩余{r['omitted_missing_target_pose_frames']}帧的目标状态未知，未将轨迹结束当作目标离开。</p>
 <img src="{s}/ground_evidence.png" alt="原框视线、贴地包络视线、地面LiDAR支持三项对比">
 <table><tr><th>检查对象：原车底核心</th><th>数量</th><th>应如何理解</th></tr><tr><td>进入过任一相机视野</td><td>{c['ever_in_fov']} / {c['queries']}</td><td>只说明投影在图内，可能仍被车挡住。</td></tr><tr><td>原GT框视线通过</td><td>{c['raw_box_ever_all_boxes_clear']} / {c['queries']}</td><td>未闭合车底缝隙，保留为错误风险对照。</td></tr><tr><td>贴地包络视线通过</td><td>{c['ever_all_boxes_clear']} / {c['queries']}</td><td>保守几何候选，尚未排除围栏等静态遮挡。</td></tr><tr><td>仅原ProPainter相机：贴地视线通过</td><td>{c['same_poc_camera_ever_all_boxes_clear']} / {c['queries']}</td><td>原补景按相机独立处理。</td></tr><tr><td>视线通过且有15cm内地面LiDAR</td><td>{c['both_box_clear_and_lidar_support']} / {c['queries']}</td><td><strong>联合候选为0，禁止标为已观测背景。</strong></td></tr></table>
 <details open><summary>实际视图：原框与贴地包络相差在哪里</summary><p>全图青框定位下面裁剪。橙色是原GT框，青色是只向下延伸到局部地面以下10cm的遮挡包络，粉点是固定世界坐标的车底地面查询。上表面与水平尺寸不变；图是诊断标注，车辆和资产没有移动。</p><img class="context" src="{s}/box_context.jpg" alt="下方裁剪在相机全图的位置"><a href="{s}/box_control.jpg"><img src="{s}/box_control.jpg" alt="源图、原GT框、贴地包络放大对照"></a></details>
 <p class="small">此图只检查近似地面，不覆盖删除后露出的墙、围栏等竖直表面。LiDAR是距离证据，未声称其与RGB取样同一条射线。未知区域仍为未知。<a href="{s}/visibility.json">本场景统计与准入状态</a> · <a href="{s}/view_counts.json">逐相机/时刻计数</a></p></section>''')

head='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 R3 · 背景观测检查</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf3f6;color:#163449;font:16px/1.75 "Microsoft YaHei",system-ui,sans-serif}header{padding:36px max(24px,calc((100vw - 1460px)/2));background:#123247;color:white}h1{font-size:34px;line-height:1.35}header p{max-width:1120px;color:#d2e6ed}a{color:#087b71}header a{color:#a8f5dd}main{max-width:1510px;margin:auto;padding:24px}section{padding:25px;background:white;border:1px solid #c8dce3;border-radius:14px;margin:28px 0}img{width:100%;height:auto;display:block;margin:15px 0;border-radius:8px}.context{max-width:900px}.flow{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.flow span{padding:12px;border:1px solid #97c4be;border-radius:8px;background:#e1f1ed}.notice{padding:18px;background:#fff2de;border-left:5px solid #bc8b35}.videos{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.videos article{border:1px solid #bcd3dc;background:#f2f8fa;border-radius:9px;overflow:hidden}.videos h4,.videos p{margin:10px 12px}.videos p{min-height:48px;font-size:13px}video{width:100%;display:block}.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:18px 0}.controls button{padding:9px 13px;border:1px solid #539b8f;border-radius:7px;background:#eaf7f1;cursor:pointer}.controls input{width:240px}table{width:100%;border-collapse:collapse;margin:20px 0}th,td{border:1px solid #c8dce3;padding:12px;text-align:left}th{background:#e6f0f3}.small{font-size:13px;color:#506d81}summary{padding:12px;background:#e8f0f5;cursor:pointer;font-weight:bold}@media(max-width:950px){.videos{grid-template-columns:1fr}section{padding:15px}main{padding:12px}h1{font-size:27px}table{font-size:13px}th,td{padding:6px}}
</style></head><body><header><p>WS-V77-PIPELINE-R3-20260926 / r1</p><h1>背景先查证据：车框下面的“缝隙”不能当成可用背景</h1><p>本轮新增贴地遮挡检查与背景参考准入状态。发现scene_0230直接使用原GT框会产生大量假可见地面；修正后，两目标车底核心都没有“视线候选 + 近邻地面LiDAR”联合支持。原GLB与已有视频保持不变，没有新模型前向或训练。</p><a href="#scene_0230">actor 22</a> · <a href="#scene_0255">actor 25</a> · <a href="../v77-pipeline-r2/index.html">R2完整时序补景控制</a></header><main>
<div class="flow" aria-label="architecture components"><span>原RGB + GT相机/轨迹 + LiDAR</span><b>→</b><span>拟合局部地面 / 固定采样</span><b>→</b><span>GT框下延贴地 / 6相机视线</span><b>→</b><span>与地面LiDAR支持取交集</span><b>→</b><span>候选 / 未知；保存证据</span></div>
<p class="notice"><strong>这轮定位了观测检查中的风险，还没有把DELETE画面修好。</strong>两场景的真实车底背景目前缺少可直接复制的依据；这不等于证明所有背景永远不可见，也没有将残影归因于GLB。原MOVE仍未准入。阴影、mask、成像各自的影响尚未分离。</p>'''
tail='''<section><h2>接下来怎样推进</h2><p>停止基于这批车底候选做“真实背景复制”，也不继续扫ProPainter视频长度。下一个有界检查针对scene_0255已有8张生成参考：围栏/杆件是否污染输入，能否选到证据更完整的参考；维持现有网格与模型权重，先完成输入对照再决定是否重跑同一个Hunyuan材质流程。</p><p>人工verdict保持空白。几何检查使用GT与LiDAR作为诊断输入，不声称RGB端到端自动。<a href="report.md">技术记录与边界</a> · <a href="summary.json">完整统计</a> · <a href="validation.json">产物检查</a></p></section></main><script>
const data=__DATA__;for(const section of document.querySelectorAll('[data-scene]')){const r=data.scenes.find(x=>x.name===section.dataset.scene),vs=[...section.querySelectorAll('video')],bar=section.querySelector('.scrub'),out=section.querySelector('output');function label(){const f=r.frames[Number(bar.value)];out.textContent=`f${String(f.frame).padStart(3,'0')} · ${(f.frame/10).toFixed(1)} s · CAM${f.camera}`;}bar.oninput=()=>{vs.forEach(v=>{v.pause();v.currentTime=Number(bar.value)/r.fps+.001});label();};section.querySelector('.play').onclick=()=>vs.forEach(v=>{v.currentTime=Number(bar.value)/r.fps;v.play().catch(()=>{})});section.querySelector('.pause').onclick=()=>vs.forEach(v=>v.pause());vs[0].ontimeupdate=()=>{if(!vs[0].paused){bar.value=Math.min(9,Math.floor(vs[0].currentTime*r.fps));label();vs.slice(1).forEach(v=>{if(Math.abs(v.currentTime-vs[0].currentTime)>.2)v.currentTime=vs[0].currentTime})}};label();}
</script></body></html>'''
(out/'index.html').write_bytes((head+''.join(sections)+tail.replace('__DATA__',json.dumps(timing,ensure_ascii=False))).encode('utf-8'))
for name in ['summary.json','registration.json']:shutil.copy2(src/name,out/name)
old=root/'outputs/v77-explicit-poc/index.html';backup=old.with_name('index_before_r3.html')
if not backup.exists():shutil.copy2(old,backup)
html=old.read_text(encoding='utf-8');banner='<div id="r3-update" style="background:#123247;color:white;padding:18px;font:17px/1.7 sans-serif">最新R3：背景观测证据检查，原GLB保留。<a style="color:#b6ffe3" href="../v77-pipeline-r3/index.html">打开新审核页 →</a></div>'
if 'id="r3-update"' not in html:old.write_bytes(html.replace('<body>','<body>'+banner,1).encode('utf-8'))
print(out/'index.html')

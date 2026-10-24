"""Build the V4.2 design and actual-point audit report from read-only CPU outputs."""
from pathlib import Path
import json, html, re, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/V42_Design_Audit_20261024'
OUT.mkdir(parents=True,exist_ok=True)
s=json.loads((HERE/'results/gate_summary.json').read_text());z=np.load(HERE/'results/audit_arrays.npz');actors=np.load(HERE/'results/actor_units.npz')
examples=json.loads((HERE/'results/point_examples.json').read_text())
names={'baseline':'基线：旧九门限','cosine_0.06':'观察角度 0.12 → 0.06','drop_boxes_clear':'取消框：仅诊断上限','mask_margin_0':'掩码膨胀 2 → 0','depth_0.5':'深度误差 0.3 → 0.5 m','minimum_views_1':'来源要求 2 → 1','sparse_radius_3':'稀疏深度半径 1.5 → 3 px','patch_box_recheck':'三角形＋跨帧框复核','native_pixel_convention':'统一原生像素中心'}
bits=sum(actors[k].astype(int)*(1<<i) for i,k in enumerate(actors.files))
data={'xy':np.round(z['xy'],3).tolist(),'ids':z['unit_ids'].tolist(),'supported':z['supported'].astype(int).tolist(),'base':z['baseline_active'].astype(int).tolist(),'actor_keys':actors.files,'actor_bits':bits.tolist(),'variants':{k:{'eligible':z['eligible_'+k].astype(int).tolist(),'counts':z['source_count_'+k].tolist(),'summary':s['variants'][k]} for k in names},'names':names}
(OUT/'points.js').write_text('window.POINTS='+json.dumps(data,separators=(',',':'),ensure_ascii=True)+';\n')
for filename in ['gate_summary.json','point_examples.json','audit_arrays.npz','actor_units.npz']:shutil.copy2(HERE/'results'/filename,OUT/filename)
shutil.copy2(HERE/'V42_DESIGN.md',OUT/'V42_DESIGN.md')

svg='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 450" role="img" aria-labelledby="title"><title id="title">V4.2完整重建主副本与独立编辑副本：设计，尚未实现</title><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10" fill="#476b81"/></marker></defs><rect width="1180" height="450" fill="white"/><g font-family="Microsoft YaHei, sans-serif" text-anchor="middle"><g fill="#edf5fb" stroke="#7199b3" stroke-width="1.5"><rect x="20" y="35" width="220" height="80" rx="10"/><rect x="365" y="35" width="300" height="80" rx="10"/><rect x="920" y="35" width="230" height="80" rx="10"/></g><g fill="#e8f5ed" stroke="#6b9b81" stroke-width="1.5"><rect x="20" y="245" width="220" height="90" rx="10"/><rect x="320" y="245" width="240" height="90" rx="10"/><rect x="620" y="245" width="250" height="90" rx="10"/><rect x="920" y="245" width="230" height="90" rx="10"/><rect x="620" y="140" width="250" height="65" rx="10"/></g><g fill="#18374c" font-size="20"><text x="130" y="68">RGB / LiDAR / 位姿</text><text x="130" y="96">已有车辆 mask</text><text x="515" y="68">原 V1 训练 → 固定主副本</text><text x="515" y="96">原生渲染器，无编辑回写</text><text x="1035" y="68">默认完整重建</text><text x="1035" y="96">原 V1 输出</text><text x="130" y="282">世界道路证据账本</text><text x="130" y="311">观测 / 先验 / 未知</text><text x="440" y="282">走廊曲面与新位置提案</text><text x="440" y="311">跨视图匹配、几何核验</text><text x="745" y="169">独立编辑副本</text><text x="745" y="194" font-size="16">独立参数 / 优化器 / 文件</text><text x="745" y="282">补点 ↔ 静态污染抑制</text><text x="745" y="311">成对干预、实际贡献验证</text><text x="1035" y="282">编辑输出</text><text x="1035" y="311">保留车辆 / 道路 / 未知</text></g><g fill="none" stroke="#476b81" stroke-width="2" marker-end="url(#a)"><path d="M240 75H360"/><path d="M665 75H915"/><path d="M130 115V240"/><path d="M240 290H315"/><path d="M560 290H615"/><path d="M870 290H915"/><path d="M515 115V172H615"/><path d="M745 205V240"/></g><g fill="#486777" font-size="16"><text x="790" y="61">完整请求直接通行</text><text x="526" y="158">单向复制</text><text x="590" y="385">核心研究假设：补几何与移除错遮挡须联合选择，不能只提高独立道路 alpha</text><text x="590" y="419">组件为 V4.2 设计；本次仅运行 CPU 门限审计，未启动 V4.2 GPU 实验</text></g></g></svg>'''
(HERE/'architecture.svg').write_text(svg,encoding='utf-8');(OUT/'architecture.svg').write_text(svg,encoding='utf-8')

# Publication-independent static plot: same point arrays as the interactive panel.
fig,axs=plt.subplots(1,2,figsize=(12,5.6),sharex=True,sharey=True)
for ax,(name,title) in zip(axs,[('cosine_0.06','Angle threshold: 0.12 to 0.06'),('drop_boxes_clear','Remove boxes: diagnostic upper bound')]):
 xy=z['xy'];new=z['eligible_'+name]&~z['baseline_active'];base=z['baseline_active'];sup=z['supported']
 for sel,color,label,size in [(~sup,'#d9dfe5','Not created',1),(sup&~base&~new,'#9faebb','Created, excluded',2),(base,'#257fba','Original eligible',3),(new,'#dc8a15','New candidate',5)]:
  ax.scatter(xy[sel,0],xy[sel,1],s=size,c=color,label=label,rasterized=True)
 for idx in [1652,2987]:ax.scatter(xy[idx,0],xy[idx,1],s=65,c='#bd2f5a',marker='x',linewidths=2)
 ax.set_title(title);ax.set_xlabel('Local longitudinal coordinate (m)');ax.grid(alpha=.15)
axs[0].set_ylabel('Local lateral coordinate (m)');axs[1].legend(markerscale=2,fontsize=8,loc='upper right')
fig.suptitle('Actual world units; candidate gains are not reconstruction gains');fig.tight_layout();fig.savefig(OUT/'candidate_distribution.png',dpi=170);plt.close(fig)

def inline(text):
 text=html.escape(text)
 text=re.sub(r'!\[([^\]]*)\]\(([^)]*)\)',r'<img src="\2" alt="\1">',text)
 text=re.sub(r'\[([^\]]*)\]\(([^)]*)\)',r'<a href="\2">\1</a>',text)
 text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
 return re.sub(r'`([^`]+)`',r'<code>\1</code>',text)

def markdown(text):
 lines=text.splitlines();out=[];i=0
 while i<len(lines):
  line=lines[i]
  if not line.strip():i+=1;continue
  if line.startswith('|'):
   rows=[]
   while i<len(lines) and lines[i].startswith('|'):
    cells=[c.strip() for c in lines[i].strip('|').split('|')]
    if not all(re.fullmatch(r'[:\- ]+',c) for c in cells):rows.append(cells)
    i+=1
   out.append('<div class="scroll"><table>'+''.join('<tr>'+''.join(f'<{"th" if n==0 else "td"}>{inline(c)}</{"th" if n==0 else "td"}>' for c in row)+'</tr>' for n,row in enumerate(rows))+'</table></div>');continue
  match=re.match(r'^(#{1,4}) (.*)',line)
  if match:out.append(f'<h{len(match[1])}>{inline(match[2])}</h{len(match[1])}>')
  elif re.match(r'^(- |\d+\. )',line):out.append('<p class="item">'+inline(line)+'</p>')
  else:out.append('<p>'+inline(line)+'</p>')
  i+=1
 return '\n'.join(out)

style='''body{margin:0;background:#edf2f6;color:#183346;font:17px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:30px 24px 70px}h1{font-size:32px;line-height:1.35}h2{font-size:24px;margin-top:32px}h3{font-size:20px}section,.card{background:white;padding:24px;border:1px solid #d0dde7;border-radius:12px;margin:18px 0}img{max-width:100%;height:auto}a{color:#08659d}table{width:100%;border-collapse:collapse;font-size:15px}td,th{padding:10px 12px;border-bottom:1px solid #dce4ea;vertical-align:top;text-align:left}th{background:#edf4f8}.scroll{overflow-x:auto}.muted{color:#597286;font-size:14px}.notice{background:#fff3d9;padding:14px;border-left:4px solid #c58b21}select,button{font:inherit;padding:7px 10px;margin:4px;border:1px solid #91abbf;border-radius:5px;background:white}canvas{width:100%;height:auto;display:block;border:1px solid #c5d6e1;background:#fafcfd;cursor:crosshair}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:14px}.example{display:grid;grid-template-columns:1fr 1fr;gap:18px}code{background:#edf2f6;padding:2px 4px;overflow-wrap:anywhere}.item{padding-left:12px}pre{white-space:pre-wrap;font-size:13px}.witness{position:relative}.witness img{width:100%;display:block}.cross{position:absolute;width:16px;height:16px;border:2px solid #ff1d50;border-radius:50%;transform:translate(-50%,-50%);pointer-events:none}details{margin-top:20px}summary{cursor:pointer;font-weight:bold}@media(max-width:700px){.example{grid-template-columns:1fr}main{padding:20px 12px}section{padding:16px}}'''
doc=markdown((HERE/'V42_DESIGN.md').read_text(encoding='utf-8'))
(OUT/'design.html').write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V4.2 算法与架构设计</title><style>{style}</style><main><a href="index.html">← 返回逐点门限审计</a><section>{doc}</section></main></html>',encoding='utf-8')

cards=[];asset=OUT/'source_frames';asset.mkdir(exist_ok=True)
for ex in examples:
 if ex['variant']=='depth_0.5':continue
 witnesses=[]
 for w in ex['witnesses']:
  fid=w['frame'];src=HERE.parent/'v41/report_assets_r5/original'/f'{fid}.png'
  shutil.copy2(src,asset/src.name)
  x,y=w['uv_legacy'];witnesses.append(f'<div><div class="witness"><img src="source_frames/{fid}.png" alt="原始帧{fid}中的候选投影"><span class="cross" style="left:{100*x/480:.5f}%;top:{100*y/320:.5f}%"></span></div><small>raw{int(fid)-8000} · 旧门限投影({x:.1f},{y:.1f}) · cos={w["cosine"]:.5f}<br>原规则失败项：{html.escape(str(w["failed_baseline_gates"]))}</small></div>')
  # This is a display preview of original RGB, never a training label.
 cards.append(f'<section><h3>单元 {ex["unit_id"]} · a{ex["actor_query_union"]} 查询区域 · {names[ex["variant"]]}</h3><p>局部坐标 {ex["local_xy"]}；合格来源 {ex["baseline_sources"]} → {ex["variant_sources"]}。红圈为旧筛选投影的显示预览，原图未改，不能用红圈旁的车身颜色当道路标签。</p><div class="example">'+''.join(witnesses)+'</div></section>')

js='''const d=window.POINTS,c=document.querySelector('canvas'),g=c.getContext('2d'),v=document.querySelector('#variant'),a=document.querySelector('#actor'),info=document.querySelector('#point');
Object.entries(d.names).forEach(([k,n])=>v.add(new Option(n,k)));v.value='cosine_0.06';d.actor_keys.forEach((k,i)=>a.add(new Option('a'+k+' 查询单元并集',i)));
const xs=d.xy.map(p=>p[0]),ys=d.xy.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),pad=42;
const px=x=>pad+(x-xmin)/(xmax-xmin)*(c.width-2*pad),py=y=>c.height-pad-(y-ymin)/(ymax-ymin)*(c.height-2*pad);let visible=[],selected=-1;
function draw(){g.clearRect(0,0,c.width,c.height);let q=d.variants[v.value],stats=q.summary;visible=[];g.font='14px sans-serif';g.fillStyle='#456376';g.fillText('局部前向 x（米） →',c.width/2-50,c.height-7);g.fillText('侧向 y',2,20);
for(let x=Math.ceil(xmin/10)*10;x<=xmax;x+=10){g.strokeStyle='#e1e9ef';g.beginPath();g.moveTo(px(x),pad);g.lineTo(px(x),c.height-pad);g.stroke();g.fillStyle='#597286';g.fillText(x,px(x)-8,c.height-24)}
for(let pass=0;pass<3;pass++)for(let i=0;i<d.xy.length;i++){if(a.value!=='all'&&!(d.actor_bits[i]&(1<<Number(a.value))))continue;const added=q.eligible[i]&&!d.base[i],lost=d.base[i]&&!q.eligible[i];const p=added||lost?2:d.base[i]?1:0;if(pass!==p)continue;let color=added?'#dd8b16':lost?'#be4383':d.base[i]?'#217db5':d.supported[i]?'#a2b0ba':'#e0e5e9';g.fillStyle=color;const r=added||lost?2.1:1.25;g.fillRect(px(d.xy[i][0])-r,py(d.xy[i][1])-r,2*r,2*r);visible.push(i)}
for(let id of [1652,2987]){let i=d.ids.indexOf(id);g.strokeStyle='#c3294f';g.lineWidth=2;g.beginPath();g.moveTo(px(d.xy[i][0])-6,py(d.xy[i][1])-6);g.lineTo(px(d.xy[i][0])+6,py(d.xy[i][1])+6);g.moveTo(px(d.xy[i][0])-6,py(d.xy[i][1])+6);g.lineTo(px(d.xy[i][0])+6,py(d.xy[i][1])-6);g.stroke();g.fillStyle='#9b1c3b';g.fillText(id,px(d.xy[i][0])+8,py(d.xy[i][1])-8)}
document.querySelector('#stats').textContent='全场合格 '+stats.eligible_supported_units+'；较基线新增 '+stats.added_vs_1919+'、失去 '+stats.lost_vs_1919+'。显示 '+visible.length+' 个网格单元。逐车筛选是世界单元并集，非当前帧覆盖率。';if(selected>=0)show(selected);}
function show(i){selected=i;let q=d.variants[v.value];info.textContent='unit '+d.ids[i]+' | 局部坐标 '+d.xy[i].join(', ')+' m | 已创建 '+!!d.supported[i]+' | 原来源 '+d.variants.baseline.counts[i]+' → 本变体来源 '+q.counts[i]+' | 原资格 '+!!d.base[i]+' → 本变体资格 '+!!q.eligible[i];}
c.addEventListener('click',e=>{let b=c.getBoundingClientRect(),x=(e.clientX-b.left)*c.width/b.width,y=(e.clientY-b.top)*c.height/b.height,best=-1,dist=144;for(let i of visible){let t=(px(d.xy[i][0])-x)**2+(py(d.xy[i][1])-y)**2;if(t<dist){dist=t;best=i}}if(best>=0)show(best)});v.onchange=draw;a.onchange=draw;draw();'''
(OUT/'audit.js').write_text(js,encoding='utf-8')
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V4.2 · 门限审计与独立编辑架构</title><style>{style}</style><main>
<p class="muted">2026-10-24 · 49 帧真实输入 · CPU只读审计 · 新增优化更新0 · V4.2尚未运行GPU</p><h1>门限有保守性；中心缺口还需要补几何和处理静态污染</h1>
<section><p><b>调参有值得验证的收益：</b>角度余弦0.12→0.06增加1,722个候选，其中818个通过不沿用旧角度门限的小三角形/分块颜色复核。但a37查询区域只新增43个，a6为0。取消三维框可多出425个，仅是需要像素级核验的上限。</p><p><b>两条路需要真正隔离：</b>完整请求直接走固定原V1主副本；只有编辑请求进入独立副本，执行道路补全、局部污染抑制与真实深度合成。</p><p class="notice">所有新增点均为候选。没有重训、没有新增质量指标，不能把“数量增加”当作“道路改善”。两个红叉位置原本没有创建单元，资格调参不能生成它们。</p><a href="design.html">阅读完整 V4.2 算法设计、先例比较和评价方案 →</a></section>
<section><h2>组件与数据流</h2><img src="architecture.svg" alt="原V1主副本完整输出，独立编辑副本进行补点与静态污染处理"></section>
<section><h2>实际世界单元分布：可切换规则、逐点查看</h2><label>变体 <select id="variant"></select></label><label>区域 <select id="actor"><option value="all">全场</option></select></label><p id="stats"></p><div class="legend"><span style="color:#217db5">■ 原资格</span><span style="color:#dd8b16">■ 新增候选</span><span style="color:#be4383">■ 失去资格</span><span style="color:#8396a4">■ 已创建但未纳入</span><span>浅灰：未创建</span><span style="color:#c3294f">× 两个中心缺口</span></div><canvas width="1120" height="470" aria-label="真实候选世界位置，点击查询单元"></canvas><p id="point">点击点查看单元ID、来源数和资格；可复核不是训练裁剪。</p><p class="muted">九项筛选精确复现，逐项差异均为0。数据是现有网格，不是新视角渲染或真实地形GT。逐车区域可重叠，不可相加。原生像素约定变体另列，避免把工程坐标变化混为算法收益。</p><a href="candidate_distribution.png">下载静态点分布图</a> · <a href="gate_summary.json">完整审计JSON</a> · <a href="audit_arrays.npz">逐点原始数组</a></section>
<section><h2>实际来源图：哪些点仅被单一规则拦住</h2><p>以下按每个变体/区域的局部x中位候选自动选取，不声称代表全部候选。完整图为原始帧的显示预览；红圈仅标投影位置。邻帧颜色相似不能证明隐藏路面真值。</p></section>{''.join(cards)}
<section><h2>设计的主张与自评</h2><p>走廊建面、MVS增点、法线/贴面约束已有先例。V4.2拟验证的是：在编辑副本中，利用删车后实际Tα贡献，联合选择补点和局部静态污染抑制，避免“补出的路面仍被错误静态挡住”。若简单调参加标准多视图增点已足够，应优先采用简单方案。</p><p>本页已完成的是审计和设计，完整V4.2训练/渲染引擎仍待实现。<a href="design.html">完整设计</a>含GaussianPro、VAD-GS、RoGS、PGSR等原论文链接及可证伪的实验矩阵。</p></section><p class="muted">quality_accepted=false · human_verdict=null · 分支 research/worldsim-v8.2-drivingeditor-v42</p></main><script src="points.js"></script><script src="audit.js"></script></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8')
print(json.dumps({'report':str(OUT/'index.html'),'files':len(list(OUT.rglob('*'))),'point_count':len(z['xy']),'examples':len(cards)},ensure_ascii=True))

"""Build the local, offline scientific evidence report and crop comparison."""
from pathlib import Path
import json
import html
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[2]
OUT = BASE/'outputs/scene008_local_colour_diagnosis'
data = json.loads((OUT/'evidence/full49_metrics.json').read_text(encoding='utf-8'))
split = json.loads((OUT/'evidence/reference_split_metrics.json').read_text(encoding='utf-8'))
rows = [r for r in data['rows'] if r['state'] == 'keep']
stages = {'raw':'原始 RGB', 'native':'原生 10K', 'gates':'仅静态门控',
          'road_off':'道路 1K', 'road_fine':'道路＋颜色校正', 'asset_fine':'完整车辆替换'}
regions = {'left_road_patch':'左侧道路', 'right_curb_patch':'右侧路缘',
           'right_building':'右侧建筑', 'left_trees':'左侧树木', 'observed_road':'已观测道路'}


def pooled(stage, roi, metric):
    total = sum(r['metrics'][stage][roi]['pixels'] for r in rows)
    return sum(r['metrics'][stage][roi][metric]*r['metrics'][stage][roi]['pixels'] for r in rows)/total


def edge(stage, metric):
    total = sum(r['boundary'][stage]['neighbor_pairs'] for r in rows)
    return sum(r['boundary'][stage][metric]*r['boundary'][stage]['neighbor_pairs'] for r in rows)/total


findings = dict(task=data['task'], frames=49, training_updates=0,
                local_blue={roi:dict(mean=pooled('native',roi,'blue_error'),
                                     positive_frames=sum(r['metrics']['native'][roi]['blue_error'] > 0 for r in rows),
                                     min=min(r['metrics']['native'][roi]['blue_error'] for r in rows),
                                     max=max(r['metrics']['native'][roi]['blue_error'] for r in rows)) for roi in regions},
                seam={s:dict(jump=edge(s,'rgb_jump'), gradient_error=edge(s,'rgb_gradient_error')) for s in stages},
                attribution_maxabs=max(r['attribution_maxabs'] for r in data['rows']),
                native_binding_maxabs=max(r.get('native_reference_maxabs',0) for r in data['rows']),
                full49_seconds=data['seconds'], pilot_seconds=json.loads((OUT/'evidence/pilot_metrics.json').read_text())['seconds'])
(OUT/'findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

font_path = 'C:/Windows/Fonts/msyh.ttc'
font = ImageFont.truetype(font_path,20)
small = ImageFont.truetype(font_path,17)
big = ImageFont.truetype(font_path,28)
canvas = Image.new('RGB',(1440,1085),'#f6f8fc')
draw = ImageDraw.Draw(canvas)
draw.text((20,12),'局部蓝色确实超出了原图；道路层又增加了颜色边界',font=big,fill='#162638')
draw.text((20,57),'raw113 / 同一相机、同一时刻、统一显示，无自动白平衡',font=small,fill='#536578')
cols = [('raw','原始 RGB'),('native','原生 10K'),('road_fine','道路恢复＋颜色校正')]
for i,(stage,label) in enumerate(cols):
    image = Image.open(OUT/f'evidence/lossless_pilot113/113_keep_{stage}.png').convert('RGB')
    x=i*480
    draw.text((x+12,92),label,font=font,fill='#182c40')
    canvas.paste(image,(x,127))
    frame = ImageDraw.Draw(canvas)
    for box,color in [((0,205,215,320),'#ed734d'),((315,177,480,290),'#13a7a6')]:
        frame.rectangle((x+box[0],127+box[1],x+box[2]-1,127+box[3]-1),outline=color,width=2)
    for j,(box,label2) in enumerate([((0,205,215,320),'左侧路面，橙框'),((315,177,480,290),'右侧路缘，绿框')]):
        crop=image.crop(box)
        crop.thumbnail((464,240),Image.Resampling.NEAREST)
        scale=min(464/(box[2]-box[0]),240/(box[3]-box[1]))
        crop=image.crop(box).resize((round((box[2]-box[0])*scale),round((box[3]-box[1])*scale)),Image.Resampling.NEAREST)
        y=480+j*295
        draw.text((x+12,y),label2,font=small,fill='#536578')
        canvas.paste(crop,(x+8,y+31))
canvas.save(OUT/'local_comparison.png')

svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="205" viewBox="0 0 1100 205"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0L7 3L0 6" fill="#58738d"/></marker></defs><rect width="1100" height="205" rx="16" fill="#f2f6fa"/><g font-family="Microsoft YaHei, sans-serif" font-size="18" text-anchor="middle" fill="#173148"><g stroke="#aac2d6" fill="white"><rect x="20" y="60" width="220" height="85" rx="10"/><rect x="290" y="20" width="260" height="165" rx="10"/><rect x="600" y="60" width="215" height="85" rx="10"/><rect x="865" y="60" width="215" height="85" rx="10"/></g><text x="130" y="94">原始 RGB、相机</text><text x="130" y="122">冻结 10K／道路状态</text><text x="420" y="55">同相机逐步对照</text><text x="420" y="86">原生 → 仅门控 → 道路</text><text x="420" y="117">→ 颜色校正 → 车辆替换</text><text x="420" y="155" font-size="16">父模型冻结，统一深度排序</text><text x="707" y="94">局部颜色残差</text><text x="707" y="122">分层 RGB／边界测量</text><text x="972" y="94">49 帧验证</text><text x="972" y="122">来源与证据边界</text></g><g stroke="#58738d" stroke-width="2" marker-end="url(#a)"><path d="M240 103H285"/><path d="M550 103H595"/><path d="M815 103H860"/></g></svg>'''
(OUT/'architecture.svg').write_text(svg,encoding='utf-8')

blue_table=''.join(f'<tr><td>{name}</td><td>{pooled("raw",roi,"blue_index"):.5f}</td><td>{pooled("native",roi,"blue_index"):.5f}</td><td><b>{pooled("native",roi,"blue_error"):+.5f}</b></td><td>{findings["local_blue"][roi]["positive_frames"]}/49</td></tr>' for roi,name in regions.items())
edge_table=''.join(f'<tr><td>{label}</td><td>{edge(stage,"rgb_jump"):.5f}</td><td>{edge(stage,"rgb_gradient_error"):.5f}</td></tr>' for stage,label in stages.items())
component_table=''
for row in split['rows']:
    for roi in ('left_road_patch','right_curb_patch'):
        terms=row['regions'][roi]
        component_table+=f'<tr><td>{row["raw"]} · {regions[roi]}</td><td>{terms["static"]["blue"]:+.5f}</td><td>{terms["environment"]["blue"]:+.5f}</td><td>{sum(v["blue"] for v in terms.values()):+.5f}</td></tr>'
compact=[dict(raw=r['raw'],state=r['state'],metrics=r['metrics'],boundary=r['boundary']) for r in data['rows']]
nativejump=edge('native','rgb_jump')
offjump=edge('road_off','rgb_jump')
finejump=edge('road_fine','rgb_jump')
title='Scene008：局部偏蓝与道路色彩边界'
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f7;color:#1a2d41;font:16px/1.7 "Microsoft YaHei",system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:30px}h1{font-size:32px;line-height:1.35;margin:0 0 16px}h2{font-size:23px;margin:32px 0 12px}h3{font-size:18px}p{margin:12px 0}.muted,small{color:#5d7082}.card{background:white;padding:24px;border:1px solid #d9e2ec;border-radius:14px;margin:18px 0}.lead{font-size:19px}.badge{display:inline-block;background:#e7f3ee;color:#176e50;padding:4px 10px;border-radius:6px;font-size:14px;margin:3px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:9px 13px;border-bottom:1px solid #dce5ed}th{background:#f3f6fa}a{color:#126999}code{font:14px Consolas,monospace;background:#edf2f6;padding:2px 5px;overflow-wrap:anywhere}.gallery{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.tile{border:1px solid #dbe4ed;border-radius:10px;overflow:hidden;background:#f5f7fa}.tile h3{margin:6px 12px;font-size:16px}.tile canvas{width:100%;height:auto;display:block;background:#09121c}.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:16px 0;position:sticky;top:0;background:#fff;padding:12px;z-index:2;border-bottom:1px solid #dae4ed}button,select{font:inherit;border:1px solid #b8cad9;background:#fff;padding:5px 12px;border-radius:6px}input[type=range]{width:240px}.wide{display:block;width:100%;height:auto}details{margin:18px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f6;padding:15px;border-radius:8px}.callout{border-left:4px solid #d17b3a;padding:4px 16px;background:#fff7ef}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:850px){main{padding:15px}.gallery,.grid2{grid-template-columns:1fr}h1{font-size:27px}.card{padding:15px}table{font-size:13px}th,td{padding:7px}}
</style><main><h1>__TITLE__</h1><p class="muted">2026-11-05 · 原始相机分辨率 480×320 · 本次仅做冻结诊断</p><p><span class="badge">49 帧原场景对照</span><span class="badge">3 帧删除状态核对</span><span class="badge">0 次训练更新</span><span class="badge">AutoDL 保持开机</span></p>
<div class="card"><p class="lead"><b>你指出的局部偏蓝确实超出了原图。道路色彩边界则在恢复道路插入时进一步增强。</b></p><p>左侧道路、右侧路缘在 49 帧里都比原图更蓝。原生重建中的方向背景通过剩余透射率参与道路和建筑成像；局部静态颜色与它相加后的误差没有完全抵消。新道路采用另一套固定颜色与透明度，插入后同时改变原静态层和方向背景的可见份额，因此边界两侧出现色差。</p><p>此前用全图平均颜色判断“原图本来就偏冷”不够充分：天空和其他区域会稀释局部异常。这份报告改为比较同相机、同位置的原始 RGB 和各阶段输出。</p><img class="wide" src="architecture.svg" alt="输入、冻结对照、逐像素分层和49帧验证架构图"></div>
<div class="card"><h2>同位置直接对照</h2><p>橙框为左侧道路，绿框为右侧路缘。下方用最近邻放大，保留实际像素；三列使用同一显示方式。</p><a href="local_comparison.png"><img class="wide" src="local_comparison.png" alt="原图、原生10K与道路恢复的全图和局部对照"></a></div>
<div class="card"><h2>49 帧切换与局部放大</h2><p>“保留”状态有同帧原图可对照；“删除”仅展示 105、113、130 三个诊断帧。被删车辆原本遮住的区域没有同帧删后真值，颜色表格排除了已知车辆区域。</p><div class="controls"><label>帧 <select id="frame"></select></label><input id="slider" type="range" min="89" max="137" value="113"><label>状态 <select id="state"><option value="keep">保留</option><option value="delete">删除（三帧）</option></select></label><label>区域 <select id="roi"><option value="full">完整图</option><option value="left_road_patch">左侧道路</option><option value="right_curb_patch">右侧路缘</option><option value="right_building">右侧建筑</option><option value="left_trees">左侧树木</option></select></label><span id="load"></span></div><div class="gallery" id="gallery"></div><p id="caption" class="muted"></p><table><thead><tr><th>阶段</th><th>相对原图的蓝色残差</th><th>RGB MAE</th><th>蓝色残差 &gt;0.03 的像素</th></tr></thead><tbody id="numbers"></tbody></table><p><a id="lossless" href="evidence/lossless_pilot113/113_keep_native.png">raw113 原生输出无损 PNG</a> · <a href="frames/113_keep_native_blue_error.jpg">原生蓝色残差图</a>（橙红：额外偏蓝；青：更少蓝；固定幅度 ±0.08）。</p></div>
<div class="card"><h2>局部偏蓝：原生重建已经存在</h2><p>蓝色量定义为 <code>B−(R+G)/2</code>。这里比较渲染值与原图的差，数值范围按 RGB 0–1 计。均值按各区域有效像素数加权，已知车辆像素排除。</p><table><thead><tr><th>固定画面区域</th><th>原图蓝色量</th><th>原生蓝色量</th><th>新增蓝色残差</th><th>残差为正帧数</th></tr></thead><tbody>__BLUE_TABLE__</tbody></table><p>左侧道路新增蓝色均值为 <b>__LEFT_BLUE__</b>，右侧路缘为 <b>__RIGHT_BLUE__</b>。左侧树木整体残差接近零，说明问题有明显位置差异，不能用一个全图色温数值概括。</p></div>
<div class="card"><h2>为什么方向背景会影响路面颜色</h2><p>实际成像为 <code>RGB = Σ(wᵢ·颜色ᵢ) + T·方向背景RGB</code>，其中 <code>T=1−总透明度</code>。方向背景是按观察方向查询的已学习 RGB；它在当前模型里承载了天空之外的画面内容。道路和建筑处只要 T 不为零，该项也会参与颜色。</p><p>为了定位相对原图的新增偏蓝，三帧诊断按每层实际覆盖权重分配同像素原图，得到 <code>层残差=层RGB−层权重×原图RGB</code>。环境项对应 <code>T×(方向背景−原图)</code>。各项相加与真实 RGB 残差一致。</p><table><thead><tr><th>同位置诊断</th><th>静态层蓝色残差</th><th>方向背景蓝色残差</th><th>所有层合计</th></tr></thead><tbody>__COMPONENT_TABLE__</tbody></table><p>左侧道路的方向背景和静态层都能带来新增蓝色；右侧路缘主要体现为方向背景的蓝色残差与静态层负残差未完全抵消。这是固定模型的成像项归因；分层本身没有独立颜色真值，也尚未证明训练中的哪一步首次造成这种补偿关系。</p><div class="grid2"><figure><img class="wide" src="evidence/113_static_blue_error.png" alt="原生静态层相对覆盖权重参考的蓝色残差"><figcaption>raw113：静态层残差</figcaption></figure><figure><img class="wide" src="evidence/113_environment_blue_error.png" alt="方向背景相对覆盖权重参考的蓝色残差"><figcaption>raw113：方向背景残差；与左图采用相同色标</figcaption></figure></div></div>
<div class="card"><h2>道路割裂：插入后的颜色与可见份额没有衔接好</h2><p>先仅应用保存的静态透明度门控，再插入固定道路，最后启用已有世界颜色场。每一步都保留相同相机与原生排序规则。统计固定道路贡献边界（road αT=0.05）两侧相邻像素的 RGB 跳变，并与原图相同位置的梯度比较。</p><table><thead><tr><th>阶段</th><th>边界 RGB 跳变</th><th>相对原图的边界梯度误差</th></tr></thead><tbody>__EDGE_TABLE__</tbody></table><p>49 帧加权边界跳变由原生 <b>__NATIVE_JUMP__</b> 增至道路插入后的 <b>__OFF_JUMP__</b>；已有颜色校正降至 <b>__FINE_JUMP__</b>。完整车辆替换前后该边界统计相同。</p><p>代码路径解释了这个现象：道路颜色从跨时刻道路观测确定，1K 阶段冻结道路颜色与几何，仅优化旧静态高斯的透明度门控。插入道路后，原静态贡献和透出的方向背景同时减少；固定道路颜色没有与边界外原生成像共同拟合。后续 11×9 世界颜色场只调道路颜色，现有幅度与正则下减轻了部分差异，接缝仍然存在。</p><p class="callout">已定位的是“原生局部颜色误差”与“道路插入后新增的边界误差”两层问题。本轮没有训练修复，也没有验证某个新修复方案的收益。</p></div>
<div class="card"><h2>验证、执行范围与复现</h2><p>全部 49 帧保留状态，另含三帧删除状态；pilot、完整核对和分层参考替换分别保存。所有颜色指标使用未裁剪的浮点渲染值。交互预览压缩为 JPEG，raw113 保留无损 PNG。固定画面框用于诊断，不代表跨帧语义跟踪。</p><p>分层 RGB 重建最大误差 <code>__PARITY__</code>；与原生参考入口最大差 <code>__NATIVE_PARITY__</code>。检查点文件、冻结张量版本、道路状态和车辆资产几何保持不变。完整 49 帧诊断耗时 __SECONDS__ 秒，独立 pilot __PILOT_SECONDS__ 秒；训练更新为 0。GPU 与实例保持开启。</p><p><a href="findings.json">摘要数字</a> · <a href="evidence/full49_metrics.json">完整浮点指标</a> · <a href="evidence/reference_split_metrics.json">分层参考替换指标</a> · <a href="evidence/inventory.json">远端现场和来源清单</a> · <a href="code/diagnose.py">诊断代码</a> · <a href="code/reference_split.py">分层代码</a></p><details><summary>复现命令及证据边界</summary><pre>cd /root/autodl-tmp/drivingeditor_repro/campaign/scene008_local_colour_diag_20261105_r1
/root/autodl-tmp/envs/drivingeditor/bin/python source/diagnose.py --pilot --output pilot_new
/root/autodl-tmp/envs/drivingeditor/bin/python source/diagnose.py --output full49_new

保存的原始浮点数组：full49/*_exact.npz、reference_split/*_exact.npz
所有输出均写入新目录。没有改动父检查点或运行训练。
failure_ledger_delta: none（本轮细化已有颜色/合成现象，不新增方法验收失败）。</pre><p>原生模型：2026-11-05 native_ownership run_r3/formal，带持久对象标签。道路父状态：native_road1k_20261105_r1/formal/checkpoint001000.pt。完整来源见 inventory。</p></details></div></main>
<script>
const DATA=__DATA__,STAGES=__STAGES__,BOXES=__BOXES__;
const frame=document.getElementById('frame'), slider=document.getElementById('slider'),state=document.getElementById('state'),roi=document.getElementById('roi');
const keys=Object.keys(STAGES),gallery=document.getElementById('gallery');
for(const key of keys){const tile=document.createElement('div');tile.className='tile';const title=document.createElement('h3');title.textContent=STAGES[key];const c=document.createElement('canvas');c.id='canvas_'+key;c.width=480;c.height=320;tile.append(title,c);gallery.append(tile);}
let token=0;
function frameOptions(){const old=Number(frame.value)||113;const vals=state.value==='keep'?Array.from({length:49},(_,i)=>89+i):[105,113,130];frame.replaceChildren(...vals.map(n=>{const o=document.createElement('option');o.value=n;o.textContent='raw'+n;return o;}));frame.value=vals.includes(old)?old:113;slider.disabled=state.value==='delete';slider.value=frame.value;}
async function update(){const ticket=++token,n=Number(frame.value),s=state.value,box=BOXES[roi.value]||[0,0,480,320];slider.value=n;document.getElementById('load').textContent='载入…';const record=DATA.find(r=>r.raw===n&&r.state===s);const selected=roi.value==='full'?'non_actor':roi.value;document.getElementById('numbers').innerHTML=keys.map(k=>{const a=record.metrics[k][selected];return `<tr><td>${STAGES[k]}</td><td>${a.blue_error.toFixed(5)}</td><td>${a.mae.toFixed(5)}</td><td>${(100*a.excess_blue_fraction).toFixed(1)}%</td></tr>`}).join('');await Promise.all(keys.map(key=>new Promise(resolve=>{const im=new Image();im.onload=()=>{if(ticket===token){const c=document.getElementById('canvas_'+key),ctx=c.getContext('2d');c.width=480;c.height=Math.round(480*(box[3]-box[1])/(box[2]-box[0]));ctx.imageSmoothingEnabled=false;ctx.drawImage(im,box[0],box[1],box[2]-box[0],box[3]-box[1],0,0,c.width,c.height);if(roi.value==='full'){for(const [name,col] of [['left_road_patch','#ed734d'],['right_curb_patch','#13a7a6']]){const b=BOXES[name];ctx.strokeStyle=col;ctx.lineWidth=2;ctx.strokeRect(b[0]+1,b[1]+1,b[2]-b[0]-2,b[3]-b[1]-2)}}}resolve()};im.onerror=()=>{if(ticket===token)document.getElementById('load').textContent='图像缺失';resolve()};im.src=`frames/${String(n).padStart(3,'0')}_${s}_${key}.jpg`;})));if(ticket===token){document.getElementById('load').textContent=`raw${n}`;document.getElementById('caption').textContent=roi.value==='full'?'颜色表格统计完整画面的非车辆像素；局部框在所有阶段保持一致。':'局部框固定于画面坐标，颜色表格排除已知车辆像素。';document.getElementById('lossless').href=`evidence/lossless_pilot113/113_${s}_native.png`;}}
frameOptions();frame.value='113';frame.onchange=update;slider.oninput=()=>{frame.value=slider.value;update()};roi.onchange=update;state.onchange=()=>{frameOptions();update()};update();
</script></html>'''
replace={'__TITLE__':title,'__BLUE_TABLE__':blue_table,'__EDGE_TABLE__':edge_table,
         '__COMPONENT_TABLE__':component_table,'__LEFT_BLUE__':f'{pooled("native","left_road_patch","blue_error"):+.5f}',
         '__RIGHT_BLUE__':f'{pooled("native","right_curb_patch","blue_error"):+.5f}',
         '__NATIVE_JUMP__':f'{nativejump:.5f}','__OFF_JUMP__':f'{offjump:.5f}','__FINE_JUMP__':f'{finejump:.5f}',
         '__PARITY__':f'{findings["attribution_maxabs"]:.2g}','__NATIVE_PARITY__':f'{findings["native_binding_maxabs"]:.2g}',
         '__SECONDS__':f'{data["seconds"]:.1f}','__PILOT_SECONDS__':f'{findings["pilot_seconds"]:.1f}',
         '__DATA__':json.dumps(compact,ensure_ascii=False,separators=(',',':')),
         '__STAGES__':json.dumps(stages,ensure_ascii=False),'__BOXES__':json.dumps(data['roi_boxes'])}
for key,value in replace.items():
    page=page.replace(key,value)
assert '__TITLE__' not in page and '__DATA__' not in page
(OUT/'index.html').write_text(page,encoding='utf-8')
readme=f'''# {title}

局部偏蓝确实超出原图；道路插入进一步增加颜色边界。49 帧同相机对照，另核对三帧删除状态，0 次训练更新。

- 左侧道路新增蓝色残差 {pooled('native','left_road_patch','blue_error'):+.5f}，49/49 帧为正。
- 右侧路缘新增蓝色残差 {pooled('native','right_curb_patch','blue_error'):+.5f}，49/49 帧为正。
- 道路边界 RGB 跳变：原生 {nativejump:.5f} → 固定道路插入 {offjump:.5f} → 既有颜色校正 {finejump:.5f}。
- 三帧的覆盖权重参考替换指向方向背景蓝色残差与静态层补偿关系；这不等于物理照明估计或训练时序归因。
- 原生与道路状态保持冻结。AutoDL 保持开机；本对话不授权 subagent 或自动关机。

![Architecture components](architecture.svg)

[完整可切帧报告](index.html) · [局部对照](local_comparison.png) · [指标](evidence/full49_metrics.json)

远端现场：`/root/autodl-tmp/drivingeditor_repro/campaign/scene008_local_colour_diag_20261105_r1`。
正式诊断约 {data['seconds']:.1f} 秒，pilot {findings['pilot_seconds']:.1f} 秒，额外分层诊断另存。所有工作均无训练更新。

failure_ledger_refs: existing color/compositing evidence; no new scientific-failure claim.
failure_ledger_delta: none.
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
print(json.dumps(findings,ensure_ascii=False,indent=2))

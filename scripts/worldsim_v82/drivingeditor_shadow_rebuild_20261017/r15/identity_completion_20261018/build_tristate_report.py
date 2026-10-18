"""Build a self-contained center viewer plus inspectable supervision/raster panels."""
from pathlib import Path
import argparse,base64,json,shutil
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path(__file__).parent/'tristate')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'outputs/identity_comparison_20261018/tristate')
    args=p.parse_args();src=args.input;out=args.output;out.mkdir(parents=True,exist_ok=True)
    read=lambda name:json.loads((src/name).read_text(encoding='utf-8'))
    audit=read('evidence_audit.json');data=read('viewer_data.json');masks=read('mask_comparison/comparison.json')
    boundary=read('boundary_origins.json')
    old5=(src/'old5_geometry/viewer_panels.json').is_file()
    if old5:
        data['panels'].extend(read('old5_geometry/viewer_panels.json')['panels'])
    for panel in data['panels']:
        obj=panel['points'];a=np.frombuffer(base64.b64decode(obj['data']),dtype='<f4').reshape(-1,12)
        limit=6000 if panel['scope']=='all' else 4000 if panel['scope'].endswith('context') else 8000
        if len(a)>limit:a=a[np.linspace(0,len(a)-1,limit).astype(int)]
        panel['points']=dict(rows=len(a),stride=12,data=base64.b64encode(a.tobytes()).decode())
    for folder in ['pixels','mask_comparison','projection','old5_geometry']:
        if (src/folder).exists():shutil.copytree(src/folder,out/folder,dirs_exist_ok=True)
    for name in ['evidence_audit.json','boundary_origins.json','cpu_complete.json']:
        if (src/name).exists():shutil.copy2(src/name,out/name)
    projection=(src/'projection/projection_receipt.json').exists()
    table=''
    for actor in [6,37,5,26,18]:
        a,b=[next(r for r in audit['actors'] if r['actor']==actor and r['stage']==s) for s in ['init','4k']]
        c=next(r for r in boundary['records'] if r['actor']==actor)
        table+=f"<tr><td>a{actor}</td><td>{a['count']:,} → {b['count']:,}</td><td>全部 1 → 全部 1</td><td>{a['centers_outside_box']:,} → {b['centers_outside_box']:,}</td><td>{c['outside_from_initial_inside_parent']:,} / {c['current_outside']:,}</td><td>{b['distance_to_original_seed_m']['p95']:.3f} m</td></tr>"
    masktable=''
    for actor in [6,37]:
        d=masks['summary'][str(actor)];f=d['native_positive_R15_domain_fraction'];t=d['totals']
        masktable+=f"<tr><td>a{actor}</td><td>{t['native_positive']:,}</td><td>{f['positive']:.2%}</td><td>{f['negative']:.2%}</td><td>{f['unknown']:.2%}</td></tr>"
    arch='''<svg viewBox="0 0 1080 230" role="img" aria-label="R15证据、几何和投影检查数据流"><defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#8ca7c2"/></marker></defs><g fill="#17304a" stroke="#6684a3"><rect x="10" y="40" width="150" height="75" rx="8"/><rect x="205" y="40" width="180" height="75" rx="8"/><rect x="430" y="40" width="180" height="75" rx="8"/><rect x="660" y="40" width="180" height="75" rx="8"/><rect x="890" y="40" width="180" height="75" rx="8"/><rect x="430" y="158" width="180" height="55" rx="8"/></g><g stroke="#8ca7c2" stroke-width="2" fill="none" marker-end="url(#arr)"><path d="M160 78H205M385 78H430M610 78H660M840 78H890M520 158V115"/></g><g fill="#eef5ff" font-family="sans-serif" font-size="16" text-anchor="middle"><text x="85" y="67">测量与图像</text><text x="85" y="92">LiDAR / SAM</text><text x="295" y="67">像素三值监督</text><text x="295" y="92">0 / 1 / unknown</text><text x="520" y="67">初始 Gaussian</text><text x="520" y="92">证据状态 + 硬归属</text><text x="750" y="67">几何训练 / 增密</text><text x="750" y="92">身份 buffer 随谱系复制</text><text x="980" y="67">冻结 4K 审计</text><text x="980" y="92">中心 / 支撑 / alpha*T</text><text x="520" y="180">池外待定测量</text><text x="520" y="201">周期审计补入 459 点</text></g></svg>'''
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R15：0 / 1 / unknown 逐层排查</title>
<style>:root{color-scheme:dark}*{box-sizing:border-box}body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#0c1521;color:#dfeaf5;margin:0 auto;padding:28px;max-width:1450px}h1{font-size:30px;line-height:1.3}h2{font-size:21px;margin-top:34px}p{max-width:1100px}a{color:#8acaff}button,select{background:#1c3047;border:1px solid #536c85;color:#e9f4ff;border-radius:6px;padding:7px;margin:3px;font:inherit}button{cursor:pointer}.note,.card{background:#132439;border:1px solid #314963;border-radius:9px;padding:15px;margin:16px 0}.blue{color:#38a5f0}.green{color:#26e68c}.orange{color:#ffb627}.tools{display:flex;flex-wrap:wrap;align-items:center;gap:10px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.view{background:#09111c;border:1px solid #314963;border-radius:8px;overflow:hidden}canvas{display:block;width:100%;height:370px;touch-action:none}.caption{padding:10px;font-size:14px;min-height:65px}.details{white-space:pre-wrap;font:13px/1.6 ui-monospace,monospace}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #314963}th{background:#18314a}figure{margin:0}img{width:100%;display:block;image-rendering:auto}.figures{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.figures figure{border:1px solid #314963;background:#09111c}.figures figcaption{padding:8px;font-size:13px}.small{font-size:14px;color:#a5bdd3}svg{width:100%;height:auto}@media(max-width:800px){body{padding:15px}.grid,.figures{grid-template-columns:1fr}canvas{height:310px}table{font-size:12px}}</style>
<h1>R15：哪些是 0、1、unknown，问题在哪一层？</h1>
<p>旧版二值作为完整身份监督的对照继续保留。这里将<strong>像素监督、Gaussian 证据状态、硬归属与几何</strong>并列检查；它们不是同一套标签。</p>
<div class="note"><strong>目前已确认：</strong>a6、a37（以及 a5、a26、a18）的内部 Gaussian，初始化与 4K 都是“确定车辆 1”，没有内部 unknown。对齐谱系后的 2,067,112 个现存后代，证据状态、身份概率和硬归属全部保持原值。状态与概率是 buffer，普通 Adam 反传没有学习它们的通路；不能把这个结果称为“unknown 尝试学归属但没学会”。</div>
ARCH
<h2>先看清三个含义</h2>
<table><tr><th>层级</th><th>0</th><th>1</th><th>unknown</th><th>如何变化</th></tr><tr><td>像素监督（按身份）</td><td>有效负像素</td><td>有效车身正像素</td><td>无效／不监督像素</td><td>预处理掩码融合，当前审计不改输入</td></tr><tr><td>Gaussian 证据状态（本页颜色）</td><td>确定 static（原始码 2）</td><td>确定 Actor（原始码 1）</td><td>证据待定（原始码 0）</td><td>增密复制 buffer；显式周期审计才更改</td></tr><tr><td>硬归属 owner</td><td colspan="3">owner=-1 为静态；owner=0…42 为具体 Actor。unknown 证据的 Gaussian 也已经有硬 owner，仍参与渲染。</td><td>当前可追溯后代的 owner 转移为 0</td></tr></table>
<p class="small">“确定 Actor”只表示当前证据类别，不是经过人工验证的车辆几何真值；也不意味着对所有 Actor 都是正例。原始 status=0 是 unknown，不能直接解释成二值背景 0。未入场测量池单独显示，池中点还不是 Gaussian。</p>
<h2>交互查看初始化与 4K 的 Gaussian 中心</h2>
<div class="tools"><label>对象 <select id="scope"><option value="a6">a6</option><option value="a37">a37</option><option value="a5">a5</option><option value="a26">a26</option><option value="a18">a18</option><option value="all">全场景</option></select></label><label><input id="context" type="checkbox">显示周围其他归属</label><label><input id="pool" type="checkbox">外部待定测量池（仅 a6/a37）</label></div>
<div class="tools"><label><input id="targetmode" type="checkbox" checked>按当前身份显示：其他确定 owner 为 0</label><label class="blue"><input data-state="2" type="checkbox" checked>0：确定非目标／static</label><label class="green"><input data-state="1" type="checkbox" checked>1：目标／Actor</label><label class="orange"><input data-state="0" type="checkbox" checked>U：待定证据</label><button id="persp">斜视</button><button id="top">俯视</button><button id="side">侧视</button><button id="reset">重置</button></div>
<p class="small">默认按当前目标派生 0／1／U：已知非目标 owner 为 0，已知目标 owner 为 1，原待定证据仍为 U。取消该选项或查看全场景时，恢复全局证据类别 0=static／1=Actor／U=unknown。两种模式都只读已有状态，不把它作为正确归属真值。</p>
<p class="small">拖动旋转，滚轮缩放，Shift+拖动平移，点击点查看状态、硬归属、概率、来源和尺度。两侧视角同步。点是固定均匀抽样的中心，所有数量来自全量统计；线框是 Actor 的诊断框。切换“周围”包含其他 owner，不能都算成该车辆。全场景将各车变换到 raw113 的世界坐标。</p>
<div class="grid"><div class="view"><canvas id="left"></canvas><div id="leftcap" class="caption"></div></div><div class="view"><canvas id="right"></canvas><div id="rightcap" class="caption"></div></div></div>
<div class="card details" id="details">点击任意中心查看数据。此视图不表现不透明度、完整椭球和遮挡；实际支撑由下面的相机光栅图检查。</div>
<h2>几何变化：已有 1 也能在训练后产生跨界后代</h2>
<table><tr><th>身份</th><th>初始 → 4K 点数</th><th>证据状态</th><th>中心在框外</th><th>4K 框外中：来自初始框内父点</th><th>后代到原始父点位移 p95</th></tr>GEOMETRY</table>
<p>a6 的 29,834 个框外中心中，25,905 个来自初始框内父点；a37 为 13,681 / 14,573。这个结果要求继续检查增密与位置优化；不能仅凭最终状态是 1 就把碎片归因于初始化。初始框内也可能有错误测量，框外也可能是合法边缘；这张表不是道路污染的真值统计。位移包含多代分裂和位置更新。</p>
<h2>输入差异：原生 SAM 正区域落在 R15 的哪个域？</h2>
<table><tr><th>身份</th><th>49 帧原生正像素总数</th><th>R15 正域 1</th><th>R15 负域 0</th><th>R15 unknown 域</th></tr>MASKTABLE</table>
<p>a6 的原生 SAM 正区域有 39.04% 被 R15 融合列入 unknown，a37 为 0.25%；两者落入 R15 已知负域的比例都是 0。这定位了<strong>a6 的监督覆盖差异</strong>，没有证明原生 SAM 边界就是正确答案，也没有证明这部分二维区域应全部成为车辆 Gaussian。</p>
<h2>逐帧对照：像素域与实际 Gaussian 支撑</h2>
<div class="tools"><label>身份 <select id="actor"><option>6</option><option>37</option><option>5</option><option>26</option><option>18</option></select></label><label>原始帧 <select id="frame"></select></label></div>
<div id="images" class="figures"></div><p id="pixelstats" class="small"></p>
<p class="small">像素域图按 0 蓝／1 绿／unknown 橙着色。原生 SAM 轮廓为白线。Gaussian 图使用真实协方差、不透明度和相机光栅，颜色来自每点证据类别；全景保留 alpha*T 遮挡，独立渲染去掉其他 owner 的遮挡。独立 alpha 出现在可见车身掩码外，可能包含合法被遮挡车体，不能直接当错归属。</p>
<h2>谱系核对与测量池</h2>
<p>初始 771,821 个 Gaussian → 4K 2,068,305 个。原 unknown 的 122,159 个种子仍有后代，共 463,520 点，全部仍是 unknown；1,325 个初始 unknown 没有现存后代。新增谱系 268 个父点有 1,193 个现存后代，均为状态 1；这是 3K 补入测量后的增密结果，不能算作原 unknown Gaussian 学会转为车辆。</p>
<p>外部候选池初始 128,654 条测量，3K 后保留 128,195 条，差额 459 条被审计补入。池内测量未参与当前渲染，不能与已有 unknown Gaussian 混为一谈。</p>
<p><a href="evidence_audit.json">全量状态、谱系、优化器与逐帧计数</a> · <a href="boundary_origins.json">框外后代来源</a> · <a href="mask_comparison/comparison.json">原生与融合掩码域交叉表</a> PROJECTIONLINK</p>
<p class="small">原始种子、R15 4K 检查点和监督掩码均未修改。无参数更新，无新训练阶段，无自动归属修复。工程诊断，human_verdict=null；没有配对编辑真值，也不宣称碎片或残影已解决。</p>
<script id="data" type="application/json">@@VIEWER@@</script><script id="audit" type="application/json">@@AUDIT@@</script><script id="inputmasks" type="application/json">@@INPUTMASKS@@</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent), AUDIT=JSON.parse(document.getElementById('audit').textContent), MASK=JSON.parse(document.getElementById('inputmasks').textContent), HAS_PROJECTION=HASPROJECTION;
const colors={0:'#ffb627',1:'#26e68c',2:'#38a5f0'};const scopes=document.getElementById('scope'),context=document.getElementById('context'),pool=document.getElementById('pool'),targetmode=document.getElementById('targetmode');
let yaw=-.65,pitch=.48,zoom=1,panx=0,pany=0,radius=3,center=[0,0,0],chosen=[];
function decode(p){if(!p.array){let b=atob(p.points.data),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);p.array=new Float32Array(u.buffer)}return p}
function selectPanels(){let scope=scopes.value;if(pool.checked&&!['a6','a37'].includes(scope)){pool.checked=false}let stages=pool.checked?['pool_init','pool_3k']:['init','4k'];let key=scope+(context.checked&&!pool.checked&&scope!=='all'?'context':'');chosen=stages.map(stage=>DATA.panels.find(p=>p.stage===stage&&p.scope===key));chosen.forEach(p=>p&&decode(p));if(scope==='all'){let vals=[[],[],[]];chosen.forEach(p=>{if(p)for(let i=0;i<p.array.length;i+=12)for(let j=0;j<3;j++)vals[j].push(p.array[i+j])});let lo=[],hi=[];for(let j=0;j<3;j++){vals[j].sort((a,b)=>a-b);lo[j]=vals[j][Math.floor(vals[j].length*.01)];hi[j]=vals[j][Math.floor(vals[j].length*.99)]}center=lo.map((v,j)=>(v+hi[j])/2);radius=Math.max(...hi.map((v,j)=>v-lo[j]))*.55}else{center=[0,0,0];radius=context.checked||pool.checked?9:3.3}drawAll()}
function transform(x,y,z,w,h){x-=center[0];y-=center[1];z-=center[2];let a=x*Math.cos(yaw)-y*Math.sin(yaw),b=x*Math.sin(yaw)+y*Math.cos(yaw),q=b*Math.cos(pitch)-z*Math.sin(pitch),r=b*Math.sin(pitch)+z*Math.cos(pitch);let f=Math.min(w,h)*.43/radius*zoom;return [w/2+a*f+panx,h/2-r*f+pany,q]}
let projected=[[],[]];function draw(which){const can=document.getElementById(which===0?'left':'right'),dpr=Math.min(devicePixelRatio||1,2),w=can.clientWidth,h=can.clientHeight;can.width=w*dpr;can.height=h*dpr;const ctx=can.getContext('2d');ctx.scale(dpr,dpr);ctx.fillStyle='#09111c';ctx.fillRect(0,0,w,h);let p=chosen[which];projected[which]=[];if(!p)return;const enabled=new Set([...document.querySelectorAll('[data-state]:checked')].map(x=>Number(x.dataset.state)));let a=p.array,draw=[];for(let i=0;i<a.length;i+=12){let s=a[i+3];if(targetmode.checked&&scopes.value!=='all'&&s===1&&a[i+4]!==Number(scopes.value.slice(1)))s=2;if(!enabled.has(s))continue;let t=transform(a[i],a[i+1],a[i+2],w,h);if(t[0]<0||t[0]>w||t[1]<0||t[1]>h)continue;draw.push([t[0],t[1],t[2],i,s])}draw.sort((a,b)=>b[2]-a[2]);ctx.globalAlpha=.83;for(const r of draw){ctx.fillStyle=colors[r[4]];ctx.fillRect(r[0]-1,r[1]-1,2,2);projected[which].push(r)}ctx.globalAlpha=1;
if(p.half_box_m){let hs=p.half_box_m,corners=[];for(let i=0;i<8;i++)corners.push(transform((i&1?1:-1)*hs[0],(i&2?1:-1)*hs[1],(i&4?1:-1)*hs[2],w,h));ctx.strokeStyle='#d1dbebaa';ctx.lineWidth=1;for(let i=0;i<8;i++)for(let bit of [1,2,4])if(!(i&bit)){ctx.beginPath();ctx.moveTo(...corners[i].slice(0,2));ctx.lineTo(...corners[i|bit].slice(0,2));ctx.stroke()}}
const origin=transform(0,0,0,w,h);['X','Y','Z'].forEach((v,i)=>{let point=[0,0,0];point[i]=1;let t=transform(...point,w,h);ctx.strokeStyle=['#f68a8a','#9ceb9c','#8faaff'][i];ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.moveTo(...origin.slice(0,2));ctx.lineTo(...t.slice(0,2));ctx.stroke();ctx.fillText(v,t[0]+4,t[1])});document.getElementById(which===0?'leftcap':'rightcap').textContent=`${p.stage} · ${p.coordinates} · 全量 ${p.total.toLocaleString()} · 显示抽样 ${p.points.rows.toLocaleString()} · 状态原码计数 ${JSON.stringify(p.counts)}`}
function drawAll(){draw(0);draw(1)}for(let el of [scopes,context,pool])el.addEventListener('change',()=>{zoom=1;panx=pany=0;selectPanels()});targetmode.addEventListener('change',drawAll);document.querySelectorAll('[data-state]').forEach(el=>el.addEventListener('change',drawAll));
for(let [id,y,p] of [['top',0,Math.PI/2],['side',0,0],['persp',-.65,.48]])document.getElementById(id).onclick=()=>{yaw=y;pitch=p;drawAll()};document.getElementById('reset').onclick=()=>{yaw=-.65;pitch=.48;zoom=1;panx=pany=0;selectPanels()};
['left','right'].forEach((id,k)=>{const c=document.getElementById(id);let drag=null;c.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,startx:e.clientX,starty:e.clientY};c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(!drag)return;let dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(e.shiftKey){panx+=dx;pany+=dy}else{yaw+=dx*.009;pitch=Math.max(-1.57,Math.min(1.57,pitch+dy*.009))}drag.x=e.clientX;drag.y=e.clientY;drawAll()};c.onpointerup=e=>{if(drag&&Math.hypot(e.clientX-drag.startx,e.clientY-drag.starty)<4){let r=c.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,near=null,best=64;for(const p of projected[k]){let d=(p[0]-x)**2+(p[1]-y)**2;if(d<best){best=d;near=p}}if(near){let p=chosen[k],row=Array.from(p.array.slice(near[3],near[3]+12));document.getElementById('details').textContent=`${p.stage} / ${p.scope}：${JSON.stringify(Object.fromEntries(DATA.columns.map((s,i)=>[s,row[i]])),null,2)}\n硬owner与证据状态独立；source_frame_index 为49帧中的下标；概率顺序 static / Actor / unknown。`} }drag=null};c.onwheel=e=>{e.preventDefault();zoom=Math.max(.1,Math.min(15,zoom*Math.exp(-e.deltaY*.001)));drawAll()}});window.addEventListener('resize',drawAll);
const actor=document.getElementById('actor'),frame=document.getElementById('frame');for(let raw=89;raw<=137;raw++){let o=document.createElement('option');o.value='0008'+String(raw).padStart(3,'0');o.textContent=raw;frame.append(o)}frame.value='0008113';function updateImages(){let a=actor.value,id=frame.value,aa='a'+a.padStart(2,'0'),items=[['pixels/'+id+'_'+aa+'.png','R15 原像素监督：0 / 1 / unknown']];if(['6','37'].includes(a))items.push(['mask_comparison/'+id+'_'+aa+'.png','R15 像素域叠加图像 + 白色原生 SAM 轮廓']);if(HAS_PROJECTION)for(let s of ['init','4k']){items.push([`projection/${s}/${id}/${aa}_state.png`,`${s} · 该身份独立 Gaussian 证据状态`]);items.push([`projection/${s}/${id}/scene_state.png`,`${s} · 全景证据状态，保留遮挡`]);items.push([`projection/${s}/${id}/${aa}_visible.png`,`${s} · 该身份全景可见 alpha*T`])}document.getElementById('images').innerHTML=items.map(([url,label])=>`<figure><a href="${url}"><img src="${url}" alt="${label}" loading="lazy"></a><figcaption>${label}</figcaption></figure>`).join('');let r=AUDIT.pixel_masks.find(x=>x.actor===Number(a)&&x.frame_id===id),d=MASK.records.find(x=>x.actor===Number(a)&&x.frame_id===id);document.getElementById('pixelstats').textContent=`该帧像素：0=${r.negative}，1=${r.positive}，unknown=${r.unknown}。`+(d?` 原生正区域 ${d.native_positive}：R15 正域 ${d.native_positive_in_R15_positive}，负域 ${d.native_positive_in_R15_negative}，unknown ${d.native_positive_in_R15_unknown}。`:'')+(HAS_PROJECTION?'':' 实际 Gaussian 支撑的 GPU 投影已排在旧基线训练及渲染之后；本页先交付 CPU 状态与输入审计。')};actor.onchange=frame.onchange=updateImages;selectPanels();updateImages();
</script></html>'''
    html=html.replace("frame.value='0008113';function updateImages()","frame.value='0008089';function updateImages()")
    html=html.replace("items=[['pixels/'+id+'_'+aa+'.png','R15 原像素监督：0 / 1 / unknown']]",
        "items=['6','37'].includes(a)?[['mask_comparison/'+id+'_'+aa+'_rgb_crop.png','原图：共同局部窗口'],['mask_comparison/'+id+'_'+aa+'_state_crop.png','同窗口 R15 三值监督'],['mask_comparison/'+id+'_'+aa+'_native_crop.png','同窗口原生 SAM 二值提议']]:[];items.push(['pixels/'+id+'_'+aa+'.png','R15 原像素监督：0 / 1 / unknown'])")
    if old5:
        html=html.replace('<h2>交互查看初始化与 4K 的 Gaussian 中心</h2>',
            '<h2>交互查看旧5基线与 R15 的 Gaussian 中心</h2><p><a href="../index.html">返回完整49帧独立／删除对照</a> · <a href="old5_geometry/geometry.json">旧5全量几何与归属映射</a>。新增旧5的归一化坐标已按实际比例还原为物理米，并使用同一真实车框。<strong>旧版没有三值证据或概率；旧5的绿色仅表示属于当前硬 owner，点击时概率 -1 表示不存在，不能解释成确定车辆证据。</strong></p><div class="tools"><label>左右对比 <select id="pair"><option value="old5" selected>旧5最终30K / 原R15 4K</option><option value="r15">R15初始化 / 原R15 4K</option></select></label></div>')
        html=html.replace("let yaw=-.65", "const pair=document.getElementById('pair');pair.onchange=()=>{zoom=1;panx=pany=0;selectPanels()};let yaw=-.65")
        html=html.replace("let scope=scopes.value;if(pool.checked", "let scope=scopes.value;let oldPair=pair.value==='old5';context.disabled=pool.disabled=oldPair;scopes.querySelector('option[value=all]').disabled=oldPair;if(oldPair){context.checked=pool.checked=false;if(scope==='all'){scopes.value=scope='a6'}}if(pool.checked")
        html=html.replace("let stages=pool.checked?['pool_init','pool_3k']:['init','4k'];", "let stages=oldPair?['old5','4k']:pool.checked?['pool_init','pool_3k']:['init','4k'];")
        html=html.replace('状态原码计数 ${JSON.stringify(p.counts)}','${p.stage===\'old5\'?\'硬owner计数（无三值证据）\':\'状态原码计数\'} ${JSON.stringify(p.counts)}')
        html=html.replace('概率顺序 static / Actor / unknown。', '概率顺序 static / Actor / unknown。旧5左图的绿色/状态1仅是owner显示，概率-1表示旧模型不存在此参数。')
        html=html.replace('全量状态、谱系、优化器与逐帧计数</a>', '原R15全量状态、谱系、优化器与逐帧计数</a>')
    if old5:
        html=html.replace('无参数更新，无新训练阶段，无自动归属修复。工程诊断，human_verdict=null；', '本页仅只读旧5与原R15检查点，没有修改这些模型。新修复4K已另行启动，见<a href="../../r15_repair_4k_20261018/index.html">新实验状态</a>。工程诊断，human_verdict=null；')
    replaces={'ARCH':arch,'GEOMETRY':table,'MASKTABLE':masktable,'@@VIEWER@@':json.dumps(data,separators=(',',':')),
        '@@AUDIT@@':json.dumps(audit,separators=(',',':')),'@@INPUTMASKS@@':json.dumps(masks,separators=(',',':')),
        'HASPROJECTION':'true' if projection else 'false',
        'PROJECTIONLINK':'· <a href="projection/projection_receipt.json">实际支撑投影与域内贡献</a>' if projection else ''}
    # Replace longer keys first; token substitution must not replace inside inserted JSON.
    import re
    html=re.sub('|'.join(map(re.escape,sorted(replaces,key=len,reverse=True))),lambda m:replaces[m.group()],html)
    html=html.replace('<h1>','<p class="note"><a href="../index.html"><strong>主要检查入口：五身份动态／静态／Delete 同步视频，可播放及逐帧拖动</strong></a>。本页保留为三维证据补充。</p><h1>',1)
    (out/'index.html').write_text(html,encoding='utf-8')
    (out/'build_status.json').write_text(json.dumps(dict(cpu_audit_complete=True,GPU_projection_complete=projection,report_bytes=len(html.encode('utf-8'))),indent=2),encoding='utf-8')
    print(out/'index.html')

if __name__=='__main__':main()

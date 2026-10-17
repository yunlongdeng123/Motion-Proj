from pathlib import Path
import json,shutil,datetime

W=Path(__file__).resolve().parent
ROOT=W.parents[2]
O=ROOT/'outputs/shadow_rebuild_v3/r14/mask_review'
receipt=json.loads((O/'verification.json').read_text(encoding='utf8'))
frames=receipt['frames']
template=r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R14 完整 mask 与 LiDAR 核查</title>
<style>
:root{color-scheme:dark;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif;background:#0b1420;color:#e2e9f3}*{box-sizing:border-box}body{max-width:1500px;margin:auto;padding:24px}h1{font-size:28px;margin:0 0 10px}h2{font-size:21px;margin:30px 0 12px}p{max-width:1100px;color:#b9c8da}a{color:#80beff}code{background:#1b2a3f;padding:2px 5px;border-radius:4px}button,select{font:inherit;padding:6px 12px;background:#1b2a3f;color:white;border:1px solid #49647d;border-radius:6px;cursor:pointer}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px;position:sticky;top:0;z-index:3;background:#0b1420f5;padding:12px;border:1px solid #34465b;border-radius:8px}input[type=range]{width:min(380px,40vw)}.grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:16px}.card{background:#111e2e;border:1px solid #2d4158;border-radius:8px;overflow:hidden}.card h3{font-size:16px;margin:0;padding:9px 12px}.card img{display:block;width:100%;height:auto;image-rendering:pixelated;background:black}.card a{display:block;padding:4px 12px;font-size:13px}.wide{grid-column:1/-1}.legend{display:flex;gap:16px;flex-wrap:wrap;padding:12px 0}.dot{display:inline-block;width:12px;height:12px;margin-right:5px;border-radius:3px}.badge{background:#193e37;color:#a1edcd;border:1px solid #377a67;border-radius:5px;padding:4px 8px}.unknown{color:#ff69d4}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;text-align:right;border-bottom:1px solid #31445c}td:first-child,th:first-child{text-align:left}.flow{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:20px 0;padding:20px;background:#111e2e;border-radius:8px}.box{padding:12px 20px;border:1px solid #6382a3;border-radius:7px;text-align:center}.small{font-size:14px;color:#9db1c9}.pixelgrid{display:grid;grid-template-columns:repeat(3,minmax(230px,1fr));gap:12px}#zoomcard{margin-top:16px}#zoomcard img{width:auto;max-width:100%;margin:auto}.hash{word-break:break-all}details{margin:20px 0}summary{cursor:pointer}@media(max-width:800px){.grid,.pixelgrid{grid-template-columns:1fr}body{padding:12px}}
</style></head><body>
<h1>完整 mask 与真实 LiDAR 核查</h1>
<p>已修复只画上沿的问题。现在显示<strong>整块半透明 mask、完整内外轮廓及原尺寸二值图</strong>，覆盖下沿、左右边界、孔洞和画面截断处。以下读取已冻结的 R14 输入；未重新推理、融合或修改监督，训练继续运行。</p>
<div class="flow"><div class="box">冻结 RGB / DEVA / SAM3<br>逐像素 mask_valid</div><span>→</span><div class="box">只读完整 mask 绘图<br>真实 LiDAR 投影对照</div><span>→</span><div class="box">轮廓＋填色 / 原始二值 mask<br>逐车放大 / 49 帧切换</div></div>
<p class="small">本页检查的是三维训练所用的<strong>二维 mask 监督及其 LiDAR 对齐</strong>。它不直接证明三维高斯归属正确；实际归属还需结合独立资产、逐高斯贡献与编辑结果。彩色 LiDAR 点表示三维框候选身份，不代表已确认的车体表面。</p>
<div class="controls"><button id="prev">上一帧</button><input id="frame" type="range" min="0" max="48" value="18"><button id="next">下一帧</button><button id="play">播放 49 帧</button><select id="actor"></select><span class="badge" id="status"></span></div>
<div class="legend" id="legend"></div>
<p><span class="unknown">洋红色：未知区域，不施加该实例的正/负 mask 标签。</span>背景上未覆盖颜色的区域不一定都是有效负标签；请以下方该实例的 <code>valid</code> 和三值监督图为准。</p>
<h2>同帧完整对照</h2><div class="grid" id="overview"></div>
<h2 id="actorhead">逐 Actor 原尺寸监督与车底细节</h2>
<p>三值图：<strong>白 = 有效车体；黑 = 有效实例外；灰 = 未知</strong>。Positive 图只记录原始正标签，必须与 Valid 图共同使用。点击图片可打开原尺寸图。</p>
<div class="pixelgrid" id="binary"></div>
<div class="card" id="zoomcard"><h3>同一裁剪区域：RGB / DEVA / SAM3 / Fused / 三值监督 / 真实 LiDAR</h3><a id="zoomlink"><img id="zoom" alt="逐车 mask 与车底放大"></a></div>
<p class="small" id="crop"></p>
<h2>当前帧原始计数</h2><table><thead><tr><th>Actor</th><th>DEVA 正像素</th><th>SAM3 正像素</th><th>融合正像素</th><th>未知像素</th><th>完整边界像素</th><th>下沿像素</th><th>可见框候选 LiDAR</th><th>落入融合正区</th></tr></thead><tbody id="counts"></tbody></table>
<p class="small">“下沿像素”指该像素属于 mask、紧邻下方像素不属于 mask。已逐帧逐车核对上下沿均包含在完整轮廓中。LiDAR 仅显示同帧真实 BUILD / QUERY 允许点，按相机深度最近层＋0.10m 过滤；没有生成或补点。</p>
<details><summary>原始输入、来源颜色与验证记录</summary><p>可直接下载当前帧 <a id="npz">原始冻结 NPZ（全部 43 个实例通道）</a>。通道包括 <code>masks</code>、<code>mask_known</code>、<code>mask_valid</code>、<code>SAM_prediction</code>、<code>source</code>、<code>uncertainty</code>。逐车二值 PNG 保持 480×320，非抗锯齿预览。</p><p>Source 图：橙＝仅 DEVA；蓝＝仅 SAM3；绿＝双方同意；洋红＝未知。该图的颜色表示来源，顶部概览颜色表示 Actor，两者图例分开。</p><p class="hash" id="hash"></p><p><a href="verification.json">49 帧只读哈希与边界核查记录</a></p></details>
<p class="small">本页展示通过正式编辑准入的 7 个 Actor：5、6、12、18、25、26、37。其余通道保留在原始 NPZ 中。旧 DEVA、新 SAM3、融合结果都作为待核查预测呈现，不视为人工真值。</p>
<script>
const frames=__FRAMES__,colors={5:'#ffb428',6:'#2ddcff',12:'#c482ff',18:'#ff78b9',25:'#64e650',26:'#ff6441',37:'#4696ff'},actors=[5,6,12,18,25,26,37];
const el=id=>document.getElementById(id);for(const a of actors){el('actor').add(new Option('Actor '+a,a));el('legend').insertAdjacentHTML('beforeend',`<span><i class="dot" style="background:${colors[a]}"></i>Actor ${a}</span>`)}el('actor').value='5';let timer=null;
function card(title,path){return `<div class="card"><h3>${title}</h3><a href="${path}" target="_blank"><img loading="lazy" src="${path}" alt="${title}"></a></div>`}
function draw(){const f=frames[+el('frame').value],fid=f.frame,a=+el('actor').value,aid=String(a).padStart(2,'0'),base=`frames/${fid}`,b=`binary/${fid}/a${aid}`;el('status').textContent=`raw${f.raw} · ${fid} · ${f.split} · ${+el('frame').value+1}/49`;
el('overview').innerHTML=[['原始 RGB','rgb'],['DEVA：完整填色＋轮廓','deva'],['SAM3：完整填色＋轮廓','sam3'],['实际融合监督：完整 mask＋未知区','fused'],['融合正标签：实心 mask / 洋红未知','labels'],['真实 LiDAR 投影（全部允许可见点）','lidar']].map(([t,p])=>card(t,`${base}/${p}.png`)).join('');
el('binary').innerHTML=[['三值监督：白车体 / 黑实例外 / 灰未知','supervision'],['Positive：原始二值正标签','positive'],['Valid：白有效 / 黑未知','valid'],['DEVA：原始完整二值 mask','deva'],['SAM3：原始完整二值 mask','sam3'],['Source：标签来源 / 洋红未知','source']].map(([t,p])=>card(t,`${b}/${p}.png`)).join('');
el('actorhead').textContent=`Actor ${a}：原尺寸监督与车底细节`;el('zoom').src=`actors/${fid}_a${aid}.png`;el('zoomlink').href=el('zoom').src;el('zoomlink').target='_blank';let af=f.actors.find(x=>x.actor===a);el('crop').textContent=`原图裁剪范围 [x0,y0,x1,y1]：${af.crop_box.join(', ')}；采用最近邻放大，不平滑或补画 mask。若无正标签与候选点，显示整幅画面。`;
el('counts').innerHTML=f.actors.map(x=>`<tr style="${x.actor===a?'background:#1e334a':''}"><td style="color:${colors[x.actor]}">a${x.actor}</td>${[x.DEVA,x.SAM3,x.positive,x.unknown,x.complete_boundary,x.bottom_boundary,x.frontmost_box_candidate_lidar,x.positive_lidar_hits].map(v=>`<td>${v.toLocaleString()}</td>`).join('')}</tr>`).join('');el('npz').href=`arrays/${fid}.npz`;el('hash').textContent=`冻结 NPZ SHA-256（绘图前后相同）：${f.frozen_sha256}`;}
el('frame').oninput=draw;el('actor').onchange=draw;function step(d){el('frame').value=(+el('frame').value+d+frames.length)%frames.length;draw()}el('prev').onclick=()=>step(-1);el('next').onclick=()=>step(1);el('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;el('play').textContent='播放 49 帧'}else{timer=setInterval(()=>step(1),500);el('play').textContent='暂停播放'}};draw();
</script></body></html>'''
(O/'report.html').write_text(template.replace('__FRAMES__',json.dumps(frames,ensure_ascii=False)),encoding='utf8')
# Preserve legacy drawings, replace only derived local review images.
local=W/'input_reviews';backup=W/'input_reviews_legacy_upper_edge'
backup.mkdir(exist_ok=True)
for old in local.glob('*_review.png'):
    if not (backup/old.name).exists():shutil.copy2(old,backup/old.name)
    shutil.copy2(O/'reviews'/old.name,old)
unified=ROOT.parent.parent/'2026-10-15/file-c-users-dengyunlong-documents-codex/outputs/DrivingEditor_unified_report.html'
assert unified.exists(),unified
html=unified.read_text(encoding='utf8');begin='<!-- R14_COMPLETE_MASK_REVIEW_BEGIN -->';end='<!-- R14_COMPLETE_MASK_REVIEW_END -->'
backup_unified=W/'unified_report.before_complete_mask.html'
if not backup_unified.exists():shutil.copy2(unified,backup_unified)
entry=f'''{begin}<section style="padding:20px;margin:20px 0;border:2px solid #547ca8;border-radius:12px;background:#10243a;color:#e9f1fa"><h2>R14 完整 mask 与 LiDAR 核查（绘图修正）</h2><p>49 帧完整填色、上下左右轮廓、原始二值 mask、逐像素有效域、7 Actor 车底放大与真实 LiDAR。只修显示，冻结监督不变；完整训练与编辑实验继续。</p><a style="color:#8ac5ff" href="{(O/'report.html').as_uri()}">打开完整 mask 核查</a></section>{end}'''
if begin in html:
    start=html.index(begin);finish=html.index(end,start)+len(end);html=html[:start]+entry+html[finish:]
else:
    pos=html.lower().find('<body');pos=html.index('>',pos)+1;html=html[:pos]+entry+html[pos:]
unified.write_text(html,encoding='utf8')
print(json.dumps(dict(report=str(O/'report.html'),unified_updated=True,legacy_review_preserved=True,frames=len(frames)),ensure_ascii=False))

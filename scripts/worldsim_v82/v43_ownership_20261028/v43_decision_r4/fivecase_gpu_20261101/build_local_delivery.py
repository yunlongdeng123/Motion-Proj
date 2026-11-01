"""取回后编码完整对照视频，生成独立报告与简洁可视化。"""
import concurrent.futures,json,pathlib,shutil,subprocess
import numpy as np
from PIL import Image,ImageDraw
HERE=pathlib.Path(__file__).resolve().parent
OUT=HERE.parent.parent/'outputs/V43_R4_FiveCase_GPU_20261101'
FFMPEG=shutil.which('ffmpeg')
def write(p,x):p.write_text(x,encoding='utf-8')
def encode(scene,op):
    case=scene['case'];src=OUT/'video_frames'/case/op;dst=OUT/'review/videos'/(case+'_'+op+'.mp4');dst.parent.mkdir(exist_ok=True)
    cmd=[FFMPEG,'-hide_banner','-loglevel','error','-y','-framerate','10','-i',str(src/'left/%03d.png'),
        '-framerate','10','-i',str(src/'right/%03d.png'),'-filter_complex','[0:v][1:v]hstack=inputs=2[v]',
        '-map','[v]','-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(dst)]
    subprocess.run(cmd,check=True)
    boxed=HERE/'boxed_frames'/case/op;boxed.mkdir(parents=True,exist_ok=True)
    for i,frame in enumerate(scene['frames']):
        for side in ['left','right']:
            folder=boxed/side;folder.mkdir(exist_ok=True)
            im=Image.open(src/side/('%03d.png'%i)).convert('RGB');draw=ImageDraw.Draw(im)
            center=np.asarray(frame['actor_center_world']);rotation=np.asarray(frame['actor_rotation_world']);dims=np.asarray(frame['dimensions'])
            camera=np.asarray(frame['camera_to_world']);fx,fy,cx,cy=frame['render_intrinsics']
            def box(c,r,color):
                signs=np.asarray([[1 if n&(1<<j) else -1 for j in range(3)] for n in range(8)])
                world=signs*dims/2@r.T+c;cv=(world-camera[:3,3])@camera[:3,:3]
                for n in range(8):
                    for j in range(3):
                        m=n|(1<<j)
                        if n==m or min(cv[n,2],cv[m,2])<=.05:continue
                        points=[(float(fx*cv[k,0]/cv[k,2]+cx),float(fy*cv[k,1]/cv[k,2]+cy)) for k in [n,m]]
                        if all(np.isfinite(x) and abs(x)<1e5 for p in points for x in p):draw.line(points,fill=color,width=1)
            if op in ['delete','shift','rotate']:box(center,rotation,'#9aafbd')
            if op!='delete':
                c=center+([0,1.5,0] if op=='shift' else np.zeros(3));r=rotation
                if op=='rotate':
                    angle=np.pi/6;r=np.asarray([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])@rotation
                box(c,r,'#ffcc54')
            im.save(folder/('%03d.png'%i))
    boxed_dst=dst.with_name(dst.stem+'_boxes.mp4')
    command=cmd.copy();command[command.index(str(src/'left/%03d.png'))]=str(boxed/'left/%03d.png');command[command.index(str(src/'right/%03d.png'))]=str(boxed/'right/%03d.png');command[-1]=str(boxed_dst)
    subprocess.run(command,check=True)
    return dict(case=case,operation=op,bytes=dst.stat().st_size,boxed_bytes=boxed_dst.stat().st_size,expected_frames=49,
        boxes='annotation trajectory projected in frozen native camera; old/requested positions only, not edited GT')

VIEWER='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>车辆编辑对照</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1420;color:#e4edf7;font:16px/1.6 system-ui,sans-serif}main{max-width:1380px;margin:auto;padding:24px}a{color:#8dc5ff}h1{font-size:28px;margin:12px 0}label{display:inline-flex;flex-direction:column;margin:0 16px 14px 0;min-width:150px}select,button,input{font:inherit;color:inherit;background:#1b2a3d;border:1px solid #48617b;border-radius:6px;padding:8px}button{cursor:pointer}.pair,.headings{display:grid;grid-template-columns:1fr 1fr;gap:10px}.pair img{width:100%;display:block;background:#080e15}.headings{text-align:center;color:#b9cfe4;margin:14px 0 6px}video{display:block;width:100%;background:#080e15}section{border:1px solid #2b4057;padding:18px;border-radius:10px;margin:20px 0}p{color:#b7cadd;margin:10px 0}.warning{color:#f0cd91}[hidden]{display:none!important}.times{display:flex;gap:12px;align-items:center}.times input{flex:1}.numbers{font-variant-numeric:tabular-nums}details{margin-top:14px}code{overflow-wrap:anywhere}@media(max-width:650px){main{padding:14px}.pair{grid-template-columns:1fr}h1{font-size:24px}}</style>
<main><a id="back" href="../../V43_R4_GPU_20261031/actor_candidates/index.html">← 车辆候选</a><h1>车辆编辑对照</h1><p>同一场景、同一相机与时刻，对照固定候选的不同绑定。</p>
<label>车辆<select id="case"></select></label><label>操作<select id="op"><option value="none">无编辑</option><option value="delete">删除</option><option value="shift">横移 1.5 米</option><option value="rotate">旋转 30°</option></select></label><label>查看方式<select id="mode"><option value="video">完整视频 · 49 帧</option><option value="anchors">逐帧候选对照 · 三个锚点</option></select></label>
<section><div id="videoPanel"><div class="headings"><strong>原绑定</strong><strong>保守候选全部重绑定</strong></div><video id="video" controls playsinline preload="metadata"></video><p class="numbers" id="videoInfo"></p></div>
<div id="anchorPanel" hidden><label>候选范围<select id="domain"><option value="conservative">保守候选</option><option value="broad">宽候选</option></select></label><label>左侧方案<select id="left"></select></label><label>右侧方案<select id="right"></select></label><div class="times"><button id="prev">←</button><input id="frame" type="range" min="0" max="2" value="1"><button id="next">→</button><strong id="frameInfo"></strong></div><div class="pair"><img id="leftImage" alt="左侧固定绑定的原生渲染"><img id="rightImage" alt="右侧固定绑定的原生渲染"></div></div>
<p id="scope"></p><p class="warning" id="association"></p><p id="loadError" role="status" hidden>当前媒体无法读取，请检查文件是否完整。</p></section>
<p>宽候选来自固定的实测点邻域；保守候选另参考整段原始画面的车身贡献。两者均是候选对照，并非已验证的真实归属或自动选出的最优方案。参数、颜色和环境保持冻结。</p><p>视频仅覆盖这两条固定方案；其余方案只在三个锚点渲染。删除后未被真实观察过的背景，不作为正确路面的真值。</p><details><summary>原始轨迹与实验范围</summary><p id="sourceInfo"></p><p>横移沿原始世界坐标 y 方向；旋转绕世界竖直方向。未增加几何、颜色修复或二维补全。</p></details></main><script src="data.js"></script><script>
const $=id=>document.getElementById(id), data=window.FIVE_DATA;let current,frameIndex=1;
const names={none:'无编辑',delete:'删除',shift:'横移1.5米',rotate:'旋转30°'};
function routeName(i,n){if(i===0)return '保持原绑定';if(i===n-1)return '全部重绑定';const k=Math.round(Math.log2(n)),s=i.toString(2).padStart(k,'0');return '重绑定第 '+[...s].map((v,j)=>v==='1'?j+1:null).filter(Boolean).join('、')+' 组'}
function actorName(c){return 'Waymo · scene-'+c.scene_key.split('-').pop()+' / '+c.actor}
data.forEach((c,i)=>{$('case').add(new Option(actorName(c),i))});
function routeOptions(){const d=current.domains[$('domain').value];for(const side of ['left','right']){const old=$ (side).value;$(side).replaceChildren();for(let i=0;i<d.routes;i++)$(side).add(new Option(routeName(i,d.routes),'h'+String(i).padStart(3,'0')));$(side).value=side==='right'?'h'+String(d.routes-1).padStart(3,'0'):'h000'}paint()}
function paint(){const d=current.domains[$('domain').value],f=d.anchors[Number($('frame').value)],op=$('op').value;$('frameInfo').textContent='原始帧 '+String(Number(f)%1000);for(const side of ['left','right']){const r=d.images.find(x=>x.route===$(side).value&&x.operation===op&&x.frame===f);$(side+'Image').src=r?r.path:''} $('videoPanel').hidden=$('mode').value!=='video';$('anchorPanel').hidden=$('mode').value==='video'}
function video(){const op=$('op').value;$('video').pause();$('video').src=current.video_pairs.find(v=>v.operation===op).path;$('video').load();$('videoInfo').textContent=names[op]+' · 原始帧 '+current.windows.join('—')+' · 49帧 / 10fps';paint()}
function choose(){current=data[Number($('case').value)];$('loadError').hidden=true;$('scope').textContent='宽候选 '+current.domains.broad.groups.reduce((a,b)=>a+b,0)+' 行；保守候选 '+current.domains.conservative.groups.reduce((a,b)=>a+b,0)+' 行。';$('association').textContent=current.dynamic_binding_mode==='multi_actor_legal_proxy'?'此车来自多车动态层，目标行使用整段观测贡献的暂定关联；未获得逐行真实身份。':current.dynamic_binding_mode==='unbound'?'原动态层没有这辆车的独立资产，本轮只编辑重绑定的候选行，其他动态车辆保留。':'';$('sourceInfo').textContent=current.scene_key+' / '+current.actor+'；3个场景中的5辆开发车辆，尚未作独立泛化检验。';routeOptions();video();history.replaceState(null,'','#case='+encodeURIComponent(current.case))}
for(const id of ['left','right','frame','mode'])$(id).addEventListener(id==='frame'?'input':'change',paint);$('case').onchange=choose;$('domain').onchange=routeOptions;$('op').onchange=video;$('prev').onclick=()=>{$('frame').value=Math.max(0,Number($('frame').value)-1);paint()};$('next').onclick=()=>{$('frame').value=Math.min(2,Number($('frame').value)+1);paint()};for(const id of ['video','leftImage','rightImage'])$(id).onerror=()=>{$('loadError').hidden=false};const wanted=new URLSearchParams(location.hash.slice(1)).get('case');const selected=data.findIndex(c=>c.case===wanted);if(selected>=0)$('case').value=selected;choose();
</script></html>'''

def report(summary):
    rows=summary['cases'];table=''.join('<tr><td>'+r['case'].replace('waymo-train-','scene-').replace('_',' / ')+'</td><td>'+str(r['parent_candidate_rows'])+'</td><td>'+str(r['refined_candidate_rows'])+'</td><td>'+str(r['main_anchor_outputs'])+'</td><td>'+str(r['full_sequence_outputs'])+'</td><td>'+f"{r['native_none_parity_max_abs']:.3g}"+'</td></tr>' for r in rows)
    mass=''.join('<li>'+r['case']+'：最大绝对质量差 '+f"{r['contribution_mass_max_abs']:.6g}"+'；最大相对差 '+f"{r['contribution_mass_max_relative']:.6g}"+'；严格门禁 '+str(r['strict_contribution_mass_pass'])+'</li>' for r in rows)
    architecture='''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="240" viewBox="0 0 1120 240"><defs><marker id="a" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0 0L10 5L0 10" fill="#345c82"/></marker></defs><g fill="#edf4fa" stroke="#6589ab" stroke-width="2"><rect x="10" y="65" width="230" height="110" rx="8"/><rect x="290" y="65" width="220" height="110" rx="8"/><rect x="560" y="65" width="250" height="110" rx="8"/><rect x="860" y="65" width="250" height="110" rx="8"/></g><g font-family="Microsoft YaHei,sans-serif" font-size="19" text-anchor="middle" fill="#17344f"><text x="125" y="100">冻结高斯、RGB、LiDAR</text><text x="125" y="132">轨迹与暂定车身掩码</text><text x="400" y="100">宽候选 / 保守候选</text><text x="400" y="132">固定共享绑定方案</text><text x="685" y="100">原生 GPU 四操作</text><text x="685" y="132">保存、恢复、跨帧重放</text><text x="985" y="100">浮点记录与完整视频</text><text x="985" y="132">有效域与未决项分列</text></g><g fill="none" stroke="#345c82" stroke-width="3" marker-end="url(#a)"><path d="M242 120H285"/><path d="M512 120H555"/><path d="M812 120H855"/></g></svg>'''
    write(OUT/'architecture.svg',architecture)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V43 R4 五例 GPU 实验</title><style>body{font:17px/1.75 system-ui,sans-serif;max-width:1120px;margin:32px auto;padding:0 22px;color:#23364a}h1{font-size:30px}h2{font-size:23px;margin-top:32px}a{color:#175d9a}img{width:100%}table{width:100%;border-collapse:collapse;font-size:15px}td,th{padding:10px;border-bottom:1px solid #d5e0e8;text-align:left}code{overflow-wrap:anywhere}.note{background:#f4f7fa;padding:16px;border-left:4px solid #7794af}</style><h1>V43 R4：五辆车的冻结表示编辑实验</h1><p>2026-11-01 · 5辆开发车辆，3个场景祖先。主任务为E0恢复及E2共享绑定小域对照；新增模型训练更新为0。<a href="review/index.html">打开编辑对照</a></p><img src="architecture.svg" alt="输入、固定候选与共享绑定、原生GPU、浮点与视频输出"><h2>实际执行</h2><p>恢复原始静态、动态、环境与运行状态。对每例完整49帧计算原始none与车身/外域伴随贡献；分别冻结宽候选和保守候选，每个候选域枚举最多8份共享绑定，覆盖none/delete/shift/rotate三个锚点。同一方案跨操作复用。另对保守域的“原绑定”和“全部重绑定”两个固定端点渲染完整49帧，端点并非事后挑选的最优方案。</p><table><tr><th>车辆</th><th>宽候选行</th><th>保守候选行</th><th>锚点输出</th><th>整段输出</th><th>原生none最大差</th></tr>'''+table+'''</table><p>宽候选为三个固定锚点的实测邻域（0.25米）首次命中分组。保守候选额外要求原始49帧伴随贡献中车身比例≥0.95、总量&gt;1e-4；这是沿用的开发候选域，不是校准后的资格或身份概率。两域结果分开比较，不能混合成一项最优结果。</p><h2>恢复与数值验收</h2><p>每个域均实际保存完整推理状态及冻结原始bundle引用，在新进程恢复相机、运动、环境、路由和RNG，重放两份端点方案、全部四操作，再进入枚举或整段入口。原始检查点没有更新；没有创建模型优化器。原生前向与保存恢复重放的误差见<a href="summary.json">逐例摘要</a>。工程重放与正常输出分列在<a href="execution_resources.json">执行账</a>。</p><p class="note">原生前向的通过不能替代伴随贡献数值检查。FP32光栅及原子伴随的全图质量总和，在部分564帧仍未通过0.05绝对/2ppm相对的严格门禁；原始FAIL保留，没有放宽该门禁冒称通过。后续继续的是独立的原生前向/恢复实验；伴随数据和保守候选均保持诊断身份，不能作为精确归属、完整表面或正式P资格证书。</p><ul>'''+mass+'''</ul><h2>结果可以说明什么</h2><p>现已得到相同冻结表示内、不同共享绑定导致的真实原生结果与保持域变化，可直接查看目标、旧位置和背景的取舍。<a href="metrics.csv">全部锚点指标</a>保留每个方案，不自动按单项残影指标挑赢家。outside_source_body_RGB_change仅表示原始车身掩码外的图像变化；平移/旋转本来就可能改变该区域，因此它不是自动的误编辑率。</p><p>212/a35与564/a22没有原动态目标资产，仅允许重绑定的静态候选参与目标操作，其他动态车辆固定。008/a37来自五车动态层，逐行原始身份未记录；本轮根据完整49帧合法贡献冻结目标动态子集，属于暂定关联控制，未将整层五车移动或删除，也不称原生真实owner。</p><h2>R4阶段边界</h2><p>20项逐操作正式资格继续unknown：尚缺独立校准的深度/表面、操作视向、历史背景显露和安全阈值。真实街景没有独立完整编辑GT，本轮不能报告精确有限域regret、完整删除/旋转成功率或跨场景学习收益。已有观测像素只用于原场景与局部诊断，未假造未知路面监督。</p><p>本轮完成五例GPU对照；尚未进入L0训练。下一步应先从这些固定方案中核实可辨识的配对收益与合法有效域，再建立学习所需的训练/校准/测试角色。不能用事后编辑效果回改事前资格，也不能为了启动训练而把未知项默认为通过。</p><h2>文件与复现</h2><p><a href="run_contract.json">输入与动作合同</a> · <a href="source/run_five.py">GPU入口</a> · <a href="source/prepare_five.py">观测与候选准备</a> · <a href="source/summarize_five.py">只读结果汇总</a></p><p>原始FP32、完整推理状态、工程失败现场保留远端；本地交付视频、逐方案锚点图、指标及恢复收据。人工verdict未代填。报告不加入共享可视化平台。</p></html>'''
    html=html.replace('分别冻结宽候选和保守候选','分别冻结宽候选、原位可见保守候选，以及原位或剥离后显露候选')
    html=html.replace('每个域均实际保存','三个域均实际保存')
    exposed_rows=''.join('<li>'+r['case']+'：宽候选 '+str(r['parent_candidate_rows'])+'；原位保守 '+str(r['refined_candidate_rows'])+'；加入显露来源 '+str(r['exposed_candidate_rows'])+'</li>' for r in rows)
    html=html.replace('<h2>恢复与数值验收</h2>','<h2>预登记显露来源控制</h2><p>原位贡献筛选使212/a0与564/a0的静态候选为空。保留这一负结果后，按R4已有的显露来源设计补齐控制：仅剥离预先冻结的目标动态子集，计算原始49帧同一车身掩码上的来源贡献。在同一个原始U和相同三锚分组内，用原位或显露条件的贡献比例≥0.95、总量&gt;1e-4取并集；没有使用编辑评分、学生输出或独立编辑参考选择行。新特征规则在该控制计算前保存，未倒写为首轮预注册。</p><ul>'+exposed_rows+'</ul><p>默认完整视频展示该显露来源域的两条固定端点方案；原位保守域的完整FP32及两域三锚结果仍保留。剥离所得颜色不等于正确隐藏背景，更多显露来源也不等于更准确归属。</p><h2>恢复与数值验收</h2>')
    full_rows=''.join('<tr><td>'+r['case']+'</td><td>'+str(r['fullsource_candidate_rows'])+'</td></tr>' for r in rows)
    html=html.replace('<h2>恢复与数值验收</h2>','<h2>完整来源候选控制</h2><p>进一步核查表明，固定0.25米实测邻域只覆盖三个锚点中约3%–14%的高车身贡献显露来源质量。此分母仅是本次候选贡献池，不是真实车辆归属或全局召回GT。因此，不能用邻域小域的负结果否定R4决策空间。新增的最后一项来源控制直接读取全场景原始静态行在三个锚点的原位或剥离后贡献，以相同0.95比例和1e-4总量建立候选，再按首个支持锚点分为最多三组。规则在新候选生成及新域渲染前保存，没有用编辑评分或独立GT选行。</p><table><tr><th>车辆</th><th>完整来源候选行</th></tr>'+full_rows+'</table><p>默认视频现在展示完整来源域的两个固定端点。该域可能包含道路、阴影、背景或大足迹高斯；纳入候选不等于属于车辆。四个候选域分开记账，不跨域拼接“最佳方案”，三个粗分组也不代表完整逐行搜索。</p><p><a href="evidence/source_recall_control.json">来源范围与质量分母</a></p><h2>恢复与数值验收</h2>')
    html=html.replace('默认完整视频展示该显露来源域的两条固定端点方案','显露来源域的两条固定端点方案仍保留远端')
    html=html.replace('三个域均实际保存','四个域均实际保存')
    html=html.replace('原始FAIL保留，没有放宽该门禁冒称通过。','原始FAIL保留，没有放宽该门禁冒称通过。')
    html=html.replace('<h2>文件与复现</h2>','<h2>工程重放记录</h2><p>CPU准备首次遇到旧Torch环境不接受Path类型mmap参数，已在兼容CPU环境重放。早期多个控制器重叠，实际并发超过两任务，导致212/a35一次OOM；该阶段从保存状态在新目录完整重放，后续控制器串接、全局最多两GPU任务。冗余564/a22审计在既有冻结目录处停止，未覆盖有效输出。部分早期stdout采用同名日志而被后续尝试替换；事件、部分浮点输出、原脚本及已取回的OOM日志仍保留，不能声称所有stdout均完整。详见<a href="evidence/engineering_attempts.json">工程记录</a>。</p><h2>文件与复现</h2>')
    write(OUT/'report.html',html)

def main():
    if not FFMPEG:raise RuntimeError('缺ffmpeg')
    # data.js是本次CPU汇总生成的独立JSON赋值，不从网页抓取执行。
    text=(OUT/'review/data.js').read_text(encoding='utf-8');data=json.loads(text[len('window.FIVE_DATA='):-1])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:
        receipts=list(p.map(lambda args:encode(*args),[(s,o['operation']) for s in data for o in s['video_pairs']]))
    write(OUT/'video_receipts.json',json.dumps(receipts,indent=2))
    viewer=VIEWER.replace('保守候选全部重绑定','完整来源候选全部重绑定')
    viewer=viewer.replace('<option value="conservative">保守候选</option>','<option value="fullsource">完整来源候选</option><option value="exposed">邻域内原位或显露候选</option><option value="conservative">仅原位保守候选</option>')
    viewer=viewer.replace("+' 行。';", "+' 行；加入显露来源 '+current.domains.exposed.groups.reduce((a,b)=>a+b,0)+' 行。';")
    viewer=viewer.replace('保守候选另参考整段原始画面的车身贡献','保守候选参考整段原始画面的车身贡献；显露候选另纳入预登记目标剥离后的来源贡献')
    viewer=viewer.replace('完整视频 · 49 帧','实验窗口视频 · 49 帧')
    viewer=viewer.replace('宽候选来自固定的实测点邻域','宽候选来自固定的实测点邻域；完整来源候选另从全场景三个锚点的原始或剥离后车身贡献建立')
    viewer=viewer.replace('<video id="video"','<label><span><input id="showBoxes" type="checkbox" checked> 显示位置参照框</span></label><video id="video"')
    viewer=viewer.replace("$('video').src=current.video_pairs.find(v=>v.operation===op).path;","const media=current.video_pairs.find(v=>v.operation===op);$('video').src=$('showBoxes').checked?media.boxed_path:media.path;")
    viewer=viewer.replace("$('case').onchange=choose;","$('showBoxes').onchange=video;$('case').onchange=choose;")
    viewer=viewer.replace('<p id="scope"></p>','<p>黄色框为请求位置，灰色框为原位置参照；删除只显示原位置。位置框来自轨迹，不是编辑结果真值。</p><p id="scope"></p>')
    write(OUT/'review/index.html',viewer)
    report(json.loads((OUT/'summary.json').read_text(encoding='utf-8')))
    shutil.copy2(__file__,OUT/'source/build_local_delivery.py')
    print(json.dumps(dict(videos=len(receipts),out=str(OUT))))
if __name__=='__main__':main()

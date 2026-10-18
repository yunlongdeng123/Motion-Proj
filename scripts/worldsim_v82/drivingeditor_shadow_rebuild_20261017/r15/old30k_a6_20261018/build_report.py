from pathlib import Path
import json,shutil,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

BASE=Path(__file__).resolve().parents[2]
SRC=BASE/'work/old30k_fragments/assets'
OUT=BASE/'outputs/old30k_a6_check_20261018'
OUT.mkdir(exist_ok=True)
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
TITLE=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',23)
anchors=json.loads((BASE/'outputs/r15_diagnostic_evidence/actors/actors.json').read_text())['anchors']
crops={x['raw']:x['crop'] for x in anchors if x['actor']==6}

def tile(canvas,p,box,label,xy,size=(250,202),font=SMALL):
    x,y=xy;w,h=size;draw=ImageDraw.Draw(canvas)
    draw.text((x+6,y+3),label,font=font,fill='#183348')
    im=Image.open(p).convert('RGB')
    if box:im=im.crop(box)
    im=ImageOps.contain(im,(w-12,h-36),Image.Resampling.NEAREST)
    canvas.paste(im,(x+(w-im.width)//2,y+30+(h-36-im.height)//2))

def observed(raw,diagnostic=False):
    fid=f'0008{raw:03d}';box=crops[raw]
    items=[(SRC/'old'/fid/'gt.png','真实 RGB'),(SRC/'old'/fid/'full.png','旧 GN11 30K＋实例 30K'),
        (SRC/'old'/fid/'static.png','旧版静态/PVG 分支（含 a6）'),
        (SRC/'r15'/fid/'full_no_shadow.png','R15 4K 完整场景／关阴影'),
        (SRC/'r15'/fid/'a06_rgb.png','R15 a6 全部归属点'),(SRC/'r15'/fid/'a06_alpha.png','R15 a6 独立 raw alpha')]
    canvas=Image.new('RGB',(1500,242),'#f3f6f8');d=ImageDraw.Draw(canvas)
    d.text((10,5),f'同一辆 a6 · raw{raw} · 全部格子使用同一原生图像裁剪；未对资产乘 body mask',font=FONT,fill='#183348')
    for i,(p,label) in enumerate(items):tile(canvas,p,box,label,(i*250,38))
    canvas.save(OUT/f'a6_raw{raw}_observed.png')
    if diagnostic:
        items=[(SRC/'a6_direct'/fid/'center_box_rgb.png','旧版：框内中心'),
            (SRC/'a6_direct'/fid/'center_box_alpha.png','旧版：框内 raw alpha'),
            (SRC/'a6_direct'/fid/'center_halo075m_rgb.png','旧版：框外扩 0.75m'),
            (SRC/'a6_direct'/fid/'support3sigma_rgb.png','旧版：3σ 支撑相交'),
            (SRC/'r15'/fid/'a06_rgb.png','R15：a6 全部归属点'),
            (SRC/'r15'/fid/'a06_alpha.png','R15：独立 raw alpha')]
        canvas=Image.new('RGB',(1500,250),'#f3f6f8');d=ImageDraw.Draw(canvas)
        d.text((10,4),f'同一辆 a6 · raw{raw} · 旧版前三种是几何选择诊断，不是既有独立资产／归属真值',font=FONT,fill='#183348')
        for i,(p,label) in enumerate(items):tile(canvas,p,box,label,(i*250,38))
        canvas.save(OUT/f'a6_raw{raw}_geometry_diagnostic.png')

audit=json.loads((SRC/'audit.json').read_text())
direct=json.loads((SRC/'a6_direct/a6_direct.json').read_text())
contrib=json.loads((SRC/'a6_contributors/contributions.json').read_text())
for raw in range(89,138):
    if raw not in crops:
        a=np.asarray(Image.open(SRC/'r15'/f'0008{raw:03d}'/'a06_mask.png'))
        yy,xx=np.where(a>0)
        crops[raw]=[max(0,int(xx.min())-24),max(0,int(yy.min())-20),min(480,int(xx.max())+25),min(320,int(yy.max())+40)] if len(xx) else [180,140,330,260]
    observed(raw,True)

for raw in [105,113,137]:
    fid=f'0008{raw:03d}'
    items=[(SRC/'a6_contributors'/f'raw{raw}_top98_rgb.png','旧：a6 像素 98% 贡献集／RGB'),
        (SRC/'a6_contributors'/f'raw{raw}_top98_alpha.png','旧：同一贡献集／raw alpha'),
        (SRC/'r15'/fid/'a06_rgb.png','R15：全部 a6 owner／RGB'),
        (SRC/'r15'/fid/'a06_alpha.png','R15：全部 a6 owner／raw alpha')]
    canvas=Image.new('RGB',(1440,310),'#f3f6f8');d=ImageDraw.Draw(canvas)
    d.text((10,5),f'同一 a6 · raw{raw} · 原相机全画幅；旧贡献集包含背景，不是认证车辆资产',font=FONT,fill='#183348')
    for k,(p,label) in enumerate(items):tile(canvas,p,None,label,(360*k,38),(360,268))
    canvas.save(OUT/f'a6_raw{raw}_contributors.png')
    canvas=Image.new('RGB',(1440,850),'#f3f6f8');d=ImageDraw.Draw(canvas)
    d.text((10,4),f'a6 · raw{raw} · 共同物理相机的几何诊断：旧 98% 像素贡献集 vs R15 owner6',font=FONT,fill='#183348')
    for row,view in enumerate(['front','rear','side']):
        labels=['旧贡献集 RGB','旧贡献集 raw alpha','R15 owner6 RGB','R15 owner6 raw alpha']
        paths=[SRC/'a6_contributors'/f'raw{raw}_{view}_top98_rgb.png',SRC/'a6_contributors'/f'raw{raw}_{view}_top98_alpha.png',
               SRC/'a6_direct'/f'raw{raw}_{view}_r15_owner6_rgb.png',SRC/'a6_direct'/f'raw{raw}_{view}_r15_owner6_alpha.png']
        for col,(p,label) in enumerate(zip(paths,labels)):tile(canvas,p,None,f'{view} · {label}',(col*360,38+row*268),(360,268))
    canvas.save(OUT/f'a6_raw{raw}_novel.png')

for src,name in [('audit.json','audit.json'),('a6_direct/a6_direct.json','a6_direct.json'),('a6_contributors/contributions.json','contrib.json')]:
    shutil.copy2(SRC/src,OUT/name)
code=OUT/'code';code.mkdir(exist_ok=True)
for src,name in [('run_fragment_audit.py','run_fragment_audit_20261018.py'),('recompute_old_isolated.py','recompute_old_isolated_20261018.py'),
                 ('run_a6_direct.py','run_a6_direct_20261018.py'),('run_a6_contributions.py','run_a6_contributions_20261018.py')]:
    shutil.copy2(BASE/'work/old30k_fragments'/src,code/name)

svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="220" viewBox="0 0 1180 220"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#61798b"/></marker></defs><rect width="1180" height="220" rx="12" fill="#f3f6f8"/><g fill="white" stroke="#b5c7d5"><rect x="15" y="55" width="220" height="115" rx="8"/><rect x="275" y="55" width="210" height="115" rx="8"/><rect x="525" y="20" width="280" height="80" rx="8"/><rect x="525" y="120" width="280" height="80" rx="8"/><rect x="845" y="55" width="320" height="115" rx="8"/></g><g stroke="#61798b" stroke-width="2" fill="none" marker-end="url(#a)"><path d="M235 112H270"/><path d="M485 90H505V60H520"/><path d="M485 135H505V160H520"/><path d="M805 60H825V90H840"/><path d="M805 160H825V135H840"/></g><g font-family="Microsoft YaHei,Arial" text-anchor="middle" fill="#19384d"><g font-size="18" font-weight="bold"><text x="125" y="84">旧 30K＋30K / R15 4K</text><text x="380" y="84">同一 a6 / 真实轨迹</text><text x="665" y="48">49 帧原生重渲染</text><text x="665" y="148">3 锚帧精确 alpha*T</text><text x="1005" y="84">a6 证据与结论边界</text></g><g font-size="15"><text x="125" y="116">冻结检查点，参数更新 0</text><text x="125" y="146">不使用 18K 道路修补版</text><text x="380" y="116">坐标 / 投影 / 原相机核对</text><text x="380" y="146">同裁剪，不乘 body mask</text><text x="665" y="78">完整场景 / 3D 框选择</text><text x="665" y="178">98%贡献集 / 共同几何视角</text><text x="1005" y="116">观测图正常 ≠ 3D 资产干净</text><text x="1005" y="146">可见贡献 ≠ 车辆归属真值</text></g></g></svg>'''
(OUT/'arch.svg').write_text(svg,encoding='utf-8')

rows=''.join(f"<tr><td>{r['raw']}</td><td>{r['body_pixels']}</td><td>{100*(1-r['static_alpha_mass_fraction_center_inside_box']):.2f}%</td><td>{100*(1-r['static_alpha_mass_fraction_center_inside_halo075m']):.2f}%</td><td>{100*r['static_alpha_mass_fraction_max_principal_scale_gt1m']:.2f}%</td><td>{r['top98_static_point_count']}</td><td>{r['top98_max_principal_scale_p50_p95_p99_m'][1]:.2f} m</td></tr>" for r in contrib['records'])
summary={'task_id':'WS-V82-R15-OLD30K-A6-20261018','failure_ledger_refs':['V82-F06'],
    'failure_ledger_delta':'none','same_track_id':direct['track_id'],'frames':49,'anchors':[105,113,137],
    'old_full_png_bitwise_matches_original_baseline':49,'training_updates':0,
    'old_a6_independent_owner':False,'conclusion':'Old a6 observed views look cleaner, but its PVG representation has nonlocal large Gaussian support and is not demonstrated to be a clean separable asset. R15 owner6 bottom fragments remain unresolved.',
    'limits':'Exact old a6-owned bottom-fragment comparison unavailable: no old owner6; pixel contributors include legitimate background. No ownership transfer or road relabel.',
    'human_verdict':None,'problem_solved':None,'records':contrib['records']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
shutdown={'at_singapore':'2026-10-18T16:04+08:00','command':'sync && /usr/bin/shutdown',
    'command_exit_code':0,'subsequent_SSH_exit_code':1,
    'subsequent_SSH_output':'Connection closed by 116.136.52.182 port 38300',
    'cloud_billing_console_status_checked':False}
(OUT/'shutdown.json').write_text(json.dumps(shutdown,indent=2),encoding='utf-8')

report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>a6 旧 30K＋30K 直接对照</title>
<style>body{{font:16px/1.75 "Microsoft YaHei",Arial;background:#f4f7fa;color:#203345;margin:0}}main{{max-width:1500px;margin:auto;padding:25px}}section,.lead{{background:white;border:1px solid #d8e2ea;padding:20px;margin:20px 0;border-radius:12px}}h1{{font-size:29px}}h2{{font-size:22px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d9e3eb;padding:9px;text-align:left}}th{{background:#eff5fa}}.small{{font-size:14px;color:#586f80}}button,select,input{{font:inherit}}a{{color:#1264a3}}.controls{{position:sticky;top:0;background:#fff;padding:10px;border:1px solid #d8e2ea;z-index:2}}input[type=range]{{width:45%}}</style>
<main><p class="small">2026-10-18 · WS-V82-R15-OLD30K-A6-20261018 · 同一 track：{direct['track_id']} · 未使用 subagent</p>
<h1>a6：旧版观测画面更正常，但旧版的对应 3D 表示也有大尺度外溢与几何纠缠</h1>
<div class="lead"><strong>结论分两层：</strong>旧 GN11 30K＋实例 30K 的 a6，在原观察视角中比 R15 4K 更自然；但它由旧静态/PVG 分支表达，未形成独立 a6 资产。直接检查同一 a6 的几何与精确像素贡献后，旧版也暴露出长条、片状和大尺度高斯支撑。<br><strong>不能把 R15 车底外延全部认定为本轮新引入；也不能据此宣称旧版已经有同一批、同样归属的车底碎片。</strong>R15 从原测量重训，没有继承旧版这些已训练高斯。当前归属资产的碎片问题仍未解决。</div>
<img src="arch.svg" alt="冻结旧版和R15检查点，同一a6对齐，经49帧原生渲染和3锚帧精确贡献审计形成结论"/>
<section><h2>先看同一辆 a6：49 帧、同裁剪</h2><p>直接加载旧最终实例 step=30000 检查点；其背景来自 GN11 step=30000。重新渲染的 <strong>49/49 原生完整 RGB 与原报告旧基线逐像素一致</strong>。没有混用后续 18K 道路修补版，也没有重训。R15 以保留的 step=4000 冻结模型关阴影渲染，便于看清几何和外观。</p>
<div class="controls"><button id="prev">←</button> <input id="frame" type="range" min="89" max="137" value="105"> <button id="next">→</button> <output id="raw">raw105</output> <select id="jump"><option>105</option><option>113</option><option>137</option><option>89</option></select></div>
<img id="observed" src="a6_raw105_observed.png" alt="同一a6原相机六列对照"/>
<p class="small">左到右：真实 RGB、旧完整场景、旧静态/PVG、R15完整场景、R15全部owner6、R15独立alpha。当前模型的整块下部外延在独立资产中可见，但在完整场景里部分被前车遮挡。旧画面正常不能单独证明旧几何可分离。</p></section>
<section><h2>旧版 a6 的可见车身，主要由框外中心的高斯共同画出</h2><p>在固定 raw105／113／137，保持当前 a6 真实可见 body mask 不变，通过原联合渲染的 feature 梯度计算每个高斯的精确 <strong>alpha×透射率</strong> 贡献；旧版这三个 a6 车身区域的贡献均来自静态/PVG，而非旧版的三个独立实例。没有按颜色或高度删点。</p>
<table><tr><th>raw</th><th>a6车身像素</th><th>贡献来自框外中心</th><th>来自外扩0.75m框外中心</th><th>来自最大主轴σ&gt;1m的高斯</th><th>98%贡献集点数</th><th>贡献集主轴σ的P95</th></tr>{rows}</table>
<p>因此，旧版 a6 的正常车形并不对应一个局部、干净的 3D 车辆表示。把覆盖其98%静态可见贡献的高斯独立渲染，会出现明显大范围射线状／片状外溢。<strong>这支持旧版已存在非局部几何解释和提取纠缠；它不把这些高斯自动判成 a6 或道路。</strong>真实背景也会贡献这些像素，贡献集不是归属真值。</p>
<p class="small">贡献审计和共同几何视角仅有raw105、113、137三个固定锚帧；滑动时这两图显示最近锚帧，图内标注实际raw编号。</p><p class="small">贡献审计和共同几何视角仅有raw105、113、137三个固定锚帧；滑动时这两图显示最近锚帧，图内标注实际raw编号。</p><img id="contributors" src="a6_raw105_contributors.png" alt="旧版a6像素98%贡献集与R15全部owner6的原相机全画幅独立图"/>
<p class="small">此图保持原相机全画幅。旧集合按原完整场景的贡献选择；独立渲染改变遮挡关系，亮度和外延不能直接作为误归属像素计数。</p></section>
<section><h2>几何选择边界检查</h2><p>对旧版同一 a6 分别取：高斯中心在真实3D框内、框每面外扩0.75m、3σ轴向包围范围与框相交。框半尺寸采用R15由实测框建立的固定a6尺寸（2.167、1.039、0.720米），逐帧中心与朝向使用真实轨迹。框半尺寸采用R15由实测框建立的固定a6尺寸（2.167、1.039、0.720米），逐帧中心与朝向使用真实轨迹。完整保留选中高斯的形状及不透明度，不裁车底，不乘 body mask。三种入口结果差异很大；严格框内集合并不能还原完整车辆。这是提取诊断，没有创建或修复旧版 a6 归属。</p>
<img id="geometry" src="a6_raw105_geometry_diagnostic.png" alt="旧版a6三种几何选择与R15完整owner对照"/>
<p class="small">相机中心对齐最大误差 {direct['max_camera_center_alignment_error_m']:.2g}m；三锚帧a6中心投影差约0.125像素。坐标映射使用原数据的世界归一化与原始轨迹，没有靠手动拖动对齐。</p></section>
<section><h2>共同物理相机的三个几何视角</h2><p>下面始终是同一 a6 的前、后、侧视角；旧版显示覆盖98%可见贡献的集合，R15显示全部owner6。两者使用同一物理相机和光栅入口。旧集合包含真实背景；不能把两栏当作已经具备相同归属语义的车辆资产打分。</p>
<img id="novel" src="a6_raw105_novel.png" alt="同一a6三个物理视角的旧贡献集与R15owner6几何图"/>
<p class="small">这些视角用于检查高斯支撑与alpha。旧版颜色固定为原观察方向的SH值，不能据此评判完整新视角外观或光照；几何和alpha没有该颜色冻结限制。</p></section>
<section><h2>这次核对能回答什么</h2><p><strong>旧版也不是干净几何基线：</strong>a6在观测视图中的好效果掩盖了非局部高斯依赖，分离后已有严重外溢／拉伸。因此“新图里见到碎片，所以这种几何问题一定是R15新增”缺少依据。</p><p><strong>R15仍需要解决独立a6资产的车底外延：</strong>旧版没有独立a6归属，无法做完全同语义的owner资产对比，也无法证明两版存在相同道路碎片。当前高斯归属修改仍须使用可靠的3D与可见性证据。发现旧版的局限，不构成R15质量通过。</p><p>本轮只读重渲染和贡献审计，训练更新0；全部原检查点保留。3个锚帧是固定坏例诊断，49帧是同窗口重建视角，没有未见测试或编辑后配对真值。</p></section>
<section><h2>证据与资源</h2><p><a href="audit.json">检查点与49帧原生一致性</a> · <a href="a6_direct.json">同一a6的49帧3D入口</a> · <a href="contrib.json">三锚帧精确贡献</a> · <a href="summary.json">结论JSON</a> · <a href="shutdown_preflight.json">关机前检查</a> · <a href="shutdown.json">关机记录</a> · <a href="code/run_a6_contributions_20261018.py">贡献审计代码</a></p><p>代码和数值结果已推送GitHub v82；AutoDL于16:04（新加坡时间）执行关机，随后SSH连接关闭。本地保存完整原始导出，远端保留全部PNG、逐点NPZ及原检查点。未检查云控制台计费状态。</p><p class="small">failure_ledger_refs：V82-F06；failure_ledger_delta：none；human_verdict=null；problem_solved=null。</p></section>
<script>const f=document.getElementById('frame'),o=document.getElementById('raw');function update(){{let r=+f.value;o.textContent='raw'+r;document.getElementById('observed').src='a6_raw'+r+'_observed.png';document.getElementById('geometry').src='a6_raw'+r+'_geometry_diagnostic.png';let a=[105,113,137].reduce((x,y)=>Math.abs(y-r)<Math.abs(x-r)?y:x);document.getElementById('contributors').src='a6_raw'+a+'_contributors.png';document.getElementById('novel').src='a6_raw'+a+'_novel.png';}}f.oninput=update;document.getElementById('prev').onclick=()=>{{f.value=Math.max(89,+f.value-1);update()}};document.getElementById('next').onclick=()=>{{f.value=Math.min(137,+f.value+1);update()}};document.getElementById('jump').onchange=e=>{{f.value=e.target.value;update()}};</script></main></html>'''
(OUT/'report.html').write_text(report,encoding='utf-8')
print(OUT/'report.html')

"""Build the local static-origin evidence report, without remote dependencies."""
from pathlib import Path
import json
from html import escape
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1]
d = json.loads((OUT/'checkpoint_origin_audit.json').read_text(encoding='utf-8'))
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1220" height="250" viewBox="0 0 1220 250"><rect width="1220" height="250" fill="#f6f8fb"/><defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#50677d"/></marker></defs>']
stages = [('输入与初始化','a37 → dynamic','a6等 → temporal static'),
          ('前10k训练','几何 / RGB / 时间 / 拓扑','已出现删车残留'),
          ('道路修复1k','旧静态本体冻结','固定道路 + 透明度门控'),
          ('Old C2 1k','新车RGB / 小幅偏移','旧层gate + OBB过滤'),
          ('New local3D 1k','仅新车RGB更新','绕过旧C2 gate；改用corridor')]
for i, (title, first, second) in enumerate(stages):
    x = 20+i*240
    color = '#fff0df' if i==1 else '#e9f0f7'
    svg.append(f'<rect x="{x}" y="32" width="220" height="120" rx="9" fill="{color}" stroke="#809ab0"/>')
    for y,text,size in [(65,title,19),(98,first,14),(124,second,14)]:
        svg.append(f'<text x="{x+110}" y="{y}" text-anchor="middle" font-family="Microsoft YaHei,sans-serif" font-size="{size}" fill="#203950">{escape(text)}</text>')
    if i<4: svg.append(f'<path d="M{x+220},90 L{x+237},90" stroke="#50677d" stroke-width="2" marker-end="url(#a)"/>')
svg.append('<text x="610" y="194" text-anchor="middle" font-family="Microsoft YaHei,sans-serif" font-size="18" fill="#9e432b">追溯：初始化表示选择 → 10k扩张谱系 → 后续门控与显露；具体致因操作待分离</text>')
svg.append('<text x="610" y="224" text-anchor="middle" font-family="Microsoft YaHei,sans-serif" font-size="15" fill="#50677d">PVD保持不变；道路先验暂缓；本次只有本地CPU审计，无新训练或GPU渲染</text></svg>')
(OUT/'architecture.svg').write_text(''.join(svg),encoding='utf-8',newline='\n')
font=ImageFont.truetype(r'C:/Windows/Fonts/arial.ttf',20)
canvas=Image.new('RGB',(960,704),'white');draw=ImageDraw.Draw(canvas)
for i,(name,label) in enumerate([('B10k_delete_raw113.png','B10k only / delete a37'),('road_r3_raw113.png','Road r3 / 1k'),('OldC2_raw113.png','Old C2 / 1k'),('Newlocal_raw113.png','New local3D / 1k, road protect 0')]):
    x=(i%2)*480;y=(i//2)*352
    im=Image.open(OUT/'evidence'/name).convert('RGB');assert im.size==(480,320)
    draw.text((x+10,y+4),label,fill='#18384b',font=font);canvas.paste(im,(x,y+32))
canvas.save(OUT/'stage_comparison_raw113.jpg',quality=93)
families=''.join(f'<tr><td>{r["initial_lineage_id"]}</td><td>{r["final_family_rows"]:,}</td><td>{r["responsible_rows"]:,}</td><td>{r["contribution_fraction"]:.2%}</td></tr>' for r in d['top_lineage_families'][:5])
html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>a37旧静态残留起源审计</title>
<style>body{{max-width:1180px;margin:32px auto;padding:0 24px;font:17px/1.8 system-ui,'Microsoft YaHei',sans-serif;color:#213b50}}h1{{font-size:29px}}h2{{font-size:23px;margin-top:32px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d3dee7;padding:10px;text-align:left}}th{{background:#edf3f8}}a{{color:#075e99}}.note{{background:#fff1df;border-left:5px solid #bd8143;padding:14px 20px}}.small{{font-size:14px;color:#526b80}}code{{font-size:14px;background:#eef2f5}}li{{margin:8px 0}}</style>
<h1>scene008 · a37删车：旧静态残留来自哪个阶段？</h1>
<p>2026-11-04 · 本地CPU审计 · PVD保持不变 · 道路先验暂缓 · 两台AutoDL保持关机</p>
<div class="note"><b>结论：旧残留在前10k结束时已经存在；已确认初始化的对象表示选择，以及残留贡献集中于训练中扩张的少数谱系。</b>后续道路和C2不改旧静态本体，但会通过gate和过滤改变旧行显露。当前还不能确定某个坏后代诞生于第几步、哪一种增点操作。</div>
<img src="architecture.svg" alt="输入、10k静态训练、道路gate、OldC2与Newlocal替换链">
<h2>1. 不经过道路修复，原生10k删车就有残留</h2>
<img src="stage_comparison_raw113.jpg" alt="同一raw113的B10k、道路r3、OldC2和Newlocal比较">
<p>左上是原生B10k仅删a37：车后下方已有明显灰黑色残留。原记录明确road_repair_updates=0、R4_or_a6_addon_loaded=false，排除了道路1k和后续PVD作为最早产生阶段。四格展示真实阶段输出，不能当作只改变一个变量的消融。</p>
<h2>2. 训练前：a6进入static是明确的表示契约</h2>
<p>精确运行的identity_contract只把a37列为dynamic；a5/a26/a18/a6交由original B temporal static/background表示。初始化前600,000个场景点经a37的0.1m邻域排除后，static为591,672行；a37 dynamic为98,452行。现成审计匹配到a6的855次同时间场景点匹配，其中798次对应的场景点仍保留在static。</p>
<p><b>因此，“static里出现a6”本身不等于掩码或轨迹bug。</b>这是单a37基线的有意设定；问题是该表达后续如何扩张、混合，并在编辑时替换成新a6表面。a37的0.1m排除仅针对采到的点云邻域，也不证明所有车体表面已从static排干净。</p>
<h2>3. 训练中：残留集中在少数被放大的初始化谱系</h2>
<p>直接读取原生10k checkpoint，static最终共1,520,544行；checkpoint保存了初始化谱系ID。历史raw113、OldC2读出下的固定PVD下部投影域有776像素，统计量为static alpha×T贡献，12,331行static贡献非零；承担其中99%贡献的5,015行仅对应<b>64个初始化谱系</b>。这些旧行的位置与原生10k checkpoint按真实时间计算的位置一致，最大CPU/GPU重算差约0.112毫米。</p>
<table><tr><th>初始化谱系ID</th><th>10k后整族存活行数</th><th>该域责任行数</th><th>该域static贡献占比</th></tr>{families}</table>
<p><b>528407这个谱系的268,093行，是一个初始化高斯的存活后代家族。</b>其中7,757行贡献了该历史诊断域85.50%的static成像。谱系代码在普通clone、空间/时间split和SGCM新增时继承父标签，prune时同步过滤，没有将不同家族合并的逻辑。这支持优先检查局部增密扩张，而非仅检查最终对象标签。</p>
<p>归属曾消耗SGCM clone额度的家族占该域贡献99.56%。但这个flag会标记整个家族，也会被普通增密后代继承；<b>它不意味着99.56%的高斯由SGCM直接创建，更不能证明SGCM单独造成268k扩张。</b>普通clone不受这个SGCM专用flag约束。</p>
<h2>4. 最大谱系可能来自a6观测：强线索，保留来源限定</h2>
<p>找到了同场景早期600k原始种子候选；用精确a37 PLY执行相同排除后也得到591,672行。对584,722个singleton谱系，89.795%的最终raw xyz与候选初始化点变换后的float32坐标逐位相同；错位root对照中位距离21.257米，正确对应中位数为0。这给候选顺序提供强数值支持。</p>
<p>按该候选顺序，528407对应raw108，与同帧a6盒内真实LiDAR返回相距0.222毫米，与同帧a37点云相距5.318米；另一个重要谱系520617也与a6返回高度重合。<b>这是“a6观测种子参与后续残留扩张”的高置信线索。</b>原B10k场景points3d.ply的原始哈希缺失，且OBB内LiDAR不等于精确车体网格，因此还不能把这个对象关联写成逐点语义GT。其余部分重要谱系靠近a37，但不能只凭距离区分车身和道路。</p>
<h2>5. 阶段归因与置信度</h2>
<table><tr><th>阶段</th><th>已有证据</th><th>目前判断</th></tr>
<tr><td>输入/初始化</td><td>仅a37动态；a6有798次同时间匹配保留在static</td><td>表示选择已坐实；并非已查出的误关联bug</td></tr>
<tr><td>10k内拓扑扩张</td><td>591,672→1,520,544行；责任集中在少数初始化家族</td><td>本轮最强的训练期问题线索</td></tr>
<tr><td>SGCM/普通增密具体操作</td><td>配置从500步后进入拓扑阶段；按归档实现SGCM首次550，普通增密首次600</td><td>无后代出生步日志，不能分解各操作责任</td></tr>
<tr><td>后期时间分裂</td><td>原生保存状态static.no_time_split=false；配置/归档实现推导7501解除、从7600增密节点起可分裂</td><td>10k结束时确已允许时间分裂；未证明其导致当前残留</td></tr>
<tr><td>道路1k</td><td>固定道路，训练150,502个旧行gate；旧静态本体不更新</td><td>不是最早来源；道路proxy也未监督隐藏a6</td></tr>
<tr><td>OldC2 1k</td><td>新车RGB/≤3cm偏移/90,369旧行gate，旧本体冻结</td><td>改变旧贡献，未新建这些static行</td></tr>
<tr><td>Newlocal 1k</td><td>仅新车RGB；native模式绕过OldC2 gate，改用corridor</td><td>可能重新显露或过滤旧贡献，不能把所有新洞都追溯为10k几何缺失</td></tr></table>
<p class="small">前述精确步数由运行配置与归档v41实现推导。B10k入口实际引用owner5训练源，未找到其独立源文件哈希，不能将归档实现的每个分支当作逐步执行日志。正式包只有10k checkpoint，未带中间出生记录；原本每100步产生的runtime.jsonl也未进入归档。原生checkpoint本次测得SHA256：{d['checkpoint_sha256']}。后续父检查记录保存的是size/mtime，未保存该SHA，故SHA逐项比对不冒充已完成。</p>
<h2>6. 针对性改进如何控制变量</h2>
<ol><li><b>先补齐出生归因，再修改增密。</b>以同一单a37 B10k配置重放，只增加不消耗RNG的birth-step、parent UID、initial lineage及操作类型记录；保存初始化、500、550、600、1000、3000、7500、7600、10k节点，检查原生删车与保留a6。当前没有运行这次重放。</li>
<li><b>优先检验保留车辆谱系的局部扩张资格。</b>确认具体操作后，再对该操作增加时空/对象观测约束或局部谱系预算；先保持PVD、道路、学习率、相机采样和10k预算不变，不同时改初始化和所有增密。</li>
<li><b>编辑侧只替换发生冲突的局部支撑。</b>268k家族可能还解释其他有效观测，不能整族删除；同时保住已观测a6、道路和标线。新局部表示与旧层过滤必须成套验收。</li></ol>
<div class="note">本次未开启远端、未做新训练或新渲染，PVD与原checkpoint未改。上述85.5%等数字只属于raw113历史OldC2诊断域，不能推广为全49帧或Newlocal raw89红框的逐像素归因。quality_accepted=false，human_verdict=null。</div>
<h2>证据与重放脚本</h2><p><a href="checkpoint_origin_audit.json">原生checkpoint/谱系核对</a> · <a href="candidate_provenance_audit.json">候选初始点关联及限定</a> · <a href="candidate_audit.json">0.1m排除核验</a> · <a href="audit_summary.json">结论机器记录</a> · <a href="source/audit_checkpoint_origin.py">CPU核对脚本</a> · <a href="evidence/B10k_identity_contract.json">原始身份契约</a> · <a href="evidence/B10k_static_seed_audit.json">原始种子审计</a></p></html>'''
(OUT/'index.html').write_text(html,encoding='utf-8',newline='\n')
print('Report built:',OUT/'index.html')


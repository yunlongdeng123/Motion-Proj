"""生成中文报告/49帧浏览器，输出只引用本轮真实结果。"""
import json,sys,html,tarfile
from pathlib import Path
from PIL import Image,ImageDraw
p=Path(sys.argv[1]);r=json.loads((p/'initialization.json').read_text());c=json.loads((p/'coverage.json').read_text());g=json.loads((p/'gpu_r2/receipt.json').read_text());cons=json.loads((p/'color_consensus.json').read_text())
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="240" viewBox="0 0 1120 240"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#72c9ff"/></marker></defs><rect width="1120" height="240" fill="#102039"/><g font-family="sans-serif" font-size="17" text-anchor="middle" fill="white" stroke="#72c9ff"><rect x="15" y="45" width="195" height="75" rx="8"/><rect x="240" y="45" width="195" height="75" rx="8"/><rect x="465" y="45" width="195" height="75" rx="8"/><rect x="690" y="45" width="190" height="75" rx="8"/><rect x="910" y="45" width="195" height="75" rx="8"/></g><g fill="#102039" font-family="sans-serif" font-size="17" text-anchor="middle"><text x="112" y="76">完整199帧</text><text x="112" y="102">RGB / LiDAR / 位姿</text><text x="337" y="76">米制世界对齐</text><text x="337" y="102">地面与遮挡核验</text><text x="562" y="76">固定20cm网格</text><text x="562" y="102">重新建立道路高斯</text><text x="785" y="76">多帧颜色证据</text><text x="785" y="102">来源与未知分开</text><text x="1007" y="76">独立道路 / 编辑合成</text><text x="1007" y="102">真实CUDA贡献</text></g><g stroke="#72c9ff" stroke-width="2" marker-end="url(#a)"><path d="M210 82H237"/><path d="M435 82H462"/><path d="M660 82H687"/><path d="M880 82H907"/></g><text x="560" y="168" fill="white" text-anchor="middle" font-family="sans-serif" font-size="19">固定 V41-B → 原生完整重建；仅编辑输出加入新道路</text><text x="560" y="205" fill="#ffb7b7" text-anchor="middle" font-family="sans-serif" font-size="17">本轮未清除静态污染：中心道路已有支撑，实际合成仍被遮挡</text></svg>'''
(p/'architecture.svg').write_text(svg,encoding='utf-8')
rows=[]
for k,label in [('deduplicated_ground_points','10cm 去重地面点'),('supported','有支撑的20cm道路单元'),('two_triangle_sources','至少两帧几何来源单元'),('color_confirmed','首对颜色一致单元')]:
    rows.append(f"|{label}|{r['variants']['window49'][k]:,}|{r['variants']['full199'][k]:,}|")
metric_table='\n'.join(['|同一CPU初始化口径|49帧|199帧|','|---|---:|---:|']+rows)
pointrows=[];grow=next(x for x in g['rows'] if x['raw']==102)
for pt in c['critical']:
    key=','.join(map(str,pt['pixel']));s=pt['exact_pixel_geometry_witnesses'];v=grow['variants']['full199']['critical'][key]
    pointrows.append(f"|({key})|raw{s[0]['raw']}–{s[-1]['raw']}，{len(s)}帧|{v['road_alpha']:.4f}|{v['actual_road_contribution']:.4f}|")
point_table='\n'.join(['|raw102关键像素|同一世界道路点的新增证据|独立道路alpha|编辑合成实际道路贡献|','|---|---|---:|---:|']+pointrows)
text=f'''# 完整片段道路初始化验证

你的判断得到验证：完整约19.9秒、199帧中存在此前49帧未包含的中心路面观测。本轮重新建立道路高斯，未修改V41-B、R或原初始化。CPU可以完成几何、来源、颜色和初始化；GPU只用于49帧零更新原生渲染核验。

![组件图](architecture.svg)

{metric_table}

两个对照使用同一26,088格候选范围、20cm间距、0.8m几何支撑半径、0.15m地面平面带；不是缩小网格或放宽角度/深度门限来增加数量。原R创建14,034、输出资格1,919；本轮49帧对照的1,624采用额外标注矩形保护、小三角形几何和首对颜色一致性，因此不能将它伪称为旧R逐位复现。源点池49,181与历史49,182相差一格，受本轮局部float32缓存/去重边界影响，支撑单元14,034一致。

{point_table}

两点在49帧对照中的独立alpha及实际贡献均为0。第一点原四角缺支撑，完整片段恢复了支撑与颜色资格。第二点同样恢复了支撑，但首个来源raw149明显较暗，raw150–159较稳定，首对颜色门限仍拒绝其中心单元；相邻有效高斯已产生0.716独立alpha，不能据此说该中心单元已启用。追加的有界分块颜色诊断确认该单元：fit150/151/152/156/157/158，check153/154/155/159；该诊断未替换主对照或用于以上GPU结果。颜色证据支持恢复其资格，暗首源的成因可能包含车辆阴影和投影边缘，未把原因判死。

![同一世界位置的露出帧](review/witnesses.jpg)

## a37与a6：分清屏幕缺口和车辆物理足迹

在49帧固定车辆轮廓内，内缩5像素的核心分母固定：a37为461,346像素帧，a6为19,313像素帧。投影到具备颜色资格单元的核心覆盖：a37从84,531（18.32%）到215,044（46.61%）；a6从0到13,953（72.25%）。这是初始化单元投影覆盖，不是逐像素真值，也不是删车成功率。

另用raw102车辆真实三维框的世界地面足迹检查：a6中心55格中，几何支撑8→55、双源0→55、首对颜色资格0→25。a37物理足迹中心50格原本就都有几何支撑，其首对颜色资格1→2，没有明显改善。它与屏幕车身位置向后相交到的道路区域不是同一分母；不能拿其中一种ROI替另一种宣布全面补全。

![固定世界范围覆盖](review/world_coverage.png)

## GPU复核与限制

固定V41-B原生主通路在49帧插入道路前后逐帧重复输出最大差0；四件基线文件大小/mtime不变，无optimizer、无训练更新。NVML一秒采样峰值{g['nvml_sample_peak_mib']}MiB（约{g['nvml_sample_peak_mib']/1024:.2f}GiB），CUDA allocated {g['cuda_peak_allocated_bytes']/2**30:.2f}GiB，reserved {g['cuda_peak_reserved_bytes']/2**30:.2f}GiB。原生只读渲染耗时{g['elapsed_seconds']:.1f}秒。NVML采样不等于分配器亚秒精确峰值。

新增道路已覆盖关键中心，但原静态污染仍位于它前方：两点的编辑合成道路贡献都是0。不能把点数/独立alpha增长当作干净删车。当前颜色是未训练的来源融合，opacity0.5、切向sigma0.13m、法向sigma0.015m；不是30K新模型，不报告无GT中心PSNR。

完整视频与原始记录raw0/102/198核对，归一化RGB MAE分别0.00416/0.00497/0.00468（视频压缩及缩放差异）；完整原始记录才用于初始化。无外部模型、无生成式伪标签。相机沿用原流程的标定与位姿，未新增畸变/滚动快门修正。新增帧是额外BUILD输入，不能作为独立测试帧；与旧49帧方法比较应明确输入预算变了。

## 对V42的调整

先把完整片段的真实观测作为道路补全强控制；这两个中心已经有证据，不再归入“全片段从未观测”。接下来应把可见性、颜色稳定性、编辑副本的局部静态污染抑制接起来。FreeInpaint保留给全片段核验后仍未知的区域，以及明确撤源的控制；不把已露出中心上生成式结果当作证明外部先验必要。原V41-B完整重建保持独立。

资产：`full199_initialization.npz`包含世界/模型位置、法线、尺度、旋转、opacity、来源RGB、逐帧资格和来源索引；`window49_initialization.npz`为同口径控制。`*_color_consensus.npz`仅为额外诊断。它们是初始化资产，不是可直接替换V41 checkpoint的完整训练状态；现有49帧硬编码接口仍须适配后再做训练smoke。

工程记录：原CPU读取器完成前段后，换用已对11个真实数组逐元素核对的packed-float读取器续接，旧日志保留；第一次GPU读取因torch mmap不接受Path对象中止，修为str后在gpu_r2独立目录成功，旧gpu目录/日志保留。所有尝试训练更新0。quality_accepted=false，human_verdict=null，failure_ledger_refs=[V82-F01,V82-F06]；本轮未新增训练失败类型。
'''
(p/'REPORT.md').write_text(text,encoding='utf-8')
def table(headers,data):return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in data)+'</tbody></table>'
summary_table=table(['同口径初始化','49帧','199帧'],[[label,f"{r['variants']['window49'][k]:,}",f"{r['variants']['full199'][k]:,}"] for k,label in [('deduplicated_ground_points','去重地面点'),('supported','有支撑道路单元'),('two_triangle_sources','双帧几何来源'),('color_confirmed','首对颜色资格')]])
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整片段道路初始化验证</title><style>body{margin:auto;max-width:1400px;padding:30px;background:#0b1525;color:#e6edf7;font:17px/1.7 system-ui}h1{font-size:34px}h2{margin-top:40px}p{max-width:1150px}a{color:#7acfff}img{max-width:100%;height:auto;border-radius:8px}section{padding:22px;background:#14233a;margin:24px 0;border-radius:12px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #40506b;padding:10px;text-align:left}.good{color:#70edb9}.warn{color:#ffcf87}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}select,button,input{font:inherit;padding:6px;background:#20344e;color:white;border:1px solid #789;border-radius:5px}figcaption{font-size:14px;color:#aec6de}figure{margin:0}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><body><h1>完整199帧：道路中心能否补上？</h1><p class="good">能补到两处关键中心。此前只看raw89–137的49帧，漏掉了后面真实露出的道路。</p><p class="warn">但新增道路在最终删车合成中仍被静态残影遮挡。本轮验证初始化及来源，不是新的训练结果。</p><img src="architecture.svg"><section>'''+summary_table+'''<p>固定20cm网格、相同几何门限。1,624是本轮同口径49帧控制，原R输出1,919不与其混为同一资格算法。</p></section><h2>同一世界道路点的真实露出</h2><p>紫色1：(255,240)，raw141–149有9帧证据；青色2：(256,219)，raw149–159有11帧证据。标记随相机运动重投影，不是把屏幕同坐标当同一路面。</p><img src="review/witnesses.jpg"><h2>49帧固定目标域</h2><p>绿色：有颜色资格单元；黄色：有几何支撑但未过颜色资格；红色：缺几何支撑。它是单元投影覆盖，不是稠密真值。</p><label>帧 <select id="raw"></select></label> <button id="prev">上一帧</button> <button id="next">下一帧</button><img id="coverage" alt="同一帧49/199初始化覆盖"><h2>真实CUDA输出：看独立道路与实际合成</h2><p>原生B完整重建始终走原通路；以下道路是未训练初始化。道路贡献图亮度对应实际alpha贡献，黑色表示没有贡献。</p><div class="grid" id="renders"></div><section><h2>中心结果</h2>'''+table(['像素','49帧独立alpha','199帧独立alpha','199帧合成贡献'],[['(255,240)','0','0.8529','0'],['(256,219)','0','0.7161','0']])+'''<p>第二个中心的原始首对颜色不一致；后续来源稳定，附加分块一致性诊断可恢复该单元资格。这项附加诊断没有回写上面的主对照。</p><p>a37屏幕核心覆盖18.32%→46.61%；a6屏幕核心覆盖0→72.25%。这不是删车成功率。固定a6物理足迹核心55格：几何支撑8→55，双源0→55，首对颜色资格0→25。a37物理足迹核心原本就有几何，增益很小，与屏幕向后投射的缺口是不同区域。</p></section><h2>世界网格</h2><p>灰：无支撑；黄：有支撑；绿：颜色资格；紫色边框：车辆物理足迹的固定中心区域。</p><img src="review/world_coverage.png"><section><h2>结论与下一步</h2><p>优先使用完整片段的真实观测，完善颜色稳定性与编辑副本的静态污染处理；生成式补全保留给完整片段仍未知的位置。原B不变，原R与旧初始化全部保留。</p><p>CPU完成初始化。GPU仅做49帧只读前向；训练更新0。NVML峰值4.47GiB，CUDA allocated3.01GiB/reserved4.20GiB。没有代替人工判定质量通过。</p><a href="REPORT.md">完整中文报告、口径与限制</a> · <a href="coverage.json">逐帧来源与覆盖JSON</a> · <a href="gpu_r2/receipt.json">GPU凭据</a> · <a href="color_consensus.json">颜色诊断</a></section><script>const raw=document.getElementById('raw');for(let i=89;i<=137;i++){let o=new Option('raw'+i,i);raw.add(o)}raw.value=102;const channels=[['native_B','固定B：完整重建'],['static_B','固定B：删除动态'],['full199_edit','199帧新道路＋原静态'],['window49_road','49帧：独立道路'],['full199_road','199帧：独立道路'],['full199_contribution','199帧：合成中的实际道路贡献']];function show(){let n=Number(raw.value);document.getElementById('coverage').src='review/coverage_'+n+'.jpg';document.getElementById('renders').innerHTML=channels.map(([ch,label])=>`<figure><img src="gpu_r2/${n}_${ch}.png"><figcaption>${label}</figcaption></figure>`).join('')}raw.onchange=show;document.getElementById('prev').onclick=()=>{raw.value=Math.max(89,Number(raw.value)-1);show()};document.getElementById('next').onclick=()=>{raw.value=Math.min(137,Number(raw.value)+1);show()};show();</script></body></html>'''
(p/'index.html').write_text(page,encoding='utf-8')
# 审阅包包含全部49帧和初始化参数；传感器原始缓存留远端，不伪称离线可重算所有来源。
names=['index.html','REPORT.md','architecture.svg','protocol.json','initialization.json','coverage.json','color_consensus.json','extraction.json','review','gpu_r2',
       'full199_initialization.npz','window49_initialization.npz','full199_color_consensus.npz','window49_color_consensus.npz']
with tarfile.open(p/'delivery.tar.gz','w:gz') as t:
    for name in names:t.add(p/name,arcname=name)
print(json.dumps(dict(delivery=str(p/'delivery.tar.gz'),bytes=(p/'delivery.tar.gz').stat().st_size)))

from pathlib import Path
import json,shutil,re
H=Path(__file__).resolve().parent;D=H/'delivery';O=H.parents[1]/'outputs/scene065_gpu_diagnosis_20261111';O.mkdir(exist_ok=True,parents=True)
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
for p in D.iterdir():
    if p.is_file():shutil.copy2(p,O/p.name)
shutil.copytree(D/'source',O/'source',dirs_exist_ok=True)
for name in ['summarize_limit.py','diagnose_candidate_limit.py','build_report.py']:
    shutil.copy2(H/name,O/'source'/name)
shutil.copy2(H/'review_008.md',O/'independent_review.md')
summary=load(D/'diagnosis_summary_final.json');traces=[load(D/f'{k}_trace_summary.json') for k in [119,198]];gradient=load(D/'gradient_v1_receipt.json')
sel=load(D/'selection_summary.json');metrics=load(D/'joint_metrics.json')
rows=''
for t in traces:
    for region,label in [('target','完整目标轮廓'),('fit','有参考的恢复域'),('keep','保持域 ∩ 目标'),('outside','图表外 ∩ 目标')]:
        r=t['regions'][region];rows+=f"<tr><td>{t['frame_key']}</td><td>{label}</td><td>{r['pixels']:,}</td><td>{100*r['selected']['fraction_of_region_static_alpha_T']:.2f}%</td><td>{100*r['unselected']['fraction_of_region_static_alpha_T']:.2f}%</td></tr>"
grows=''.join(f"<tr><td>{r['frame']}</td><td>{r['cosine_road_vs_target_preserve']:.4f}</td><td>{100*r['gradients']['target_preserve']['increase_opacity_L1_fraction']:.1f}%</td><td>{100*r['gradients']['target_left']['increase_opacity_L1_fraction']:.1f}%</td></tr>" for r in gradient['records'])
proxy=''.join(f"<tr><td>{r['raw']}</td><td>{dict(gates_off_road_off='不加道路／不做交接',gates_off='加道路／不做交接',road_off='做交接／不加道路',final='已发布：道路＋交接')[r['mode']]}</td><td>{r['regions']['real']['pixels']}</td><td>{r['regions']['real']['proxy_real_reference_linear_MAE']:.5f}</td></tr>" for r in metrics['records'] if r['raw'] in [119,198] and r['mode']!='road_only')
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="236" viewBox="0 0 1160 236"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#89accd"/></marker></defs><rect width="1160" height="236" rx="12" fill="#112031"/><g fill="#1a3048" stroke="#456981"><rect x="20" y="72" width="180" height="92" rx="9"/><rect x="240" y="72" width="180" height="92" rx="9"/><rect x="460" y="72" width="190" height="92" rx="9"/><rect x="690" y="24" width="210" height="74" rx="9"/><rect x="690" y="140" width="210" height="74" rx="9"/><rect x="940" y="72" width="200" height="92" rx="9"/></g><g fill="none" stroke="#89accd" stroke-width="2" marker-end="url(#a)"><path d="M200 118h38"/><path d="M420 118h38"/><path d="M650 118h18V61h20"/><path d="M668 118v59h20"/><path d="M900 61h20v57h18"/><path d="M900 177h20v-59"/></g><g fill="#eaf2fb" font-family="Microsoft YaHei, sans-serif" text-anchor="middle" font-size="18"><text x="110" y="109">冻结30K场景</text><text x="110" y="139" font-size="15">归属／道路／交接</text><text x="330" y="109">指定白车删除</text><text x="330" y="139" font-size="15">原始权重只读</text><text x="555" y="109">一次联合深度排序</text><text x="555" y="139" font-size="15">分层颜色精确加和</text><text x="795" y="54">逐行贡献追踪</text><text x="795" y="79" font-size="14">199帧重算候选与排除原因</text><text x="795" y="170">固定状态梯度与开关</text><text x="795" y="195" font-size="14">0参数更新</text><text x="1040" y="109">定位恢复受限环节</text><text x="1040" y="139" font-size="15">保持冲突／覆盖／参考</text></g></svg>'''
(O/'architecture.svg').write_text(svg,encoding='utf-8')
doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>白车删除：GPU分层与交接诊断</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0b111b;color:#e6eff8;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1260px;padding:28px 24px 70px;margin:auto}}h1{{font-size:28px;line-height:1.35;margin:10px 0 16px}}h2{{font-size:21px;margin:30px 0 12px}}p{{max-width:1080px}}a{{color:#80c9ff}}img{{display:block;width:100%;height:auto;border-radius:8px}}figure{{margin:15px 0}}figcaption,.muted{{font-size:14px;color:#a9bbce}}.lead{{background:#142337;border-left:3px solid #e5b368;padding:13px 18px}}.toolbar{{display:flex;align-items:center;flex-wrap:wrap;gap:14px;margin:18px 0}}button,select{{font:inherit;background:#172b42;border:1px solid #3b5976;border-radius:6px;color:#eaf1f8;padding:6px 10px}}input[type=range]{{width:230px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;padding:9px 12px;border-bottom:1px solid #2d3c4e}}th{{background:#142235}}.scroll{{overflow:auto}}details{{border:1px solid #2c4057;border-radius:8px;padding:12px 16px;margin-top:18px}}summary{{cursor:pointer}}code{{color:#b6d6f4;font-size:13px}}.findings{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}}.findings>div{{padding:15px;background:#122033;border:1px solid #293f58;border-radius:8px}}.findings b{{display:block;color:#f4cf96}}@media(max-width:700px){{.findings{{grid-template-columns:1fr}}}}</style><main>
<p class="muted">scene-065 · 自车前方白色轿车 · GPU诊断 · 2026-11-11</p><h1>残影来自其他场景层；恢复约束与交接范围同时受限</h1><p class="lead">已完成真实GPU对照并实际看图。静态分层颜色中仍有完整车形和彩条；目标车组已退出。008的保持规则漏迁移有实际梯度冲突，但只改这条规则仍控制不到恢复域外的大量残影高斯。</p>
<div class="findings"><div><b>保持约束冲突</b>后段三帧中，目标保持项与道路恢复项梯度反向。</div><div><b>控制范围缺口</b>图表外约76%的静态贡献来自未选候选，改损失不会让这些行参与交接。</div><div><b>道路参考不足</b>第119帧仅529个目标像素有现有实拍参考，不能把未知区域全部填成道路。</div></div>
<figure><img src="architecture.svg" alt="冻结场景、指定删除、一次联合深度渲染、逐行与梯度诊断的数据流"></figure>
<h2>同一深度排序下的真实颜色分层</h2><div class="toolbar"><label>锚帧 <select id="frame"><option>60</option><option selected>119</option><option>150</option><option>198</option></select></label><input id="slider" aria-label="切换四个诊断锚帧" type="range" min="0" max="3" value="1"><label>查看 <select id="view"><option value="layers">分层颜色</option><option value="controls">道路／交接开关</option></select></label><a id="original" href="119_layers.png">原尺寸</a></div>
<figure><img id="panel" src="119_layers.png" alt="FRONT 第119帧，同排序颜色分层"><figcaption id="caption">上排：已发布删除、静态颜色、动态颜色；下排：道路颜色、环境项。全部为同次排序中的直接贡献，曝光一致；各项相加还原最终RGB。</figcaption></figure>
<p>右侧白色皮卡和前方黑色皮卡应保留。动态层中可能包含合法显露的邻车，不能统一当作待删白车。静态／动态是表示分支；未知身份不等于目标车辆。</p>
<h2>候选行追踪：恢复域内和域外是两种情况</h2><p>逐帧复算了全部199帧的候选选择，45,449个行号与原结果逐项一致。再用真实光栅的标记梯度追踪每条静态高斯在目标像素上的贡献。下表百分比的分母为各区域内的静态alpha×T；区域可重叠，不能相加。</p>
<div class="scroll"><table><thead><tr><th>帧</th><th>目标内区域</th><th>像素数</th><th>已选行贡献</th><th>未选行贡献</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>恢复域内超过99.94%的静态贡献已经进入候选；保持域冲突有实际作用路径。完整目标域中，因恢复贡献不足而未选的静态贡献为38.03%／36.92%（119／198帧），保护条件再排除1.64%／1.70%。仅因正实例身份筛选而排除的贡献为0，不能把这个额外条件当作本例主因。</p>
<h2>梯度对照：保持项确实在拉回旧残影</h2><p>固定最终交接参数，不创建优化器、不执行更新，分别对道路恢复、目标保持和目标范围外保持求导。负余弦表示两项局部下降方向相反。增加不透明度会让旧高斯继续遮挡后方表面。</p>
<table><thead><tr><th>帧</th><th>道路恢复 vs 目标保持梯度余弦</th><th>目标保持：增加不透明度的梯度L1份额</th><th>额外保持：同口径份额</th></tr></thead><tbody>{grows}</tbody></table>
<p>第119、150、198帧存在明显冲突，第60帧不明显。该证据确认固定状态下的机制，不能替代修正规则后的完整配对实验，也不能推断修复全部残影的比例。</p>
<h2>把所有已选候选强制退出，也没有得到干净背景</h2><figure><img src="119_candidate_limit.png" alt="已发布删除、所有原候选静态高斯退出的极端诊断和差异"><figcaption>第119帧极端控制：仅在私有渲染行上将原交接候选的不透明度置0。上部残影仍在，下部暴露动态碎片和异常环境。它移除了合法背景成分，不能作为恢复方案或正式删除结果。</figcaption></figure>
<details><summary>第198帧同一控制</summary><figure><img src="198_candidate_limit.png" alt="第198帧所有候选退出的诊断"></figure></details>
<p>“仅道路＋环境”开关也不是纯道路图：道路未覆盖处会显露原生环境贴图，出现彩色纹理与车形暗块。最终合成中的道路贡献仍很局部。不能通过扩大透明区域来宣称补全正确。</p>
<details><summary>已有道路参考域中的数值对照</summary><table><thead><tr><th>帧</th><th>私有行开关</th><th>现有实拍参考像素</th><th>线性RGB代理MAE</th></tr></thead><tbody>{proxy}</tbody></table><p class="muted">这些参考参与BUILD，来自现有平面、语义和可见性规则，不是独立无车真值。道路和交接均带来局部数值收益，但实际图像仍有车形，因此不判恢复通过。</p></details>
<h2>下一项应验证什么</h2><p>先在通用代码中恢复008已经采用的完整目标域保持规则，再分别核验候选覆盖和可显露表面支持。保持规则修正、候选扩展和表面补全应分别记录；右侧皮卡、前方皮卡、标线与未编辑区域保留独立对照。不能把全部未知静态行改归目标，也不能把目标轮廓后的内容一律指定为道路。</p>
<details><summary>运行与核验记录</summary><p>四个锚帧，全部199帧仍参与原生训练，无留出。本轮227次光栅前向、0训练更新、0优化器；分层颜色加和最大差{summary['same_sort_RGB_sum_maxabs']:.2g}，四帧已发布RGB和四帧原语义删除RGB均精确复现。候选IDs完全一致，但累计浮点fit／guard得分非逐位一致（最大绝对差{sel['selected_fit_score_maxabs_vs_saved']:.2g}／{sel['selected_guard_score_maxabs_vs_saved']:.2g}），未改变筛选集合。</p><p>原生检查点元数据和各场景参数只读检查通过。正式编辑代码、已发布页面、scene-029及人工verdict均未改。本报告检查了原始PNG、数值守恒与静态文件链接；未冒称浏览器UI验收。AutoDL保持开机。</p><p><a href="diagnosis_summary_final.json">运行汇总</a> · <a href="selection_summary.json">全帧候选复算</a> · <a href="gradient_v1_receipt.json">梯度原始记录</a> · <a href="119_trace_summary.json">119逐行贡献汇总</a> · <a href="198_trace_summary.json">198逐行贡献汇总</a> · <a href="independent_review.md">sol6.1 max独立复核</a> · <a href="../scene065_edit_diagnosis_20261109/index.html">上一轮CPU版本与覆盖审计</a></p><p>task <code>WAYMO-065-GPU-DIAGNOSIS-20261111</code>，沿用失败记录V82-F06。远端目录 <code>campaign/waymo_val065_delete30k_gpu_diagnosis_20261111_r1</code>；完整浮点图、全量选择得分与逐行权重留在远端。</p></details>
</main><script>
const keys=[60,119,150,198],frame=document.getElementById('frame'),slider=document.getElementById('slider'),view=document.getElementById('view'),panel=document.getElementById('panel'),original=document.getElementById('original'),caption=document.getElementById('caption');
function draw(){{const raw=Number(frame.value),mode=view.value,path=raw+'_'+mode+'.png';slider.value=keys.indexOf(raw);panel.src=path;panel.alt='FRONT 第'+raw+'帧 '+(mode==='layers'?'同排序颜色分层':'只读开关对照');original.href=path;caption.textContent=mode==='layers'?'上排：已发布删除、静态颜色、动态颜色；下排：道路颜色、环境项。全部为同次排序中的直接贡献，曝光一致。':'上排：原语义删除、仅加道路、仅做交接；下排：已发布的道路＋交接、仅道路＋环境。开关会重新排序，差值不能当独立层颜色；这些是诊断图。';}}
frame.addEventListener('change',draw);view.addEventListener('change',draw);slider.addEventListener('input',()=>{{frame.value=keys[Number(slider.value)];draw();}});draw();
</script></html>'''
(O/'index.html').write_text(doc,encoding='utf-8')
for ref in re.findall(r'(?:src|href)="([^"]+)"',doc):
    if not ref.startswith(('#','http')):assert (O/ref).is_file(),ref
for k in [60,119,150,198]:
    for mode in ['layers','controls']:assert (O/f'{k}_{mode}.png').is_file()
from PIL import Image
for p in O.glob('*.png'):
    with Image.open(p) as im:im.load();assert im.mode=='RGB'
(O/'README.md').write_text('''# scene-065 白色轿车删除：GPU诊断

task=WAYMO-065-GPU-DIAGNOSIS-20261111。用户授权继续GPU诊断、sol6.1 max独立复核；AutoDL保持开机。入口index.html，简单模块图architecture.svg。065原生30K用户通过、029未通过的状态保留；本轮未改正式代码或发布新的编辑质量结果。

冻结原30K模型、归属、道路与交接，四帧60/119/150/198；20次同排序RGB／四格开关，4次局部损失梯度，199次候选复算＋2次逐行最终贡献，另2次极端候选退出，共227次光栅前向，0更新、0优化器。每个成功模块时间含加载与输出，不是CUDA核时间。RTX4080 SUPER，显存峰值约3.22GB，单线程CPU整理。短程零优化诊断，不适用小时级训练smoke；未追加训练、重建或删车正式预算。

颜色分层加和最大差1.311302185e-6；已发布与原语义删除四帧RGB差0。全199候选IDs完全复现，selected score因浮点累计非逐位一致；数值差与集合一致性分开记录。225次基础诊断delivery.tar.gz已取回，另2次极端控制的图、receipt、源码补取回，完整NPZ、全量score、逐行权重留远端。所有原PNG解码与报告静态引用检查通过；未进行浏览器UI验收。

结论：静态RGB本身携带车形；fit内超过99.94%的静态贡献已入候选，keep/left局部梯度与恢复项冲突；fit外约76%静态贡献不在候选。仅因positive-owner的额外排除在所测两帧贡献0，非主因。极端退出所有原候选仍有上部车形与下部动态／环境污染，不能作正式方法。参考不足、保持冲突、候选覆盖须分开验证。alphaT和代理MAE不当隐藏真值或修复成功。

源码见source，独立复核independent_review.md。GPU调用与参数只读检查见各receipt。failure_ledger_refs=[V82-F06]；failure_ledger_delta=updated:V82-F06；human_verdict=null。未关机。
''',encoding='utf-8')
print(O)

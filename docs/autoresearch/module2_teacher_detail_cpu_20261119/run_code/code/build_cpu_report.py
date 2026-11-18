from pathlib import Path
import json,html,shutil
W=Path(__file__).resolve().parent;R=W.parents[1];O=R/'outputs/road_teacher_detail_cpu_20261119_r1';O.mkdir(parents=True,exist_ok=True)
p=json.loads((W/'protocol.json').read_text(encoding='utf-8'))
shutil.copy2(W/'protocol.json',O/'protocol.json');shutil.copy2(W/'bound_recipe.json',O/'bound_recipe.json');shutil.copy2(W/'architecture.svg',O/'architecture.svg')
arms=[['高斯均色','原观测许可','零'],['高斯均色','观测＋教师许可','零'],['逐像素世界材质','原观测许可','零'],['逐像素世界材质','观测＋教师许可','零'],['逐像素世界材质','观测＋教师许可','TRELLIS.2部分投影HF']]
summary=dict(schema='teacher_detail_CPU_preparation_final.v1',task_id=p['task_id'],status='CPU_PREPARED_STOP_BEFORE_GPU',
    assessment='原来源账本不能直接充当教师细节学习许可；固定高斯均色再混合有额外细节损失风险，需与逐像素世界材质进行实际native对照',
    common_rows=89887,common_high_bound=.34,common_low_bound=.30,train_frames=[86,110],diagnostic_frames=[134,179,198],
    new_199_frame_run=False,actual_scene_optimizer_updates=0,native_forward_calls=0,native_backward_calls=0,
    new_model_inferences=0,GPU_operations=0,power_operations=0,pending_GPU_controller=False,
    CPU_evidence=p['CPU_evidence_summary'],CPU_module_tests=dict(root=11,permission=9,pixel=10,T2=7,total=37),
    actual_native_scope_basis_gradients_K_and_persistent_reload='not executed; mandatory future smoke',
    GPU_authorized=False,training_authorized=False,production_promoted=False,human_verdict=None,problem_solved=None,
    remote='/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1',
    evidence=dict(permission='evidence/permission/final_r1/recipe.json',old_permission='evidence/permission/old_audit_r1/receipt.json',
        average='evidence/permission/numeric31_r1/receipt.json',average_cache='evidence/permission/avg_sparse_gh31_r1/upload_verified.json',
        pixel='evidence/pixel_render/preparation_receipt.json',pixel_cache='evidence/pixel_render/read_operator_box_r1/verification.json',
        T2='evidence/trellis_hf/seed_entry.json',local_copy='evidence/local_copy_verified.json'))
(O/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
matrix='| 读取方式 | 高频许可 | B阶段H初始化 |\n|---|---|---|\n'+'\n'.join('| '+' | '.join(r)+' |'for r in arms)
md=f'''# 教师细节蒸馏：CPU准备与实验协议

已完成CPU数据、算子和实验准备，停在GPU执行前。主问题是教师细节能否进入世界材质，以及颜色平均是否消耗了本可复现的细节；不增加训练步数、不训练SD1.5或TRELLIS模型。

![组件与数据流](architecture.svg)

## 评估结果

1. 原始材质来源与训练许可承担不同职责。旧C域47886个成像贡献高斯中25103个足迹高频门完全为0；S域6707个中170个为0，不能概括成所有阴影都没有高频能力。按保存的行最大成像贡献加权，C/S平均许可为0.06675/0.23132；这是行级代理，不是逐像素或联合可见贡献。
2. 原±0.04幅度也限制细节。可信教师HF与基础HF残差的p99为C0.09683、S0.15835；超过0.04的比例约3.70%/36.13%。本轮在GPU结果之前按两域p99较大值与实际0.5高通系数，统一锁定五组H界±0.34。它不是全部纹理都可复现的证明。
3. 高斯均色与逐像素读取需要在同几何/覆盖下对照。逐像素方案为新的世界材质着色器：从真实G_s交点与像素足迹查询同一材质场，再用原生GS实际eligible权重合成；不使用整张road alpha后贴图，不修改只读高斯颜色贡献。
4. TRELLIS的棕偏在原PNG中已有，未发现可证UV翻转、通道或尺度错误。旧GH高分辨积分问题已独立修复。本次只复用旧推理的带限HF初始化，保留底色、去影教师和相同损失；编码中的过度侵蚀、混合边界与网格重建语义问题已修正并保留负例。

## 共同数据与参数

真实A、孔洞伪教师C、阴影伪教师S严格分域。训练raw86/110，诊断raw134/179/198；199帧仅继承BUILD，本轮未重跑199帧、不称盲测。

原source/known/unknown/观测RGB账本不改。独立teacher evidence决定新增高频许可；比较HF/结构冲突，底色RGB差只做诊断。新teacherH允许34269世界节点，其中25378旧门为0；147处HF冲突不新开教师路径。单有效来源不冒称双视角一致，原C/S损失与评估分母均保留。

所有组共用89887个候选可写高斯（原84266全保留＋5621新资格）、相同基础材质、同latent/halo/low mask及正则归一化。只有1267个合格缺底色节点填入冻结12cm低通教师底色，不偷渡教师HF。候选scope仍需真实native资格检查；无几何、无教师或受保护位置不因旧unknown标签而自动放行。

低频12cm、高频2cm；L界±0.30，H界±0.34；H仍采用0.5b(tanh−Gaussian6cm(tanh))及幅度/能量/DC/低频泄漏约束。H界按 `ceil((2*max(p99_C,p99_S)+4/255)/0.01)*0.01=0.34` 在CPU阶段锁定。相同geometry、normal、scale、opacity、active、相机、静态场、车辆、扩散/阴影/特征模型全部冻结。

## 实验矩阵

{matrix}

每组仍为A低频200步 → B冻结低频/训练高频300步 → C联合300步。每种reader复用同一A200完整前缀；主四组2800物理更新，次要T2再600，合计最多3400（各组逻辑800）。同seed101、400/400采样轨迹、A+0.5C+S预乘L1与相同正则；T2只替换B入口H初值，A0/low/A200/AdamH零状态均不改，无额外prior loss。步数相同不代表GPU算力相同，实际速度留smoke测量。

旧P的±0.04/84266范围与独立RGB结果仅作历史参照，不能把新旧收益全归于单独修改许可；本轮2×2内部条件一致，分别识别许可和读取方式的影响。

## CPU验证与修复

- 9项真实数据梯度测试覆盖source0 unknown、source3 generated、observed S、retrieved S及冲突/保护位置；合格孔洞H可以产生梯度，真未知/教师冲突/保留位置不被隐式放行。
- 新幅度下GH15/GH21未过原数值门，确定性升为GH31；150真实行四类场与独立GH61抽查通过，门限未放宽。完整89887×961物理查询已准备，真实道路坐标反查误差≤9.09e-13m。
- 固定valid/mH/几何时，读取可严格写为 `B+K_L L+K_H H`；K_H已包含分别插值的valid和mH，不得再次乘门。平均读取缓存与同积分颜色最大差1.33e-15、CPU梯度差0。它是静态算子降本，仍只训练世界网格，不增加逐高斯或逐帧RGB参数。
- 五帧全部122604 canonical像素的逐轴自适应box积分已编译，共6093288点，无cap16、无缩域；代表性density2/4差p99=0.00131，缓存/原积分差1.81e-14。训练A/C/S/fullS全部有真实G_s交点；C/S材料质量均非0。A有7像素无材料支持，保留原eligible-ci回退，完整A16904与qualifiedA16897并列，不隐藏只读floor。
- 原D38416中218个几何fallback单列，不把拟合平面当真实G_s。准备仅使用旧缓存的几何字段，没有读取ideal RGB或理想标线作为真实监督。
- chunk分组后的pixel ID必须散射回图像；平均缓存ID是完整拓扑global world row，必须映到active位置。新增运行时防止误reshape或混用ID导致条纹/错位。

37项模块CPU测试通过，最终发布源另做隔离验收。CPU证据不能替代真实native权重、CUDA稀疏反传或最终视觉验收。

## TRELLIS次要对照的边界

复用同一次旧推理，零新推理/零模型训练。先按物理尺度带限到2cm网格，再提取6cm高通；仅对缺少可信干净材料的合法教师恢复域准备初始化，不用原known/source标签一刀切。纯已观测/纯域外受到约束，连续混合边界单列。

最终可实现HF初始化RMS0.003348、最大0.02；纯域外节点最大1.11e-5、quarter最大2.17e-5，基础材质保持。它只保留目标能量58.05%、cos0.828，未通过原cos≥0.9忠实传递门；作为**部分投影HF初始化探索对照**有效，不代表充分测试TRELLIS材质容量。主四组不依赖它，负结果不能否定TRELLIS后训练潜力。

节点目标、source、许可和均值固定；CPU编码的数值迭代及失败现场单独记账，未进行场景优化。局部先验节点的DC绝对值最大约3.11e-5、低频泄漏RMS约0.001115，不能声称严格正交或完全零低频。

## GPU开启后的执行顺序

先做真实短程smoke：同scope的eligible RGB＋unit weight导出、road/joint常量场、CUDA sparse前向/反向、未知孔洞H梯度、7A回退分层、全五帧K77266、保存/评价/完整Adam-RNG-sampler恢复、下一步/B/C入口与持久资产重查。若实测队列预计小时级，按项目规则完成约10分钟生命周期smoke，工程开销独立记账。

联合图的权重必须是真实同排序alpha×透射率；road-only K位置可有被车遮住的道路贡献，只有joint K是硬保护门。RGB/alpha/静态/车辆/原来源台账均须核验。新的pixel资产保存共享世界场、renderer_mode和geometry/scope；五帧权重缓存只是派生数据，新视角/编辑请求必须重新取原生权重与真实路面映射，不能只烘焙平均RGB后声称保留pixel细节。

smoke通过后再依次执行四组主对照及T2次要对照。分别评价教师拟合、可信细节、诊断条纹、跨视角、真实A保护、道路覆盖、联合可见贡献及持久化；不以单个loss或CPU理想图替代最终联合渲染。

## 当前停止点与入口

本轮场景优化0、native forward/backward 0、新模型推理0、GPU执行0、开关机操作0；没有待启动控制器，默认场景未改。GPU执行合同仍为false，human_verdict/problem_solved=null。

- 远端：`{summary['remote']}`
- 共同输入：`permission/final_r1/teacher_evidence.npz` 与 `recipe.json`
- 平均算子：`permission/avg_sparse_gh31_r1/operator`
- 像素算子：`pixel_render/read_operator_box_r1/{{086,110,134,179,198}}`
- T2初始化：`trellis_hf/seed_r5_nodal_b034/H_seed.npz::H_seed_latent`
- 检查入口：`CUDA_VISIBLE_DEVICES= /root/autodl-tmp/envs/motionproj/bin/python code/gpu_entry_guard.py --contract execution_contract.json --phase inspect`
- 完整协议：`protocol.json`；硬件状态与native结果不在本轮CPU结论内。
'''
(O/'report.md').write_text(md,encoding='utf-8')
rows=''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>'for c in r)+'</tr>'for r in arms)
table=f'<div class="scroll"><table><tr><th>读取方式</th><th>高频许可</th><th>B阶段H初值</th></tr>{rows}</table></div>'
labels=dict(permission='教师许可与共同输入',old_permission='原高频门审计',average='均色数值检验',average_cache='均色缓存核验',pixel='逐像素准备',pixel_cache='逐像素缓存核验',T2='高频初始化',local_copy='取回核验')
source=''.join(f'<li><a href="{html.escape(v)}">{html.escape(labels[k])}</a></li>'for k,v in summary['evidence'].items())
page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>教师细节蒸馏 · CPU准备</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0a111b;color:#e7eef8;font:16px/1.65 "Microsoft YaHei",system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:28px}h1{font-size:30px;line-height:1.4}h2{font-size:21px;margin:0 0 12px}p{margin:9px 0}.tag{color:#71d7bf;letter-spacing:.05em;font-size:13px}.muted{color:#adc1d5}.note{border-left:3px solid #e9bd70;padding:10px 14px;background:#1b2839}section,.card{background:#121e2d;border:1px solid #2c435d;border-radius:12px}section{padding:20px;margin:20px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.card{padding:16px}.number{font-size:27px;font-weight:650;color:#73d9c1}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #31465f;text-align:left;padding:11px}th{font-weight:500;color:#b1c7dc}a{color:#91c7ff}img{width:100%;display:block}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:20px}.small{font-size:13px}.evidence{columns:2;list-style:none;padding:0}code{font-family:ui-monospace,monospace;overflow-wrap:anywhere}footer{font-size:13px;color:#a6bbcf}@media(max-width:760px){main{padding:14px}.cards{grid-template-columns:repeat(2,1fr)}.twocol{grid-template-columns:1fr}h1{font-size:25px}section{padding:14px}.arch img{min-width:1000px}.evidence{columns:1}}</style></head><body><main>
<div class="tag">模块2 · CPU准备完成 · GPU未执行</div><h1>让二维教师细节进入持久世界材质</h1><p>拆开原始来源与教师许可，分别验证高频学习能力和颜色读取方式。主要实验采用固定道路几何及现有二维教师。</p>
<p class="note">当前没有GPU结果。每组仍计划800步，未训练生成模型、未替换默认资产；本轮GPU入口保持关闭。</p>
<div class="cards"><div class="card"><div class="number">4＋1</div>主对照＋次要初始化对照</div><div class="card"><div class="number">89,887</div>共同候选可写高斯</div><div class="card"><div class="number">±0.34</div>统一高频幅度界</div><div class="card"><div class="number">0</div>本轮GPU／场景训练</div></div>
<section><h2>组件与数据流</h2><div class="scroll arch"><img src="architecture.svg" alt="教师证据、共享材质、两种读取与原生高斯合成组件"></div></section>
<section><h2>评估结论</h2><div class="twocol"><div><p><b>许可限制已确认。</b>孔洞相关47,886个贡献高斯中25,103个足迹高频门完全为0；阴影相关仅170／6,707为0。新许可来自合法教师证据，原source／known／unknown账本保持。</p><p>RGB底色差与HF冲突分开。34,269个世界节点获得教师H许可，其中25,378个旧门为0；原C／S监督分母未缩减。</p></div><div><p><b>均色损失风险进入直接对照。</b>像素方案查询真实路面世界颜色，并按原生eligible权重合成；保留其余高斯贡献，不能用整张道路alpha替代联合可见性。</p><p>原±0.04也截掉部分教师细节。±0.34由CPU幅度审计预先确定，五组一致；旧结果只作历史参照。</p></div></div></section>
<section><h2>固定条件的实验矩阵</h2>TABLE<p class="muted small">低频200 → 高频300 → 联合300；每组逻辑800，按reader共用A200。主四组2800物理更新，次要组追加600。实际GPU时间留smoke测量；无额外prior loss。</p><p>raw86／110训练；raw134／179／198诊断。199帧仅继承BUILD，不是本轮盲测。</p></section>
<section><h2>CPU已验证与GPU待验收</h2><div class="twocol"><div><p><b>CPU已完成</b></p><p>真实孔洞H梯度；GH31＋GH61抽查；全五帧609万点的无上限像素积分；两种读取的稀疏缓存与原采样／梯度等值。</p><p>C／S均有真实路面交点和可用材质。真实A的7个只读回退像素单列，A16904与qualifiedA16897同时报告。</p></div><div><p><b>GPU开启后先验收</b></p><p>真实native eligible RGB＋unit权重、CUDA sparse反传、joint K77266、完整状态恢复与下一阶段、权威材质保存重载和新视角权重更新。</p><p>CPU理想图和源码接口检查不替代这些实际路径。smoke通过后才执行正式对照。</p></div></div></section>
<section><h2>TRELLIS.2仅提供部分高频初始化</h2><p>复用旧推理，不替换底色与去影教师。最终HF RMS约0.00335、最大0.02；基础材质不变，纯已观测／纯域外受保护，混合边界单列。</p><p class="note">保留目标能量58.05%，cos0.828，未通过忠实传递门。它可作部分投影初始化探索，主四组不依赖它；不能用负结果否定TRELLIS后训练能力。</p><p class="muted small">原棕偏在模型PNG已有，未证实UV／通道错误；本轮修正初始化的支持窗、混合边界和实际网格重建语义。</p></section>
<section><h2>协议与复核</h2><p><a href="protocol.json">冻结实验协议</a> · <a href="bound_recipe.json">幅度选择</a> · <a href="report.md">技术报告</a> · <a href="results.json">机器摘要</a> · <a href="finish_receipt.json">收尾记录</a></p><ul class="evidence">EVIDENCE</ul><p class="small muted">完整输入、稀疏算子、物理查询和原资产留远端。本目录为轻量报告与证据，不是完整资产备份。</p></section><footer>GPU授权=false · 无待启动队列 · human_verdict／problem_solved=null</footer></main></body></html>'''
(O/'index.html').write_text(page.replace('TABLE',table).replace('EVIDENCE',source),encoding='utf-8')
print(json.dumps(dict(report=str(O),GPU_authorized=False,scene_updates=0)))

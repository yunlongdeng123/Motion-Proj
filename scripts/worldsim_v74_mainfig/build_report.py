import pathlib,json,shutil,html,zipfile
B=pathlib.Path(__file__).resolve().parent;O=B.parents[1]/'outputs/V74_Main_Paper_Figures';O.mkdir(exist_ok=True);E=O/'evidence';E.mkdir(exist_ok=True)
A=json.loads((B/'evidence/aggregate.json').read_text());S=json.loads((B/'secondary/summary.json').read_text());W=json.loads((B/'white_witness/audit.json').read_text());owner='204704542f8642dc8ab046ffbd70e0c5'
names={'vggt':'官方 VGGT','omega512':'VGGT-Ω 512','dvgt1':'DVGT-1','pi3x':'Pi3X','dvgt2':'DVGT-2','dggt':'DGGT 深度头','nksr':'NKSR','noksr':'NoKSR'}
def triplet(c):return f"{c['hit']} / {c['early']} / {c['near_vertices']}"
white=[]
for m in ['vggt','omega512','dvgt1','pi3x','dvgt2','dggt']:
 cells=[]
 for v in ['six','twelve_common6','twelve']:
  if m in ['dvgt2','dggt']:r=next(r for s in S['grid_summaries'] if s['method']==m and s['variant']==v and s['scene']=='scene-0520' for r in s['rows'] if r['owner']==owner and r.get('protocol')=='cal_build_1.0')
  else:r=next(r for r in A['per_actor'] if r['method']==m and r['variant']==v and r['owner']==owner and r['protocol']=='cal_build_1.0')
  cells.append(triplet(r['counts']))
 white.append('| '+names[m]+' | '+' | '.join(cells)+' |')
native=[]
for m in ['nksr','noksr']:
 if m not in S:continue
 rows=S[m]['rows'];r=next(x for x in rows if x['owner']==owner);c=r.get('counts')
 native.append(f"| {names[m]} | {S[m]['evaluable']}/75 | {triplet(c) if c else r['status']} | {c['early_with_later_correct'] if c else '—'} |")
macro=[]
for m in ['vggt','omega512','dvgt1','pi3x']:
 d=next(t for t in A['table'] if t['method']==m and t['variant']=='twelve' and t['protocol']=='cal_build_1.0')['equal_log_mean'];macro.append(f"| {names[m]} | {d['hit']*100:.2f}% | {d['early']*100:.2f}% | {d['near_vertices']*100:.2f}% |")
noksr_state='已完成原生推理与 QUERY 评价' if 'noksr' in S else '官方 Carla Serial 权重已下载，运行依赖准备中；没有方法级结论'
text=f'''# V7.4：从重建表面到物理首回波的证据与主图

任务：`WS-V74-MAINFIG-01 / 20260915-first-return-r1`；扩展：`WS-V74-MAINFIG-SECONDARY-01 / 20260915-secondary-r1`。日期：2026-09-15。单 RTX 3090，零训练，人工 verdict=null。

**当前能支持的结论：多个官方几何模型经明确的普通尺度、标定与表面读出后，仍会产生与实测首回波冲突的较早交点；原生表面重建参照也可能在存在后方正确交点时提前返回。现有证据不能证明“所有 SOTA 普遍凭空生成假几何”，也不能证明 false-safe 或闭环事故。**

建议将主张收窄为：**表面点覆盖、图像重建质量与物理首回波正确性是不同目标。** 需要把几何、尺度、表面合并、材质透射与传感器响应逐项分开。下面保留成功案例和普通控制带来的改善，不以更极端筛选维持原主张。

![组件图](figures/architecture_components.png)

## 1. Reviewer 先看这三张图

1. [共同低位置回波：真实 RGB → 四模型早交三角形 → 同一束距离 → occupancy](figures/fig06_shared_low_return_witness.pdf)。这是四模型共同冲突的实测示例，明确标为车底附近的回波。
2. [原始 0.323 m 射线的反例](figures/fig08_legacy_beam_counterexample.pdf)。同一束车门附近回波，四个首批官方模型全部恢复到预先定义的 Hit 容差内。旧案例不能直接迁移成官方 VGGT 的失败。
3. [输入观测数量与输出表面数量控制](figures/fig03_white_car_evidence_vs_surface_count.pdf)。DVGT-1、Pi3X 在共同六张输出图上增加输入证据后同时提高 Hit、降低 Early；合入更多表面才增加 Early。这反对“补全必然以 Early 为代价”的笼统说法。

完整图版还包括跨场景红色早交叠加、逐日志分母、置信度控制、DGGT 原生 Gaussian 渲染，以及第二批原生重建参照。所有红色标记来自已保存的实际三角形或交点；没有使用生成式绘图补造场景或结果。

## 2. 实际执行范围与输入信息

| 方法 | 本轮结果 | 信息 / 输出边界 |
|---|---|---|
| 官方 VGGT | 6 场景 × 6/12 图，共 12 次官方前向 | RGB 输入；官方原始权重。与旧微调 DPT＋LiDAR 融合系统分开 |
| VGGT-Ω | 12 次；官方结构 strict=True 加载通过 | 用户指定 `1kaiser/vggt-omega-jax` 镜像的原始 512 权重，4576706117 bytes；不是 416 reproduction。不能独立认证镜像发布者身份，训练集重叠未排除 |
| DVGT-1 | 12 次；严格加载完整权重 | 原生 metric ego_0 点图，训练单位系数 0.1 |
| Pi3X | 12 次；严格加载后采用官方 RGB-only 路径 | 原生局部/全局点图与相机；本轮没有向模型输入 LiDAR |
| DVGT-2 | 12 次；严格加载完整网络，包括 ego head | 因果几何模型输出 ego_n 点图，按各时刻第一相机对应 ego 位姿变换；checkpoint 内锚点缓冲恢复初始化文件，再完整严格加载。没有开展规划评测 |
| DGGT | 12 次前向、12 组完整输入视图 Gaussian 渲染 | 官方 nuScenes 权重；72+? 视图见 evidence 中实际记录。渲染使用官方 mode-2 方程和 sky 模型，以模型预测 sky mask 替代数据集 GT mask。输入视图重建，不是 NVS 基准 |
| NKSR | 34 个原生网格；33 个有 QUERY 可评价 | 官方 ks，额外 BUILD LiDAR＋PCA16 法向；保留原生三角形，无面数截断 |
| NoKSR | {noksr_state} | 官方 Carla Serial；同一 BUILD 点与法向；与纯 RGB 的信息预算不同。采用官方 enable_flash=False 的 FP32 attention 路径，完整严格加载权重 |

首批完成 48 次前向、48 组七项协议评价及 24 组共同六图输出控制。第二批 DVGT-2 / DGGT 完成 24 次前向和 36 组评价。DGGT 共渲染 **108 张输入视图**（6 场景 × (6+12)），同时保存 RGB、透明度与期望深度。

**DGGT 不是一份独立的几何失败票数。** 本轮全部 12 个窗口中，它的深度和相机姿态与官方 VGGT 数值完全一致（最大绝对差均为 0）。原生 Gaussian 渲染提供的是不同输出路径的证据，不能把重复深度结果当成第六个独立基础几何模型。

NKSR / NoKSR 使用四个 BUILD 时刻的 LiDAR 点与已知刚体轨迹，信息多于两时刻 RGB。PCA16 需要至少 16 个点：75 个对象中 34 个满足；41 个保留为输入不足。33 个满足输入且有 QUERY，另一个没有 QUERY。不能把无法运行对象记成 0 Early 或 100% 正确。

## 3. 冻结的数据与读出定义

全部数据来自已曝光的旧 DEV：6 场景、5 日志、75 对象。52 对象有非歧义、框关联 QUERY 回波，23 没有，保持未定义。每模型 11886 束 QUERY。没有消费封存 reserve，没有新日志独立确认，也没有训练。

RGB 输入：BUILD sample3 的六环视，或 sample3+4 的十二图；间隔约 0.5 秒。QUERY sample2/5 从未进入模型、尺度估计或网格构建。BUILD-only 点/法向导出与原生重建求解器分开存储；评价器独立加载 QUERY。

**“对象所属”是 3D 标注框外扩 0.1 m 后的空间关联，并排除了重叠歧义，不是语义实例表面标签。** 因而可包含车底道路回波、透射回波或框内其他结构。不能把所有这些点自动称为车身真值。速度≤0.5 m/s 和到 BUILD 点≤0.20 m 的支持分层分别报告，未知速度保持未知。

统一硬表面诊断：原生相机局部点图 / 深度 → 已知相机与目标位姿 → 图像邻接三角化。固定断边阈值为 max(0.15 m, 0.05×最小相机深度)，目标范围外扩 1 m；没有跨大空洞强行搭桥。已知内参重投影、普通尺度和置信度过滤分别设控制。它是公开的诊断适配器，不能冠名为各模型官方 LiDAR simulator。

主协议 `cal_build_1.0`：已知内参、相机/目标轨迹，加单个全局 BUILD 背景 LiDAR 尺度；尺度只用共同 sample3 六图中 `diagnostic_mask AND owner为空`、1–80 m 的锚点，取 z/pred_z 中位数。没有逐对象或 QUERY 拟合。此协议是“视觉＋度量锚点＋位姿标注”的诊断，不能当纯视觉同信息排行榜。VGGT/Ω 的相机基线尺度自身不稳定，因此本轮无法给出可信的无锚点纯视觉米制排名。

固定置信保留率 100/90/75/50%。`native_base`、`native_build`、`cal_base` 与主协议一起保留。这里 native 指保留模型局部点坐标；仍使用已知相机/目标位姿，不等于原生端到端仿真。

射线单位方向固定，t_pred 为最早正向三角交点，r_GT 为实测首回波距离。Hit：|t_pred−r_GT|≤0.20 m；Early：t_pred<r_GT−0.20 m；Late：t_pred>r_GT+0.20 m；MISS：无交点。另报 0.1/0.3/0.5 m Early。四类互斥完备。

“表面覆盖”在本报告严格称为 **0.20 m 表面顶点邻域召回**：QUERY 点到任一预测顶点足够近。它受顶点密度影响，不是网格连续表面 completeness 的无偏估计。额外统计任意正确交点，以及最早交点错误但后方仍有正确交点。

## 4. 白车配对结果：保留反例与普通控制

每格为 **Hit / Early / 顶点邻域召回数量**，同一分母 752；都采用主协议。

| 方法 | 6 输入 / 6 输出图 | 12 输入 / 共同 6 输出图 | 12 输入 / 12 输出图 |
|---|---|---|---|
{chr(10).join(white)}

共同六图控制是发现后的普通诊断，不是独立确认。DVGT-1 从 256/25 到 300/23，Pi3X 从 458/39 到 464/37，说明更多真实输入证据可以同时改善命中与 Early。其后并入额外输出面，Early 才增加。对固定十二图预测，十二输出面的集合包含前六输出面，因此最早交点向前移动是集合求交的自然结果；不能据此单独声称模型内部的补全机制必然产生 phantom。

额外 LiDAR 信息的原生重建参照（与上表不同信息预算）：

| 方法 | 可评价对象 / 全分母 | 白车 Hit / Early / 顶点召回，N=752 | 白车 Early 且后方仍有正确交点 |
|---|---|---|---|
{chr(10).join(native)}

NKSR 沿用已记录的空层级网格提取兼容修复；不改变学习隐式场或权重。实现、版本和 NoKSR attention 后端见 `evidence/model_manifest.json`。

NKSR 白车 Early 中 95/111 有 BUILD 近邻支持；0.5 m 阈值仍 56 束。它有真实学习隐式场和原生双网格提取，不依赖图像邻接三角化。但 PCA 法向、输入稀疏、材质、刚体关联等仍可能解释部分异常，不能全部称为模型幻觉。

NoKSR 白车为 Hit458、Early80、Late50、MISS164，顶点邻域召回715/752；80 束 Early 中 60 束有 BUILD 近邻支持，38 束后方仍有正确交点，0.5 m 阈值仍30束。两个原生方法都出现冲突，但它们使用额外点云与估计法向，不能据此推出纯视觉基础模型具有同一种内在生成缺陷。

## 5. 同一束共同 Early 的审计

白车 scene-0520，owner `{owner}`，QUERY sample2、ray243。实测 r_GT=8.564013 m。四模型主协议提前量：VGGT **0.457686 m**、Ω **0.412555 m**、DVGT-1 **0.482968 m**、Pi3X **0.666080 m**。这些是实际求交数值，不是画图设定的偏移。

四模型早交三角形最长边分别约 3.29 / 2.27 / 11.52 / 1.98 cm；每个三角形的全部顶点都在该 GT 射线终点前方，排除仅由跨巨大空洞连线导致所选交点的解释。50% 置信保留后，这一束仍保持相同的较早交点。四模型共有 14 束 BUILD-supported Early，双精度独立求交的最大差小于 1.7e-6 m，全部仍为 Early。

**重要边界：ray243 的 GT 点在标注框底部以下约 2.65 cm；14 束共同 Early 中 9 束位于框底附近或更低。** 主图因此使用“低位置/车底附近的实测回波”，没有写“击中真实车身”。余下 5 束位于框内、约 z=0.05 m，可能涉及透射/内部回波；尚无逐束材质标签或车身表面可靠性确认。相机与 LiDAR 存在约 32 ms 时间差，RGB 投影只用于定位解释，不能充当新的几何真值。

该射线也并非通过所有控制：DVGT-1 在不加 BUILD 尺度的 `cal_base` 下提前约 0.190 m，属于 Hit；Pi3X 的无 BUILD 尺度控制为 MISS。完整协议表保留在 evidence，不能只报告最有利于主张的控制。

原始 legacy ray382 位于较高的侧面位置：旧系统早交 0.322696 m；四个官方模型主协议分别为约 −0.020 m（稍晚）、0.117 m、0.098 m、0.142 m（稍早），均为 Hit。这是明确保留的成功案例。NKSR 同一束约早交 0.303 m，属于不同信息/方法路径。

## 6. 跨日志结果及适用范围

十二输入 / 十二输出图，主协议，5 日志等权均值：

| 方法 | Hit | Early | 顶点邻域召回 |
|---|---|---|---|
{chr(10).join(macro)}

各日志射线数：0919=169，1089=74，0519+0520=10922，0359=25，0048=696。两场景来自同一日志，不能算独立日志。0359 仅 25 束，最高 BUILD 支持的静态候选也缺少 sample3 对应位姿，不适合作为主视觉案例；统计仍完整保留。图中展示 4 个不同日志的真实 RGB，包括树遮挡、夜间模糊和围栏等现实条件，没有把所有场景自动称为“干净可见表面”。

这些是固定发现窗口的发生情况，而非驾驶总体上的普遍失效率。置信过滤通常降低 Early，同时损失覆盖；不能从本轮曲线断言任何普通阈值都无效，更不能跨不同模型置信标度直接比较同一阈值。

## 7. 对传感器、occupancy 和规划的因果链

**已测量**：保存的重建表面在某束实测首回波之前被硬射线击中 → 该硬表面读出的模拟 LiDAR 提前返回 → 该束逆传感器模型的 occupied 终点向传感器方向移动，终点之前 free，之后保持 UNKNOWN。

图中以 5 cm 单束射线单元演示 occupancy 终点差异，全部由同一束实测距离与预测交点计算。它不是完整多束融合 BEV benchmark。低位置回波若被完整占用系统判为道路，未必直接产生障碍物，因此不能从低位置 witness 自动推导“车体障碍物移位”或碰撞。

**需要单独验证**：不同波长/材质的透射与反射、透明度/阈值规则、传感器束宽、时序同步、真实实例表面归属、多束 free/occupied 融合、物体感知与闭环规划。Early 可能造成假障碍、保守制动或遮蔽后方真实结构；这些是潜在机制。本轮没有测得 false-safe、碰撞率、闭环驾驶失败率，也没有运行规划器。

DGGT 的 Gaussian expected depth 是透明度加权相机深度。它能生成看似完整的 RGB，但不应直接硬化成三角网格后再把首交误差归为官方 Gaussian simulator 的错误。当前原生渲染是输出证据，物理 LiDAR 响应仍未定义。

## 8. 推进判定与停止条件

这轮已经形成可用的主图素材和可复算的发现证据。**建议采用“重建质量与首回波一致性的缺口”作为探索问题；暂不采用“VGGT 及所有升级 SOTA 普遍凭空造假几何并导致 false-safe”作为论文结论。**

若进一步推进，最有价值的有限补证是：冻结具体射线失败定义，在新日志中先确认干净可见、非透射目标及参考可靠性；再检验尺度/标定控制后的残余，并比较原生材质感知的传感器读出。必须把车底/透射机制与实体前方假几何分开。新日志数量和数据权限需按后续任务单独登记，本轮未消费 reserve。

若可靠实体表面上的残余被普通控制消除，或主要由材料透射与 opaque 假设解释，应降低“生成假几何”的优先级，转向几何与传感器接口问题；不通过不断极端筛选或增加诊断版本强迫正结论。原始 440 项 V8.1 任务没有重跑。

## 9. 复现、资源和验证

远端证据根：`/root/autodl-tmp/runs/worldsim_v74_h2/WS-V74-MAINFIG-01/20260915-first-return-r1`。原生输出与权重保留在仓库外；代码入口在 `scripts/worldsim_v74_mainfig/`。脚本和注册文件随交付保存。

主预测目录 `predictions/`；正式主评价 `evaluation_bgscale/`；共同六图控制 `evaluation_common6/`；第二批在 `secondary/`。早期错误背景掩码生成的部分 `evaluation/` 仅作历史保留，不进入任何最终表图。修正使用冻结的原生输出，没有重跑主模型。

验证包括：所有完整权重严格加载、解析双平面最早/后方交点、Hit/Early/Late/MISS 互斥完整计数、多个 Early 阈值嵌套、共同 14 束 float64 复算、逐图视觉检查。对真实代码输出做数值比较，不增加文件 hash/checksum。单卡最大片段显存约 12 GiB 量级，推理和缓存完成后无需多卡扩容；未训练、未关机、未创建自动调度。

failure_ledger_refs：V73-F03/F09、V74-F06、V74-H2-F12；failure_ledger_delta：V74-H2-F13。人工 verdict 保持 null。

来源：[官方 VGGT](https://github.com/facebookresearch/vggt)、[VGGT-Ω](https://github.com/facebookresearch/vggt-omega)、[用户指定 Ω 512 镜像](https://huggingface.co/1kaiser/vggt-omega-jax/blob/main/vggt_omega_1b_512.pt)、[DVGT-1/2](https://github.com/wzzheng/DVGT)、[Pi3X](https://github.com/yyfz/Pi3)、[DGGT](https://github.com/xiaomi-research/dggt)、[NKSR](https://github.com/nv-tlabs/NKSR)、[NoKSR](https://github.com/theialab/noksr)、[Open3D raycasting](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html)。
'''.replace('72+? 视图见 evidence 中实际记录。','108 张视图。')
(O/'REPORT.md').write_bytes(text.encode('utf-8'))
captions={
'fig06_shared_low_return_witness':('Shared first-return conflict in official geometry models','The same measured heldout beam (8.564 m) intersects reconstructed triangles 0.413–0.666 m earlier in four official geometry models under the declared calibration and BUILD-scale diagnostic. RGB locates the target and low-return region; the side profile and range bars expose the otherwise nearly coincident image projections. The GT endpoint lies 2.65 cm below the annotated box bottom and is not asserted to be a car-body return. A single-beam inverse sensor model moves the occupied endpoint toward the sensor and leaves the occluded interval unknown. No closed-loop or false-safe outcome is measured.'),
'fig08_legacy_beam_counterexample':('A successful control retained from the original badcase','The original legacy beam is 0.323 m early in an adapted VGGT–LiDAR fusion system. All four primary official models are within the fixed ±0.20 m Hit band on this same beam under the primary diagnostic. Native LiDAR reconstruction references use additional information and are shown separately. This retained counterexample prevents transferring a legacy system failure to current official models.'),
'fig03_white_car_evidence_vs_surface_count':('Input evidence and output surface count are separate interventions','For the same 752 heldout returns, increasing RGB inputs from six to twelve is compared at a common six-map readout and at the full twelve-map readout. DVGT-1 and Pi3X improve Hit while reducing Early in the common-map control. Their additional Early errors appear after unioning more output surfaces. Since unioning a fixed superset can only move the first intersection closer, this plot does not establish an inevitable intrinsic completion trade-off. The common-map control is post-hoc discovery analysis.'),
'fig02_cross_scene_official_models':('Official-model first-return diagnostics across scene contexts','Rows show metadata-selected near-static targets from four distinct discovery logs, plus the original white target. All columns use the same RGB crop. Red polygons are actual triangles causing >0.20 m early intersections on BUILD-supported heldout returns. Counts retain the full object denominator. Empty red regions and difficult visibility conditions are retained; 3D-box association does not certify semantic car-body ownership. Omega uses the user-provided original-512 mirror, not the 416 reproduction checkpoint.'),
'fig04_cross_log_early_rates':('Cross-log rates with explicit denominators','Early-return rates use the same fixed calibrated-depth grid and global BUILD-background scale. All five discovery logs are retained, including the weak-reference log with only 25 rays. Two scenes belong to one log. These rates describe the exposed discovery window, not population prevalence or an independent confirmatory benchmark.'),
'fig05_confidence_controls':('Ordinary confidence filtering','Per-view confidence retention is fixed at 100, 90, 75 and 50 percent. Each curve uses identical calibration and scale, with equal-log aggregation. Filtering reduces Early but also removes surface-point coverage. All settings are shown; no threshold is selected to maximize a failure claim. Coverage is vertex-neighborhood recall and depends on sampling density.'),
'fig07_dggt_native_rendering':('DGGT native Gaussian reconstruction and depth','Official nuScenes weights reconstruct the input views through Gaussian RGB+expected-depth rasterization and the official sky model, with a predicted semantic sky mask replacing the dataset GT mask. Depth colors share a scale within each row. This is an input-view reconstruction check, not novel-view synthesis or physical LiDAR evaluation. All twelve depth/pose predictions exactly match VGGT, so the depth-grid results do not constitute an independent geometry failure.'),
'fig09_secondary_cross_scene':('Second-batch reconstruction references with explicit information budgets','DVGT-2 RGB predictions use the declared calibrated-depth grid. NKSR and NoKSR receive additional BUILD LiDAR and PCA normals and retain their native learned surfaces. Red marks actual early-intersection triangles on BUILD-supported heldout rays. The same metadata-selected objects and full QUERY denominators are retained. Missing method inputs are not counted as successful predictions.'),
'fig01_legacy_white_car_story':('Legacy adapted-system example, retained as historical context','Actual RGB, projected early-hit triangles, BEV rays, a metric range zoom and a single-beam occupancy diagnostic show a 0.323 m early intersection. The source is an adapted VGGT DPT plus LiDAR fusion surface, not official VGGT. The endpoint shift is measured; false-safe, obstacle-level BEV performance and closed-loop planning are not evaluated.'),
'architecture_components':('Components and information flow','BUILD RGB enters frozen official geometry models. Saved native predictions feed a declared surface adapter and controlled first-hit raycasting. Known cameras and actor poses plus BUILD-only metric anchors are extra information; QUERY LiDAR is used only by the evaluator. Occupancy is a single-beam diagnostic and closed-loop planning remains untested. NKSR/NoKSR and DGGT rendering follow separately declared input and representation contracts.')}
(O/'CAPTIONS.md').write_bytes(('# Figure captions\n\n'+'\n\n'.join(f"## {name}\n\n**{title}.** {cap}" for name,(title,cap) in captions.items())).encode())
for src,name in [(B/'evidence/aggregate.json','primary_aggregate.json'),(B/'secondary/summary.json','secondary_summary.json'),(B/'white_witness/audit.json','shared_ray_audit.json'),(B/'evidence/white_float64_qa.json','float64_qa.json'),(B/'evidence/cohort.json','cohort.json')]:shutil.copy2(src,E/name)
for name in ['final_qa.json','omega_status.json','dggt_vggt_geometry_comparison.json','registration.json','omega512_registration.json','common6_registration.json','evaluation_amendment.json','readout_qa.json','model_manifest.json','noksr_config.yaml']:
 if (B/'evidence'/name).exists():shutil.copy2(B/'evidence'/name,E/name)
cards=[]
for name,(title,cap) in captions.items():
 if not (O/'figures'/f'{name}.png').exists():continue
 cards.append(f'<article id="{name}"><h2>{html.escape(title)}</h2><p>{html.escape(cap)}</p><a href="figures/{name}.pdf">PDF</a> · <a href="figures/{name}.svg">SVG</a> · <a href="figures/{name}.png">PNG</a><img loading="lazy" src="figures/{name}.png" alt="{html.escape(title)}"></article>')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7.4 Main Paper Figures</title><style>body{margin:0;background:#f4f6f8;color:#172b46;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1250px;margin:auto;padding:36px 24px}h1{font-size:32px;line-height:1.25}h2{font-size:22px}a{color:#28649b}header,article{background:white;border-radius:12px;padding:26px;margin-bottom:26px;border:1px solid #e0e6ed}article img{display:block;width:100%;height:auto;margin-top:22px}.lead{background:#eaf3ee;padding:18px;border-left:4px solid #008577}.limit{background:#fff3df;padding:16px}nav{display:flex;gap:18px;flex-wrap:wrap}</style><main><header><p>V7.4 · 单 RTX 3090 · 冻结模型 · 无训练</p><h1>从重建表面到物理首回波</h1><p class="lead">已找到多个官方模型共同的首回波冲突，并完成真实 RGB、距离与 occupancy 因果链图。当前最稳妥的研究问题是：重建覆盖和视觉质量，何时不等于物理首回波正确性？</p><p class="limit">旧 0.323 m 射线已被四个首批官方模型修复；共同冲突集中于车底和可能透射区域。现有证据不能证明所有 SOTA 凭空生成假几何，也没有证明 false-safe 或闭环事故。成功案例和反向证据全部保留。</p><nav><a href="REPORT.md">中文完整报告</a><a href="CAPTIONS.md">English captions</a><a href="evidence/primary_aggregate.json">主实验数据</a><a href="evidence/secondary_summary.json">第二批数据</a></nav></header>'''+''.join(cards)+'</main></html>'
(O/'index.html').write_bytes(page.encode());print('report and gallery generated',O,flush=True)

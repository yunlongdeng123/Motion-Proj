"""保存第九里程碑的有限补证；不将接口准备计作模型实测。"""
import json,re,shutil
from pathlib import Path
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/Simulation_Impact_Research';E=W/'evidence_fresh_baseline/evidence';S=W/'milestone9_stage';D=S/'docs';M=D/'autoresearch/worldsim_simimpact/milestone9'
M.mkdir(parents=True,exist_ok=True);shutil.copytree(E,M/'evidence',dirs_exist_ok=True);(M/'figures').mkdir(exist_ok=True);shutil.copy2(O/'F21_Fresh_Real_Baseline_Control.svg',M/'figures/F21_Fresh_Real_Baseline_Control.svg')
b=json.loads((E/'fit_admission.json').read_text());r=json.loads((E/'fit_admission_route.json').read_text());br=b['rows'][0];rr=r['rows'][0]
evidence={'milestone':9,'task_id':'WS-SIM-FRESH-NATIVE-01','run_id':'20260915-r1','fixed_command':b,'route_control':r,'new_policy_forwards':16,'new_vehicle_forecasts':16,'cumulative_policy_vehicle_forecasts':818,'cumulative_CenterPoint_forwards':222,'cumulative_geometry_forwards':72,'cumulative_feedback_loops':{'HUGSIM':18,'SplatAD':4},'new_scene_fits':0,'fresh_source_count':1,'source_training_split':'nuScenes train / train_detect','failure_ledger_delta':'V74-H2-F22','human_verdict':None,'goal_status':'active'}
(O/'FRESH_BASELINE_EVIDENCE.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8');(M/'evidence/report_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
table='\n'.join(f"| {i:02d} | {x['ADE_vs_recorded_ego_m']:.3f} | {y['ADE_vs_recorded_ego_m']:.3f} | {x['FDE_vs_recorded_ego_m']:.3f} | {y['FDE_vs_recorded_ego_m']:.3f} | {int(x['actor_overlap_any'])} → {int(y['actor_overlap_any'])} |" for i,(x,y) in enumerate(zip(br['individual_outcomes'],rr['individual_outcomes'])))
report=f'''# 新来源真实驾驶基线：一次路线控制后仍未准入

**结论：不启动 scene-0002 的 SplatAD 拟合。** 真实传感器上的固定直行基线平均 ADE/FDE 为 1.667/4.642 米；加入一次冻结的路线提示后为 1.422/3.727 米，八起点框交叠从 1 次降为 0。两者均未通过实验前规定的平均 ADE≤1 米、FDE≤2 米、无交叠条件。这是规划基线的适配限制，不能当作重建坏例。

![完整基线组件与八起点](F21_Fresh_Real_Baseline_Control.png)

```mermaid
flowchart LR
    A[相同真实 RGB 与 LiDAR] --> C[冻结 TransFuser]
    B[固定直行 / 真值路线提示] --> C
    C --> D[官方 PDM 车辆跟踪]
    D --> E[与真实轨迹和演员框对比]
    E --> F[未通过：停止新场景拟合]
```

## 冻结范围与来源

task/run：`WS-SIM-FRESH-NATIVE-01 / 20260915-r1`。先在挂载数据 shard 的 scene-0001…0100 有限窗口中做元数据筛查，排除已曝光仿真/主图场景所在的整个日志；按场景和时间顺序选择，最多六个不同日志，不为凑数放宽标准。要求足够未来评价窗口、初始速度 2–15m/s、前方 4–30m 且侧向±6m 内的车辆/行人至少八个 LiDAR 点、九关键帧持续存在、记录车体间距0.8–5m，并出现可观测的相对横向变化或减速。

只有 scene-0002 满足条件，日志 `6b6513e6c8384cec88775cae30b78c0e`，初始索引3。目标车辆初始前向4.04m、侧向−5.89m、687个点，记录最小框间距3.15m。完整六相机和LiDAR的1770个原始文件已提取，缺失0。此处“新来源”仅表示此前本轮仿真/主图未评价；它在官方 nuScenes `train` 集中，**不是模型训练未见或封存确认集**。

八个评价起点对应原始索引3…10。策略只接当前真实传感器与由当前/过去两个相机记录计算的速度、加速度、角速度。PDM 初始转角使用相同车辆轴距的自行车关系，评价参考完整覆盖4秒；这些修正均发生在首次基线推理前。旧实验不重跑，缺少角速度字段的旧输入仍沿用零初值。

## 一次普通路线控制

作者提供的 NAVSIM/HUGSIM 客户端 `hugsim/dataparser.py` 固定设置 `command[1]=1`，忽略传入路线，而本日志持续左转。第一次失败结果落盘后，只增加一次预先固定的输入控制：沿记录的空间路线向前20m，以相对侧向偏移±2.5m决定左/直/右；八起点均得到左转。无备选阈值、无最佳指令搜索。

这条提示从记录真值路线导出，属于**额外路线信息**。未来速度、数值轨迹、演员框和传感器仍不进入策略；不能把该条件称为纯视觉同信息优势。输入传感器、权重、状态、预处理和车辆执行均保持配对。原始失败准入文件完整保留，另写路线控制结果。

| 起点 | 固定 ADE(m) | 路线 ADE(m) | 固定 FDE(m) | 路线 FDE(m) | 交叠 |
|---|---:|---:|---:|---:|---:|
{table}
| 平均 / 次数 | {br['mean_ADE_m']:.3f} | {rr['mean_ADE_m']:.3f} | {br['mean_FDE_m']:.3f} | {rr['mean_FDE_m']:.3f} | 1/8 → 0/8 |

两次条件下，记录自车用相同车辆参数重放均无框交叠。这里的交叠只是演员框几何诊断，未计算地图合法性、事故责任或完整 PDM 分数。16次策略前向及16次4秒车辆预测不是传感器反馈闭环；没有新几何推理或重建干预。

## 下一条接口与停止规则

停止本来源的 TransFuser 指令/阈值试探与昂贵场景拟合。BridgeSim 的代码核查发现其 `pdm_closed_adapter.py` 自称 PDM-lite、使用特权状态和自定义规则；runtime ray LiDAR 入口是二维射线，另有可选 PointCloudLidar，尚未验证其重建资产读出。不能将其直接当作原版 PDM-Closed 与物理3D首交的实测。[作者代码](https://github.com/VAIL-UCLA/BridgeSim)

下一步只准备一个原生 nuScenes 规划器：官方 SparseDrive-S stage2，代码版本 `ec0225d4b7a2dd7e6ce10179a2b7660dcb74b2f1`，权重由作者 release 下载；SparseDriveV2 公开主线为 NAVSIM，本轮不并铺。六相机输入、标定、时序记忆、原生三秒路线提示和评价已冻结；两帧预热＋八个原评价起点，未运行模型。该原生路线提示仍来自真值，训练集重叠保持明示。CAN 采用因果消息；源码审查显示官方推理用自身预测状态，CAN字段主要为训练目标，不能宣称它在推理中起效。[SparseDrive 官方发布](https://github.com/swc-17/SparseDrive)，[SparseDriveV2 官方范围](https://github.com/swc-17/SparseDriveV2)

SparseDrive 的三秒原生轨迹指标将独立报告，不能与这里四秒 PDM 误差混排。先检验真实图像基线与输入契约，才决定重建传感器评价是否有意义。当前没有可晋级主论文的新增严重几何因果坏例；已有Ω局部定位恢复与普通删除同效的结论不变。

## 证据与执行

完整数字见 `FRESH_BASELINE_EVIDENCE.json`；远端 `/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FRESH-NATIVE-01/20260915-r1` 保存两次注册、全16轨迹、输入、参考及准入结果。`docs/autoresearch/worldsim_simimpact/milestone9/evidence` 归档轻量证据；图以PNG/PDF/SVG三格式输出。

本轮新增16次策略/PDM配对，累计818；CenterPoint累计222、几何前向72、HUGSIM18＋SplatAD4次反馈闭环不变；新场景拟合0。单RTX3090，代码/权重准备正在继续，SparseDrive前向未计数。绘图已检查，所有八起点与摘要逐项对应。整体goal active；人工verdict=null；failure_ledger_delta=`V74-H2-F22`。
'''
(O/'FRESH_BASELINE_CLOSEOUT.md').write_text(report,encoding='utf-8')
remote_report=report.replace('(F21_Fresh_Real_Baseline_Control.png)','(autoresearch/worldsim_simimpact/milestone9/figures/F21_Fresh_Real_Baseline_Control.svg)')
(D/'WORLDSIM_SIMULATION_FRESH_BASELINE.md').write_text(remote_report,encoding='utf-8')
note='''# 代码核查更新：新来源基线未通过，准备原生nuScenes规划器（2026-09-15）

BridgeSim版本 `eb727f87918c6bc34de82adfb66620b2b953cd61` 已下载并核查源码，未安装、未运行。`bridgesim/evaluation/models/pdm_closed_adapter.py` 明确标作PDM-lite：特权状态、候选路径、IDM与自行车模型、自定义评分；其模块说明以GT未来中心线为路线，另有live lane/fallback分支，不能称所有路径都固定用GT。它不是未经改动的原版PDM-Closed，也没有自动消费本轮CenterPoint检测框。

`evaluation/utils/lidar_utils.py` 的 `ray_lidar_to_ego_points` 把二维射线距离分数转成恒定高度点。`evaluation/core/environment_manager.py` 另有独立可选PointCloudLidar入口，因此不能声称BridgeSim完全不支持3D；其真实扫描、重建资产和物理首交契约仍待验证。本轮不把README的“runtime LiDAR”直接当作已完成3D几何因果接口。[官方源码](https://github.com/VAIL-UCLA/BridgeSim)

新来源scene-0002的真实TransFuser/PDM基线和一次额外路线提示均未通过固定准入：平均ADE/FDE 1.667/4.642m→1.422/3.727m，框交叠1/8→0/8；不拟合该场景，不将基线域适配问题写成重建危害。已冻结官方SparseDrive-S stage2的两帧预热＋八起点原生六相机输入；运行环境和公开权重准备中，尚无新模型结果。其三秒规划须独立于四秒PDM评价，scene-0002训练集重叠和GT路线信息明确保留。[官方SparseDrive](https://github.com/swc-17/SparseDrive)

```mermaid
flowchart LR
    A[真实六相机与标定] --> B[官方SparseDrive-S]
    R[原生路线提示：额外真值] --> B
    B --> C[感知与运动预测]
    C --> D[原生三秒规划]
    D --> E[先与真实日志比较]
    E -. 基线可靠后 .-> F[同接口重建渲染与局部资产干预]
```

以下为此前接口复核，保留历史范围。

'''
p=D/'WORLDSIM_SIMULATION_PLANNER_INTERFACE_REVIEW.md';old=p.read_text(encoding='utf-8');p.write_text(note+old,encoding='utf-8');(O/'PLANNER_INTERFACE_REVIEW.md').write_text(note+old,encoding='utf-8')
summary='''# 当前：新来源真实基线与一次路线控制完成，停止该场景拟合（2026-09-15）

WS-SIM-FRESH-NATIVE-01 / 20260915-r1：有限新日志窗口仅scene-0002达元数据条件，1770原始文件完整。八真实起点＋八路线提示对照共16策略/PDM配对：ADE/FDE 1.667/4.642m→1.422/3.727m，框交叠1/8→0/8；均未通过固定ADE≤1m、FDE≤2m准入，不启动新SplatAD拟合。路线提示是额外GT空间路线信息，真实基线残余不能归因重建。scene-0002是本轮新来源但属于模型训练集，非封存确认。

BridgeSim代码核查为PDM-lite及二维runtime ray LiDAR，另有独立可选3D PointCloudLidar；尚未运行。准备官方原生nuScenes SparseDrive-S stage2，两预热＋八起点已冻结、权重/环境准备中，尚未前向；不把接口准备计为科学结果。累计818策略/PDM、222 CenterPoint、72几何前向、18HUGSIM＋4SplatAD反馈闭环。goal active；人工verdict=null；failure_ledger_delta=V74-H2-F22。见docs/WORLDSIM_SIMULATION_FRESH_BASELINE.md。以下为历史。

'''
# 历史打包器不再发布全局状态；结果保存在本轮报告/证据中。
failure=D/'research_failures/entries/V74-H2-F22.md';failure.parent.mkdir(parents=True,exist_ok=True);failure.write_text('''# V74-H2-F22：新来源的真实驾驶基线未达到几何归因条件

- task/run：WS-SIM-FRESH-NATIVE-01 / 20260915-r1。
- 类型：下游基线与域/路线接口限制；不是重建科学失败。
- 冻结范围：scene-0002，八起点，两条件，16官方TransFuser/PDM；输入/权重配对，非反馈闭环。
- 观察：固定直行ADE/FDE1.667/4.642m、交叠1/8；一次GT路线提示后1.422/3.727m、0/8。真实记录车体同参无交叠。仍未满足ADE≤1m、FDE≤2m及无交叠准入。
- 控制：路线由20m空间前瞻与±2.5m规则预先冻结，额外信息明示，不挑指令、不重跑旧矩阵。
- 停止：不继续调该来源TransFuser，不启动新场景拟合，不拿基线误差作重建危害。
- 开放：原生nuScenes六相机规划器能否给出可用真实基线，随后才评价重建/修复的下游效果。SparseDrive准备中，尚无结果；scene-0002属于train，训练集重叠需保留。
- 证据：../../WORLDSIM_SIMULATION_FRESH_BASELINE.md、../../autoresearch/worldsim_simimpact/milestone9/evidence。
- 人工verdict：null。整体goal active。
''',encoding='utf-8')
status='''# 最新状态：第九里程碑

新来源真实输入基线与一次路线控制已完成：16次策略/PDM配对，仍未通过冻结准入，不启动scene-0002拟合。路线修正消除一次框交叠，但平均FDE仍3.73m，不能归因重建。

原生nuScenes SparseDrive-S stage2正在准备，尚未前向。下一步先评价真实六相机；新来源属于train，不是模型训练未见集。当前无新增严重几何因果坏例。

[报告](FRESH_BASELINE_CLOSEOUT.md) · [数值](FRESH_BASELINE_EVIDENCE.json) · [全起点图](F21_Fresh_Real_Baseline_Control.png) · [接口核查](PLANNER_INTERFACE_REVIEW.md)

累计818策略/PDM、222CenterPoint、72几何前向、18HUGSIM＋4SplatAD反馈闭环。整体goal active，人工verdict=null。
''';(O/'STATUS.md').write_text(status,encoding='utf-8')
page=(O/'index.html').read_text(encoding='utf-8');pos=page.index('<section>');page=page[:page.index('<main>')]+'<main><span class="status">新来源真实基线未通过 · 原生规划器准备中</span><h1>先验证真实驾驶基线，再归因重建危害</h1><p>scene-0002八起点的固定直行与一次路线提示对照已完成。平均ADE/FDE从1.67/4.64m降到1.42/3.73m，框交叠1/8降到0/8，仍未达预定准入；不启动该场景拟合。</p><p>全部原始RGB、八起点轨迹与失败条件保留。下一条是官方nuScenes SparseDrive-S基线，权重与环境准备中，尚无新模型结论。此前四模型六日志矩阵和局部资产证据见下方。</p><p><a href="FRESH_BASELINE_CLOSEOUT.md">最新报告</a> · <a href="FRESH_BASELINE_EVIDENCE.json">全部数值</a> · <a href="PLANNER_INTERFACE_REVIEW.md">规划接口核查</a></p><section><h2>真实输入也有明显误差：先完成基线控制</h2><a href="F21_Fresh_Real_Baseline_Control.png"><img src="F21_Fresh_Real_Baseline_Control.png" alt="原始RGB、组件图、八起点完整轨迹及固定路线控制"></a><p class="small"><a href="F21_Fresh_Real_Baseline_Control.pdf">PDF</a> · <a href="F21_Fresh_Real_Baseline_Control.svg">SVG</a></p></section>'+page[pos:];(O/'index.html').write_text(page,encoding='utf-8')
for path in S.rglob('*'):
    if path.suffix in ['.md','.json']:
        path.write_text(path.read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
print('MILESTONE9_PACKAGED')

# OmniDreams 24例：DriverQ辅助修订待审版

Task `WS-V75-OMNI-REVIEW-02`，run `20260923-proposal-r4-driverq-review`。本版只有原始 nuScenes / DriveStudio 10 Hz 视频和计划干预；所有 factual、counterfactual 与新评分栏位均待推理和人工 review。`model_calls: 0`，`failure_ledger_refs: []`，`failure_ledger_delta: none`。

```text
nuScenes原始数据 → DriveStudio 10Hz RGB/位姿/track
                           ↓
                 DriverQ SQLite 场景/运动/投影查询
                           ↓
                   24例10秒原片 + 指定反事实
                           ↓
                       用户人工确认
                           ↓
           OmniDreams factual / counterfactual（待运行）
```

DriverQ 官方源码固定在 `/root/autodl-tmp/external/DriverQ`，commit `acf14f8`，来源 <https://github.com/bluebarryz/DriverQ>。本地适配器 `scripts/build_driverq_processed_adapter.py` 把已有的3个V7.5场景和8个V5完整预处理场景写入其 SQLite schema：11场景、2,156自车位姿、48,183对象位姿、503轨迹、12,368个每0.5秒采样的几何投影行。DB为 `/root/autodl-tmp/data/worldsim_v75_downstream_bench/driverq/processed-11-scenes.db`；元数据和四张地图在 `map_metadata_driverq11/`。DriverQ自带的 `braking` 查询已能在此数据库上运行；其2 Hz阈值没有被当成本项目的评价结论。

这个适配器使用 DriverQ 的场景/运动/相机查询结构，**没有**运行其对原始 nuScenes 全量数据的官方 exporter，也没有部署 React 前端。释放包自带的 DB 只含 val 场景，而我们有完整10 Hz RGB的11个场景均为 train，因此没有下载该不匹配的数据库/大批相机包。`visibility_level` 在适配库中为空：有框投影不等于无遮挡；SQL筛选之后仍交给用户审原片。地图事件表（lane change、cut-in等）尚未填入，不借其空表声称检测了事件。

| 旧例 → 新版 | 原片/目标 | 计划反事实 | 修订原因 |
| --- | --- | --- | --- |
| SPEED-ACTOR-02 R4 → R5 | scene-0998，前视汽车 #11 | 0.5秒起沿原轨迹1.5×加速 | 替换与第3例共用巴士的窗口；DriverQ筛到20/20采样帧在画内 |
| SPEED-ACTOR-03 R3 → R4 | scene-0255，后视巴士 #3 | 0.5秒起沿原轨迹1.5×加速 | 用户认为原巴士已慢，反事实改为加速 |
| LATERAL-EGO-01 R3 → R4 | scene-0535，前视、自车行驶 | 0.5秒起1.8秒内向左3.5米 | 旧局部+Y实为前进轴；新源横移中心在可行驶区域 |
| LATERAL-EGO-02 R3 → R4 | scene-0471，前视、自车行驶 | 0.5秒起1.8秒内向右3.5米 | 完全替换静止自车原片 |
| LATERAL-EGO-03 R3 → R4 | scene-0255，前视、自车 | 0.5秒起1.8秒内向左3.5米 | 原片保留，修正自车横移轴 |
| LATERAL-ACTOR-01 R3 → R4 | scene-0535，后视汽车 #1 | 0.5秒起1.8秒内汽车自身向右3.5米 | 换掉原来刁钻且目标看不清的右前视 |
| LATERAL-ACTOR-02/03 R3 → R4 | 原片保留 | 分别向左/向右3.5米 | 明确有符号方向，不等同屏幕左右 |

全部横移的有符号值规定为 **+ 左、− 右**，以车辆行驶方向为参照。DriveStudio自车的雷达位姿是 +Y 前进、+X 向右，因此自车左移必须在该位姿下取 −X；对象位姿是 +X 前进、+Y 向左。新版 `coordinate_convention=vehicle_forward_left_v2` 仅用于新提案，历史推理输入契约不回写。1.8秒达到目标偏移后保持至10秒结束。

CPU门控：替换后的两条自车横移原片在10秒100帧内低于0.5米/秒的帧数均为0。新后视汽车在事实和计划分支的几何投影均覆盖100/100帧；新前视加速汽车事实100/100、计划89/100。三个自车横移与新后视汽车的**中心点**在nuScenes drivable area 的20个0.5秒采样均为20/20；这不是车身足迹、碰撞、车道方向或生成画面的合格结论。LATERAL-ACTOR-03是骑行者，drivable-area中心点0/20不能按机动车道路门控否决骑行环境，仍请用户看原片。其他旧候选没有凭这些新指标自动升级为合格。

本版8例修订，16例逐字段保持上版，旧R3/R4、原始视频和用户先前四例口述意见均保留。完整参数见 [r4候选清单](candidates-driverq-review.json)。全部 `approval.status=pending`、`inference_allowed=false`。本地HTML交付在 `outputs/cfbench-20260923/omnidreams-case-review/index.html`；远端完整版本在 `/root/autodl-tmp/runs/worldsim_v75/WS-V75-OMNI-REVIEW-02/20260923-proposal-r4-driverq-review/index.html`。实际推理须使用用户最终批准的case、相机、目标和变量；ReSim继续停止。

# 代码核查更新：新来源基线未通过，准备原生nuScenes规划器（2026-09-15）

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

# 感知到规划接口复核（2026-09-15，尚未运行新规划器）

当前 CenterPoint 的定位偏差不能直接解释先前的 TransFuser 轨迹：后者接收 RGB＋LiDAR，不消费这些检测框。要接通另一条实际链，必须将检测结果送入真正读取它的规划模块，并先检查真实输入基线；不能只用框计算 TTC 就宣称实际制动。

```mermaid
flowchart LR
    A[重建网格] --> B[模拟 LiDAR]
    B --> C[冻结 CenterPoint]
    C --> D[检测框与速度]
    D -. 待接通 .-> E[消费检测的规划器]
    E --> F[轨迹与执行]
```

- **WorldEngine / PDM-Closed**：已安装源码的 `pdm_policy.py` 读取 agent input 的 DetectionsTracks，因此在模块上可以接收感知结果；但同时调用 `get_maps_api(..., nuplan-maps-v1.0, map_location)`、道路 block 路线和导航车道。当前 nuScenes 数据不能直接作为原生 nuPlan 场景运行。自写简化地图/GT路线时必须标成适配实验，不能报告为完整官方 native benchmark。
- **BridgeSim**：作者官方仓库说明支持 nuScenes 场景转换、PDM-Closed privileged agent 及闭环 runtime LiDAR。值得作为后续接口候选，但本轮未安装或运行；README 的功能声明不等于已验证其 PDM 检测替换接口或传感器真实性。[官方仓库](https://github.com/VAIL-UCLA/BridgeSim)
- **PKL**：作者实现可直接比较不同检测输入下预训练规划分布，API 为 `calculate_pkl(gt_boxes, pred_boxes, ...)`，适合作为成熟的检测到规划诊断参照；是 CVPR 2020 方法，不是 2026 SOTA，也不是车辆执行闭环。较大 KL 仍只代表分布差异，须结合真实未来质量评价才可称退化。本轮只完成接口核查，未跑 PKL。[作者代码与说明](https://github.com/nv-tlabs/planning-centric-metrics)

研究决策：先完成当前六日志冻结感知矩阵并审查新候选。只有出现能在真实输入基线上可靠评价的目标，才投入下一条规划器适配；不将几何误差、框匹配阈值或额外规划指标本身改写成严重仿真危害。

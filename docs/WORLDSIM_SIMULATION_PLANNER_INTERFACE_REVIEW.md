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

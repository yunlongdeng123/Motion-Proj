# WorldSim V6.5 Failure Analysis

## V65-F01 — static-label target semantics mismatch

P1 T0 的 trajectory condition 对 unit 内 shuffle 有可测响应，却无法改善冻结 q0，说明问题不是条件通路完全失效，
而是监督目标要求 task query 去解释 task-agnostic static hidden-FREE。

迁移依据：WoTE（ICCV 2025）用候选轨迹预测未来结果并评价轨迹；UniAD（CVPR 2023）与 VAD（ICCV 2023）
都把 planning query 与未来 occupancy / actor / map 表示交互。项目内对应迁移是保留 q0 的物理语义，另建 task
risk，不把两者混成一个需要同时提升 global voxel AUROC 的分数。

- WoTE：https://openaccess.thecvf.com/content/ICCV2025/papers/Li_End-to-End_Driving_with_Online_Trajectory_Evaluation_via_BEV_World_Model_ICCV_2025_paper.pdf
- UniAD：https://github.com/OpenDriveLab/UniAD
- VAD：https://github.com/hustvl/VAD

若 P1R 仍无 fixed-opportunity 增量，则关闭纯 trajectory task-risk family，转向 Actor×time/query outcome，而不再
修改同一 residual。

## V65-F02 — frozen temporal-info capability mismatch before quality read

首版 fresh cohort 依据 scene description、既有 processed availability 与旧 config exposure 冻结，但冻结 IR-WM
worker 会直接索引 `nuscenes_temporal_infos_train.pkl["infos"][scene]`。该 pickle 只有 700 个 scene keys；
`0520/0781/0800/0106` 缺失，因此失败发生于官方模型输入构造前，而非模型表现阶段。

官方 BEVFormer 的标准方案是执行 nuScenes `tools/create_data.py` 生成 temporal train/val infos：
https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md 。本项目当前采用更小的迁移边界：
不重建 infos，不改变 CAN bus/schema/checkpoint contract，只把 cohort 换成冻结 pickle 已支持且旧 configs 未使用的
`0996/0443/0002/0043/0023/0072`。该替换只读 capability metadata，发生在任何 P2 quality read 前；因此不形成
selection bias，也不改变 P2 gate。后续 cohort freeze 必须在 metadata selection 时同时审计 backend key availability。

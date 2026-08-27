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

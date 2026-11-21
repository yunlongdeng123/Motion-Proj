# 表面高斯 student：实际 v1 与后续 v2 入口

当前整车候选为 `diagnostic_rejected_geometry`，不得 promotion。历史 v1 已完成预览、9 次真实 smoke 更新、固定 200+200 次拟合和独立进程重载；新 v2 入口的 CPU 灰模硬门拒绝当前候选，不追加 GPU。

```mermaid
flowchart LR
    A[可信V几何与RGB锁定] --> B[仅U补面]
    B --> G[灰模四项证据硬门]
    G --> M[材质与表面高斯]
    M --> R[真实GPU预览与短smoke]
    R --> T[获准的固定预算拟合]
```

`check_grey_asset_gate.py` 只用标准库，核对四项灰模证据、V 几何/RGB 保护、生成仅 U、文件哈希及任务特定阈值。所有项必须明确 passed；缺失或 pending 拒绝。`run_surface_student.py` 非 `cpu-import` 在导入 legacy/torch 前调用它。当前 checker 和 student 入口均以 2 退出，torch 与 CUDA 均未初始化。合成 schema 只测试验证逻辑，禁止真实入口使用。

实际 v1 合同：几何、法向、尺度、轨迹与高阶 SH 固定；vehicle 阶段只更新真实可见支持内 DC 和有界 opacity，joint 阶段沿原方法冻结 opacity，更新 DC 与许可 1191 道路 RGB。可见支持来自共同 renderer 对真实 86/110 核心的 feature VJP；生成 alpha 不参与。

车辆真实 loss 中 road color detach；道路 teacher loss 使用统一 DELETE 渲染，完整 vehicle packet 全 tensor detach。逐 loss autograd 证明跨分支为 0，各自活跃分支非零。固定 75 行 a9 sidecar 施加于原生和表面 packet，不重新筛选。U 不作为负监督，134/179/198 不拟合。

GPU smoke：完整 Adam/RNG 状态保存恢复 bit-exact，下一步参数最大差 1.19×10⁻⁷、loss 差 0；compact 资产重载渲染差 0。正式 400 次更新耗时 110.64 秒；五帧 8 个输出在独立进程重载中全部最大差 0。torch 峰值 allocated 5.13 GiB、reserved 7.55 GiB。

物理 optimizer 更新总数为 **409 = 400 正式 + 9 smoke**；9 次已包含 2 次 replay，smoke 之外额外更新为 0。继承 ledger 的 `extra_replay_updates=2` 原字段保留，只通过 `resource_ledger_corrected.json` 解释为 smoke 内部重放，不加成 411。formal 的 647 raster / 600 branch backward 只属于该进程，不冒充 whole-task 调用总数。

资产保留所有 surface metadata、完整初始 NPZ 与最终状态；道路只保存 1191 行 RGB patch 与 R2 base binding，不复制完整道路。实际执行代码已冻结 `student/source/run_surface_student_actual_v1.py`；当前 `run_surface_student.py` 是未来 v2 gate 入口。独立 float 渲染保留远端。

上述接线通过属于工程证据；当前整车生成替换没有保护可信可见几何，不满足新的灰模前置协议。旧 `shape_gate_passed/material_gate_passed` 布尔值只保留为历史诊断入口许可。

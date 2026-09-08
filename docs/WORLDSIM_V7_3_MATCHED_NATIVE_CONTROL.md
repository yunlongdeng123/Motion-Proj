# V7.3：同全轨迹监督的原生几何控制

截至2026-09-08 07:48 UTC，R10联合模型和R11原生强控制都在实际训练中，分别到第10/24轮，尚无最终结果。R5恢复训练已完成30轮并退出。R12有限宽束free仅登记、未启动。本文描述当前对照，不把早期“未启动”记录当作现状；详细历史保留在三本研究台账及git历史。

## 当前两项训练

共同数据是完整489 Actor的population，旧输入路径中371 FIT参与训练、67 development可预测，51零build LiDAR对象保留缺失。两项都从M1r3初始化，读取全24视图/744份冻结前缀；仅FIT使用既定full_track几何标签，模型输入仍只有build图像/LiDAR/标定/已知轨迹，development没有梯度。

共同目标为target→surface覆盖＋0.5 hard range free＋0.05 box envelope＋1.0 build native像素深度，event关闭；seed7304、AdamW lr1e-5、目标30轮。native depth使用当前相机像素的真实build LiDAR，独立于预测框内候选，保留F06修复。相机、轨迹、build尺度只读。

| 方法 | 实际run与进程 | 可训练通路 | 物理表面 |
|---|---|---|---|
| R10 joint | `20260907T233000Z__population-joint-full-track-s7304-r10`；code26a7e509，PID53472 | DPT32654562＋query1670517参数 | 局部空间查询生成的位置/法向/片形状 |
| R11 native_only | `20260907T233000Z__population-native-only-full-track-s7304-r11`；code9541ac7d，PID38460 | DPT32654562参数，query冻结 | native＋LiDAR中心的PCA曲面片 |

以上均位于`WS-V73-M2-GLOBAL-ACTOR-01`。日志分别为`/root/autodl-tmp/controller_logs/v73_population_joint_r10.log`、`v73_population_native_r11.log`。当前observed allocated峰值分别10.213784/2.346402GiB；两者运行不同阶段且并行，不能直接用峰值相加代替总显存测量。

R11原生深度在正确相机曝光和Actor轨迹下回投规范坐标，融合原build LiDAR和512个native FPS支持，使用min(build,1024)+512中心、0.06m PCA片。PCA邻域/方向每步重算但本步停止梯度；所选native中心的真实位置梯度回传DPT。无相机或无有效几何/native数据梯度时，保留相应LiDAR读出，呈现不虚报为optimizer更新。30轮计划11130个Actor呈现，最终真实更新数由日志统计。

query模块在R11仅提供固定grid/数量定义，无逐点MLP或局部交互；DPT内部多层多尺度解码仍完整。训练每步重新解码，绝不跨优化步缓存DPT输出。只有一次固定权重的初始化/最终no-grad评价可按scene/view共享深度；冻结聚合前缀继续共享。不减少视图、分辨率或原始观测。

## 比较与尚未完成的结论

R10−R5主要回答FIT标签由短窗扩大为full_track的问题；R10−R7是同full_track下视觉联合路径与LiDAR路径比较；R10−R11比较query生成通路与认真训练的原生头/融合控制。最后一对同时改变位置生成及法向/片形状参数化，不能把差异自动归因为空间attention；完整population同容量pointwise控制仍未运行。

R5本身在覆盖改善时出现early/free退化，已在`WORLDSIM_V7_3_JOINT_R5_RESULTS.md`记录；这既不能替代当前R10/R11比较，也不能推导所有视觉几何适配失败。R5原第22轮意外退出只保留第21轮状态，107个未保存更新留原日志；恢复完成有效11130更新，旧checkpoint缺完整RNG的执行差异已披露，不声称逐比特连续。

R12登记`20260908T050000Z__population-joint-full-track-beam-range-s7304-r12`，从与R10相同的M1r3/seed初始化，仅改变free为`beam_tube_range`、width.03m/resolution32；full_track/native1/free.5/event0/30轮不变。它尚未启动，无等待启动队列。如最终采用该目标，也需同目标的原生强控制，不能拿hard range的R11冒充完全匹配。

visual-only训练入口虽已实现，但尚无真实新反向结果；R10/R11/R12继续原cohort。输入条件扩大、物理目标变化与空间机制要分别解释，见`WORLDSIM_V7_3_COMPARISON_PROTOCOLS.md`。新20日志质量确认仍未读取，整个V7.3未完成。

原生解码路径依据：[VGGT官方训练说明](https://github.com/facebookresearch/vggt/blob/main/training/README.md)和[DPT实现](https://github.com/facebookresearch/vggt/blob/main/vggt/heads/dpt_head.py)。微调DPT是强控制的既定工作，本身不作为新贡献。

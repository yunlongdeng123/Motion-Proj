# V4.1：最终质量约束下的固定几何道路补层

task=`WS-V82-R15-GN11-V41-PRE-GPU-20261023`。当前只准备固定几何 **V4.1-R**，已停在启动 GPU 前；新真实场景训练更新为 0。旧 V4-R 30K 完整保留，旧 RP 及其心跳继续暂停。RP 先验和 a6 几何补全均不在本轮执行入口。

![输入、模型、可导合成、质量监督与输出](architecture.svg)

## 问题与改动

V4 独立道路只监督每帧约 448 个可靠中心，无法直接检查约 3.1 万受影响的车身外像素；道路增大不透明度后还会改变原 V1 的透射与颜色累积。冻结 V1 的参数本身不能保证最终图像或车身不退化。

V4.1 仍先完成原 V1 一次更新，然后以当前 V1 为常量，在道路更新中做同次真实深度光栅化。道路只学习 DC 和 opacity，原 14,034 个高斯的位置、尺度、朝向及数量固定；R 可学习/输出范围仍为 1,919 个已有可靠观测单元。V1 损失、SGCM/NPS、随机采样、学习率、增密和裁剪不改；151 个原生文件不变，train.py 删除两块道路 hook 后逐字回到原 V1。未引入 V3 动静态分层改动。

**M：可靠可见路面。** 使用固定受支撑参考面、同帧最近 LiDAR 地面回波的小三角形、至少第二帧几何确认及车辆/三维框遮挡排除。三角形最大世界边 1.6m、图像边 24px、深度容差 0.3m、法线余弦至少 0.9；四邻格缺支撑不外插。插值深度是派生证据，不是实测密集深度或语义道路 GT。五身份、全车辆与原生 proposal 掩码共同排除遮挡。最终 M 为 645,888 像素，平均 **13,181.39/帧**；243,177 个有效局部 SSIM 窗口，平均 **4,962.80/帧**。车身/边缘与 M 的交集为 0，未知/先验单元获得真实 RGB 标签数量为 0。

**P：M 以外的最终合成保护。** 所有其余像素以当前冻结的 V1 完整重建为目标，而不是以原图充当独立道路 GT。车辆内部和边缘分别按身份求均值，再在非空身份间求均值；其余保护区独立求均值。五身份标签绑定 `[5,26,18,6,37]`，另单列其余车辆联合域。边缘使用 7×7 方形膨胀减侵蚀内部。不会把大面积背景或大车的均值当作小车质量保证。

这项完整保护是检查后的必要补齐：仅用初始可靠路面/已知非道路保护域时，旧 R 受影响像素有 42.37% 进入 M、25.71% 进入已知保护、31.91% 仍未知。最后一类现在也进入冻结 V1 的**合成参考约束**；它们仍无道路真实颜色标签。该约束可能限制补景能力，需通过删车有效覆盖检查判断，不预先宣称成功。

## 明确的损失与梯度

每个道路采样帧使用：

```
L = 0.8 × mean_M |C_joint - RGB|
  + 0.2 × mean_valid_windows (1 - masked_SSIM(C_joint, RGB; M))
  + mean_M (sqrt((C_road/alpha_road - RGB)^2 + 0.05^2) - 0.05)
  + mean_nonvehicle_P |C_joint - stopgrad(C_V1)|
  + mean_nonempty_actor(mean_body |C_joint - stopgrad(C_V1)|)
  + mean_nonempty_actor(mean_edge |C_joint - stopgrad(C_V1)|)
  + 原 V4 中心覆盖约束（权重 0.1，目标 alpha 0.95）
```

SSIM 使用 11×11 Gaussian 窗、sigma=1.5，仅取 M 内有效像素的归一化局部矩，且有效像素占窗口至少 80%，窗口中心也须属于 M。无效/车辆像素不参与均值与方差；不是把图像乘 mask 再算 SSIM。训练浮点不截断。原 V4 可靠来源中心覆盖、私有采样顺序及 DC/opacity 学习率保留；新增稠密道路颜色监督使用同帧原生 RGB，不融合颜色、不把初始化灰色标成真值。

每步道路仍采两帧，逐帧 backward 后只做一次 Adam 更新。Road-only / native-reference / joint 前向增加计算，不能称为等计算量。道路自己的 RNG 隔离；静态、动态、环境颜色及其输入全部 detach，环境色后的 `(1-alpha)` 对道路 opacity 保留梯度。道路使用独立屏幕缓冲，不融合原分支增密梯度。

## 车身与道路必须同时验收

`evaluate_v41.py` 准备了同 checkpoint、同 49 帧原 V1 / 联合 / 道路独立 / 删车的浮点保存与评价。正式头部口径为原 V1 `[0,1]` 限幅，PSNR/SSIM/LPIPS 使用原实现和参数；原始不截断诊断另存。完整图、可靠路面、保护域、正常背景、联合车身与边缘、**五辆车各自的内部和边缘**都保留逐帧误差、均值、最差帧和退化帧数。局部 SSIM 只在有有效窗口时计算，空域记缺失，不冒充通过；LPIPS 为完整图原口径。

删车隐藏道路没有原图 GT。a37/a6 及其他身份的固定 query 分母分别报告几何支撑、R 可学习单元、未知、独立 alpha、完整/删车中的真实道路贡献。阈值 0.05/0.25/0.5 下记录覆盖率、8 连通分量和最大连续分量占比；再与同世界单元在可靠可见路面的颜色误差（MAE≤0.05 的诊断阈值）结合。该颜色迁移检查不是隐藏纹理真值。不能因为 opacity 高、PSNR 恢复或道路变透明就判完成。

独立 CPU 工具 `compare_v41_states.py --native-only` 在正式 30K 后与既有 V4-R 30K 比较原 V1 参数、Adam、RNG 与采样状态；仅排除执行路径与道路元配置。真实 CUDA 的精确一致性目前未验证。检测到不一致须如实报告并定位，不能自动加 B。

## 启动前检查与未验证项

20 项 V4.1 CPU 机制检查、28 项继承回归通过；真实 49 帧相机/稠密域/身份和原生配置绑定通过。涵盖稀疏中心之外的误差、遮挡像素不进入 SSIM、车身/边缘独立梯度、小车不被大背景稀释、透明道路不能通过有效覆盖、深度交错变暗的道路梯度、环境透射梯度、V1 参数/Adam/RNG 隔离、道路精确恢复和固定几何。

这些 CPU 渲染测试使用解析替身，**不等于原生 CUDA 光栅反传、显存或恢复通过，也不证明实际质量改善**。a37 的连续有效道路覆盖仍待实验；a6 不存在的几何单元没有被补出。GPU smoke、GPU 评价与正式训练本轮均未执行。

真实生命周期 smoke 已准备独立目录：R 连续 1000 步，以及 R 500 步保存退出→精确恢复到 1000 步；200/500/1000 都触发原生评价/保存，比较完整原 V1 与道路状态后执行新 49 帧评价。工程开销另计 V1/道路各 2000 更新，不属于正式 30K；实际工作累计至少 600 秒，不以 sleep 凑数。任一阶段失败、恢复不一致或不足时长都不准入正式长跑，保留现场。完成后只产生凭据，不自动启动正式训练。

若 CUDA 本身非确定导致连续/恢复不同，记录差异并定位，不能把 CPU 通过或放宽阈值当作已通过。正式入口仍为同共同起点 fresh 30K，V1/道路各更新 30K，10K/20K/30K 各保存静态/动态/环境/完整训练状态/道路五件与 manifest。固定几何 R 未通过质量和覆盖之前，不启动 RP 或几何扩展。

## 路径与复现入口

主机 `wm-vgpu-1011`；Python `/root/autodl-tmp/envs/drivingeditor/bin/python`。

```
campaign: /root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023
dense:    /root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023_dense_r4/manifest.json
V1:       /root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137
road:     /root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json
```

本轮仅执行 dry-run：`tools/run_v41_smoke.py --campaign <campaign>` 和 `tools/run_v41_arm.py --campaign <campaign> --arm R`。没有 `--execute` 不启动 GPU。未来实际 smoke 需显式 `--execute`；正式入口额外要求方法/配置一致的真实 smoke 凭据。调用工具不会开机。已检查旧心跳为 PAUSED。

从原输入复现：`prepare_v41_dense.py` 生成 R2 证据；`prepare_v41_vehicle.py` 生成 R3 车辆独立域；`prepare_v41_protection.py` 生成 R4 全图合成参考保护。R1/R2/R3 开发证据保留；R4 是唯一活动输入。`build_v41_campaign.py` 只创建新目录并从原 V1 安装；不覆盖已有 campaign。数据/模型/日志不进 Git。

failure_ledger_refs=`[V82-F01,V82-F06]`，failure_ledger_delta=`none`；quality_accepted=`false`，human_verdict=`null`。旧 R 额外失败/重放账沿用既有记录，本轮真实场景优化更新为 0。

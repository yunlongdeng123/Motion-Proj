> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# WorldSim V7.2 E1：双基座、RGB 与原生 beam 合同

日期：2026-09-07；task=`WS-V72-E1-VGGT-EVIDENCE-IO-01`；状态=`running`。本报告记录 E1 已成立的接口与一次瓶颈诊断，不把训练窗口观测误差写成方法收益。

## 已完成

- 官方有效基座固定为 VGGT-1B 与 Pi3X。VGGT repo/checkpoint=`a288dd0f` / `f164acf6...0467e`；Pi3X repo/checkpoint=`9fa3ddb3` / `69972d6e...1669a`。VGGT checkpoint 仅多出主动关闭的 394 个 `track_head.*` tensors、无 missing key；Pi3X 为严格 `0/0`。两者均输出点图、相机位姿、内参、置信度和非空 `[3,27,48,2048]` patch features。
- build-only 多相机合同记录 RGB SHA-256、原始内参、OpenCV 相机位姿和 original-to-model pixel transform；target 不进入 backbone cache。
- 原生 beam 合同把查询与监督分开。`valid_emission_mask=false` 对应 unknown/invalid，目标 `return_count=-1`；有效 firing 上无回波、单回波、双回波分别为 `0/1/2`。Waymo 解码沿用官方 `range>0` hit 规则，但 firing validity 必须由独立来源提供。
- Waymo context 在任何 payload/quality 下载前按固定 SHA-256 排序冻结：train/development/route-select=`600/99/99`，官方 validation 的 `202` 个 context 整体作为 source-test。原始 payload 因 Waymo 账号与 gcloud 授权尚未物化。
- 有序表面回波测度的最小实现已加入：场景级深度排序、blocking/detection 分离、无回波概率、proper NLL 与 primitive split 质量守恒。它是 E2 的机制地基，不是实际精度结论。

## 正式双基座诊断

canonical=`run://worldsim_v72/WS-V72-E1-VGGT-EVIDENCE-IO-01/20260907T144500Z__e1-vggt-pi3x-train-observation-s7201-r4`；code=`4c86e621`；dataset role=`train`；target/source/external test read=`false/false/false`。

同一 nuScenes `scene-0015` / log `d31dc715...26ed` / sample `234b0f37...5e08` 使用 `CAM_FRONT_LEFT/FRONT/FRONT_RIGHT`，统一输入为 `3×3×378×672` RGB、同一已知标定和 build LiDAR。窗口 fingerprint=`7066771c...dabe9`。选择只依据 metadata 顺序与 payload 完整性，不看质量。

| 层次指标 | VGGT-1B | Pi3X |
|---|---:|---:|
| native valid fraction | 1.000 | 1.000 |
| native pairwise camera stress | 0.0955 m | 0.0448 m |
| Sim(3) aligned camera-center RMSE | 0.0667 m | 0.0414 m |
| alignment scale | 169.022 | 2.234 |
| retained surface fraction | 1.000 | 0.500 |
| LiDAR correspondence count | 5,823 | 4,167 |
| median surface residual | 74.668 m | 3.412 m |
| median absolute depth error | 52.041 m | 3.104 m |
| early / hit / late at 0.2 m | 47.62 / 0.31 / 52.07% | 3.74 / 0.67 / 95.58% |
| inference / peak GPU | 20.63 s / 7.61 GiB | 21.86 s / 6.20 GiB |

两者都通过真实输入能力检查，Pi3X 在这一次 metric driving observation 上明显更接近 LiDAR；VGGT 的任意尺度和车载 rig 外推需要额外适配。两者的 0.2 m hit 都很低，说明冻结视觉点图不能直接替代回波/占据模型，也给 B1–B7 留出真实解释空间。当前数字包含单帧跨传感器时差、动态物体与公共 Sim(3) 的共同影响，统计单位只有一个已暴露 train window，不能作为跨场景排名或论文表。

## 失败与修复

r1 在 VGGT BF16 位姿上调用 `torch.linalg.inv` 失败；按 VGGT 官方实现换为 SE(3) 闭式逆。r2 成功但暴露 `[B,S,N,C]` token 轴切错，VGGT feature channel 为零；r3 修复后仍引用旧 Git SHA。提交修复后生成 r4，manifest 与代码一致。统一工程记录为 `V71-F67`，没有 target quality、source-test 或 external-test 暴露。

## E1 剩余边界

E1 保持 `running`。还需在授权取得后物化 Waymo development payload，实测原始负 range、固定 grid validity、逐 pixel pose、双回波与相机同步；随后冻结 scene-level 主指标数值容限和预算。当前可并行推进 E2 的 synthetic properness、同信息 B0–B4 接口和 CAPA 环境，不把数据授权当成算法停止条件。

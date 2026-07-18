# Route Pivot R1：真实时间采样与 SVD fps 审计

> **任务 ID**：`RP-R1-02`
> **状态**：`done`；机器决策已冻结，32 个盲审 pair 为补充诊断，不阻塞 A0
> **正式 run**：`route-pivot-r1-temporal-s20260718-v1`
> **代码提交**：`f4b4cd5872d732dc2694d6fb9bae53fcf6dd7304`
> **配置指纹**：`a1ed2f2527e9f9ea6a07ed6830b0c4ee6522c38b2fcae5dd1d6db239e86e10ed`
> **结果指纹**：`01cfc57ed98bb1f94f301f2a7f102ec02a33c0681967402fced354cade19c7e4`
> **最后更新**：2026-07-18

## 1. 冻结结论

nuScenes `CAM_FRONT` 的 32 个 scene-distinct 训练 clips 显示，中位相邻帧间隔为 `0.5000 s`，
中位有效采样率为 `2.0000 Hz`，8 帧中位覆盖 `3.5000 s`。因此真实训练视频时间尺度与当前
SVD `fps=7` micro-conditioning 在数值上确有 mismatch。

但是，直接把 SVD fps 输入降到 2 或 4 不是安全校准：两者相对 `fps=7` 都显著增加 Base motion，
却没有同时通过首帧、画质、track survival 与 acceleration safeguard。正式决策为：

```text
generation.fps = 7
decision = keep_reference
reason = no_lower_fps_met_significance_and_safeguards
```

该选择只冻结 V5 后续 A0/A1/B0 的生成协议；不改写旧 run，也不声称 `fps=7` 等于真实世界 7 Hz。

## 2. 预注册协议

- 真实审计：官方 train split 抽取 32 个不同场景的 8-frame clips，使用逐帧微秒时间戳、ego pose
  与真实 3D annotations；真实 future 只用于数据时间/运动统计。
- 生成审计：官方 val split 固定 8 个不同场景的首帧条件；`fps ∈ {2,4,7}`，每条件 2 个 seeds，
  共 48 个视频。
- 配对控制：同一 condition/seed 的三档 fps 共用相同 condition noise 与 initial video latents；
  25 denoising steps、8 frames、motion bucket、guidance、noise augmentation 与权重均相同。
- 评估隔离：生成与 generated-video evaluator 不读取 future GT；CoTracker3 offline 仅从生成 RGB
  计算 camera-compensated image-plane 统计。
- 决策：至少 14 个配对组；motion 相对变化中位数至少 10% 且 paired bootstrap 95% CI 不跨 0；
  同时要求首帧、锐度、闪烁、track survival、acceleration 与 low-motion floor 全部通过。

正式 run 在干净提交上完成，48/48 生成和 48/48 打分有效，16/16 配对组有效；
`same_initial_noise_verified=true`，`uses_future_gt_for_generation=false`。

## 3. 真实视频时间尺度

| 统计量 | 样本数 | 中位数 | p05–p95 | 解释边界 |
|---|---:|---:|---:|---|
| timestamp delta | 224 | `0.5000 s` | `0.4000–0.5500 s` | nuScenes keyframe 间隔并非逐项严格 0.5 s |
| effective fps | 224 | `2.0000 Hz` | `1.8182–2.5000 Hz` | 后续真实速度/加速度必须使用实际 delta-t |
| clip duration | 32 | `3.5000 s` | `3.3775–3.6225 s` | 8 帧不是 1 秒级短片 |
| ego translation / frame | 224 | `2.9895 m` | `0.0012–5.4740 m` | 混合停车与高速自车运动场景 |
| ego translation / second | 224 | `6.0078 m/s` | `0.0024–10.9592 m/s` | 使用真实时间戳归一化 |
| actor track length | 671 | `8 frames` | `2–8 frames` | 共 639 条至少两帧的有效 actor tracks |
| actor global center speed | 3660 | `0.7522 m/s` | `0–6.4524 m/s` | 仅真实标注统计，不进入生成评估 |

## 4. Base fps 对照

下表均为相对冻结参考 `fps=7` 的 16 个 condition-seed 配对中位数；正的 motion change 表示运动量增加。

| 候选 fps | dynamic change（95% CI） | image velocity change（95% CI） | 首帧 PSNR delta | 锐度比 | 闪烁比 | survival 比 | acceleration p95 比 | 决策 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | `+24.74%` (`+13.46%`, `+34.51%`) | `+77.97%` (`+32.16%`, `+317.11%`) | `-1.899 dB` | `0.820` | `1.275` | `0.761` | `2.116` | reject |
| 4 | `+10.05%` (`+5.44%`, `+18.54%`) | `+110.71%` (`+13.40%`, `+168.92%`) | `-0.228 dB` | `0.951` | `1.182` | `0.859` | `1.950` | reject |

`fps=2` 的首帧、锐度、闪烁、survival、acceleration 五项均失败；`fps=4` 保住首帧、锐度和闪烁，
但 survival 与 acceleration 失败。两者都通过 anti-low-motion floor，因此本结果不是“候选少动”，
而是“更多图像运动伴随更差的可跟踪性和高阶不稳定”。

## 5. Reviewer 预判与主张边界

### 5.1 “既然真实数据是 2 Hz，为什么不用 fps=2？”

SVD fps 是 learned micro-conditioning，不是经过物理标定的播放速率。真实 timestamp 一致只提供选择先验，
不能覆盖生成质量门禁。当前 paired evidence 表明直接数值匹配会显著改变 sampler 分布并损坏安全端点。

### 5.2 “加速度恶化只是 CoTracker nuisance 吗？”

这里不把 image-plane acceleration 称为真实世界加速度，也不单独据此决策。结论依赖同 condition/noise 的
配对比值，并同时观察 first-frame、锐度、闪烁、track survival、dynamic degree 与 velocity。32 个盲审
pair 用于检查 evaluator nuisance，但不会被事后用来改阈值或选择 fps。

### 5.3 “保留 fps=7 是否掩盖时间错配？”

没有。时间错配被明确记录为表示与监督风险；A0/A1 的真实 target 一律使用真实 delta-t，且模型内部表示
是否能分离 ego-induced 与 actor-residual motion 要通过 scene-disjoint probe 单独证明。R1 只否定
“直接改一个 fps 标量即可修复”的快捷方案。

### 5.4 “人审未完成为何可以进入 A0？”

R1 的冻结决策由预注册机器 safeguard 给出，且无低 fps 候选合格。人审材料是 evaluator/主观运动可信度
补充诊断，与 A0 的真实几何 target legality 独立。若未来人审强烈反驳机器排序，只能以新任务重开
calibration，不得改写本次 run。

## 6. 人工复核材料

正式 run 已生成 32 个盲审 cases，每个 condition/seed 含 `2 vs 7` 与 `4 vs 7` 两组；A/B 侧随机化，
所有视频统一以 7 fps 编码，模板 verdict 保持 `null`。

```text
/root/autodl-tmp/runs/route-pivot-r1-temporal-s20260718-v1/review/REVIEW_PROMPT.md
/root/autodl-tmp/runs/route-pivot-r1-temporal-s20260718-v1/review/cases.jsonl
/root/autodl-tmp/runs/route-pivot-r1-temporal-s20260718-v1/review/reviews.template.jsonl
/root/autodl-tmp/runs/route-pivot-r1-temporal-s20260718-v1/review/videos/
```

`review_key.jsonl` 只用于完成标注后的聚合解盲，评审时不得打开。当前状态为 `awaiting_reviews`，
但 `RP-R1-02` 机器 gate 已完成并解锁 `RP-A0-03`。

## 7. 可复现证据

```text
/root/autodl-tmp/runs/route-pivot-r1-temporal-s20260718-v1/
  manifest.json
  resolved.yaml
  real_temporal_clips.jsonl
  real_temporal_summary.json
  generation_cases.jsonl
  scored_cases.jsonl
  metrics.jsonl
  result.json
  summary.json
  review/
  COMPLETE
```

基础模型 `model_index.json` SHA256 为
`9119b8837600736ae38009c5dc80c76112307cb2d229a2cfb477d54c329ff53d`；CoTracker 仓库提交为
`82e02e8029753ad4ef13cf06be7f4fc5facdda4d`，checkpoint SHA256 为
`2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`。

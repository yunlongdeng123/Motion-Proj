# Route Pivot A0：真实 Ego–Actor Motion Target 合法性审计

> **任务 ID**：`RP-A0-03`
> **状态**：`machine_pass / awaiting_reviews`；只解锁 A1 machine probe
> **正式 run**：`route-pivot-a0-real-motion-s20260718-v3`
> **代码提交**：`45cb279`
> **配置指纹**：`758a43dcd594ebc44a2bdd0d76709c9e2a013169e48974c09090592b2b62d81a`
> **selection 指纹**：`ff879944bacd2e0804b842ddd569ab7c7fc9cc93cc23e991ab7552d7958e3926`
> **结果指纹**：`b5115548bcc0f0c15b3944ab70bb7668d16a3b5228573e4b766e318b7db1560e`
> **最后更新**：2026-07-18

## 1. 冻结结论

在 16 个不同 nuScenes train scenes、每个 8 帧的真实视频上，基于真实微秒时间戳、相机标定、ego pose、
3D instance annotation 与稀疏 LiDAR 的 target 通过全部 A0 machine legality gates：

- actor residual 在 camera/ego compensation 后仍能区分 moving 与 stationary attributes；
- target 具有足量共同图内支持，可用于局部 feature query，而不是依赖出画位置或极端投影；
- sparse ego-induced background flow 与真实 RGB 的 RAFT 方向一致；
- residual 与 ego speed 的相关性低于冻结上限，没有被自车运动幅度单独决定；
- schema、visibility、timestamp、calibration、finite 与 LiDAR coverage 均通过。

因此 A0 只证明“真实 target 在机器审计上存在、合法且可局部化”，解锁 `RP-A1-SCAN-04A`。它不证明
冻结 SVD feature 已编码这些信号，不解锁 A2、训练、生成改进或论文方法名。12 个 panel 的人工 target
复核仍为 `awaiting_reviews`，最终 Route A promotion 必须等待该 gate。

## 2. Target 与使用边界

### 2.1 Actor residual

对相邻帧同一 `instance_token`，实际中心投影为 $p_{t+1}$；把时刻 $t$ 的 3D center 视为世界静止，
通过两帧 `cam2ego/ego2global` 变换得到预测投影 \(\tilde p_{t+1}\)。监督量为：

\[
r_t=p_{t+1}-\tilde p_{t+1},\qquad v_t=r_t/\Delta t.
\]

所有速度使用真实 camera timestamp 的 \(\Delta t\)。该量是 camera-compensated image-plane residual，
不是世界坐标物理加速度。

### 2.2 Background ego target

primary truth 只使用当前帧投影 LiDAR z-depth、逐帧内外参和 ego pose，把世界静止点投影到下一帧；
不使用 Depth-Anything 作为主 target。RAFT 只在 GT box 外、FB confidence ≥0.5、两侧 flow magnitude
≥0.25 px 的稀疏点上做方向 sanity，不是训练 target。

### 2.3 严格范围

```text
target_scope = real_training_representation_only
uses_future_gt_for_generated_evaluation = false
uses_future_gt_as_inference_condition = false
```

真实 future annotation/LiDAR/ego pose 只能进入真实训练视频的 representation probe 或 auxiliary loss；
不得进入自由生成 rollout 的条件、candidate scorer 或正式 evaluator。

## 3. Schema 修复

原数据代码读取了 `min_box_visibility`，但没有实际过滤。A0 已在任何投影和轨迹构造前应用该门槛，
并保留旧字段，additive 增加：

```text
annotation_token, attributes, center_cam, corners_cam, velocity_global
intrinsics_frames, cam2ego_frames
```

正式 16 clips 中 `missing_schema_count=0`、`visibility_violation_count=0`，逐帧 calibration 与 intrinsics
最大漂移均为 0；全仓 synthetic tests 覆盖 visibility 断裂、instance continuity、微秒单位、behind-camera、
纯 ego translation/yaw、横/纵 actor motion 与可变 delta-t。

## 4. v1/v2/v3 证据链

| Run | 状态 | 唯一变化 | 结论 |
|---|---|---|---|
| `route-pivot-a0-real-motion-s20260718-v1` | checker bug preserved | 初始冻结协议 | 把部分可见但 3D center 已出画的 box 也放入 clipped-xyxy 分母，得到 `0.9596 < 0.98`；34/34 failures 全是 center 出画，图内 failure 为 0，不构成 target reject |
| `route-pivot-a0-real-motion-s20260718-v2` | machine pass / superseded | 只修正 center-in-image eligibility；所有阈值、场景、RAFT 不变 | 808/808 图内 centers 在 box 内，34 个 offscreen centers 单独报告；全部原门槛通过 |
| `route-pivot-a0-real-motion-s20260718-v3` | machine pass / awaiting reviews | 在 A1 前再要求 actual_t、actual_t+1、static_t+1 三点共同图内，并加 `localizable_fraction >= 0.85` | 共同支持仍有 392 pairs / 89 tracks，全部最终机器门槛通过；这是 A0 当前事实源 |

v1 不是研究失败，也没有覆盖；修复由 synthetic regression tests 固化。v3 没有降低原 `AUC=0.75`、
projection `0.98`、background `0.70` 或 finite `0.95` 阈值。

## 5. v3 结果

| Gate | 冻结阈值 | 结果 | 状态 |
|---|---:|---:|---|
| finite actor targets | ≥ `0.95` | `420/421 = 0.9976` | pass |
| localizable common support | ≥ `0.85` | `392/421 = 0.9311` | pass |
| valid paired actor tracks | ≥ `32` | `89` | pass |
| moving / stationary support | 各 ≥ `8` pairs | `181 / 208` | pass |
| moving vs stationary residual AUC | ≥ `0.75` | `0.8600` | pass |
| in-image center projection-in-box | ≥ `0.98` | `808/808 = 1.0000` | pass |
| velocity projected direction positive | ≥ `0.60`，至少 16 pairs | `354/364 = 0.9725` | pass |
| abs(residual, ego speed) Spearman | ≤ `0.50` | `0.2226` | pass |
| sparse ego vs RAFT ≤45° agreement | ≥ `0.70`，至少 256 points | `0.9870` / `157,394` points | pass |
| minimum LiDAR points / frame | ≥ `64` | `2,518` | pass |

共同支持内 moving residual speed 中位数为 `5.9944 px/s`、p95 为 `42.8201 px/s`；stationary 分别为
`0.7430 px/s`、`4.1152 px/s`。AUC 使用 pair-level ranks，极端幅值不会按大小线性支配结果。

## 6. Reviewer 预判

### 6.1 “moving attribute 与 annotation motion 同源，AUC 是否循环论证？”

是同一真实标注体系内的 semantic sanity，不是外部 ground-truth benchmark。A0 只用它拒绝明显错误的符号、
单位和补偿定义；A1 必须在 scene-disjoint holdout、label shuffle、single-frame 与 no-temporal controls 下证明
SVD representation 可辨，不能把 AUC 直接写成方法效果。

### 6.2 “共同支持会不会只留下容易对象？”

这是正式 open risk，而非已解决问题。v3 报告保留率 `93.11%`、moving/stationary 数与 unique tracks；
34 个 center 出画 observations 没有删除，只是不进入 clipped-box 分母。A1 必须继续报告 category、depth、
visibility、speed 与 scene coverage，不能只汇总 probe accuracy。

### 6.3 “RAFT 一致率很高是否说明 target 与 evaluator 同源？”

不是。primary background target 是 LiDAR + calibration 的解析几何；RAFT 只读真实 RGB，且只验证方向，
不参与 target 构造、actor residual、生成评价或后续 label。该结果不证明绝对 flow magnitude 正确。

### 6.4 “低 residual–ego 相关是否已解决 driving motion entanglement？”

没有。`0.2226` 只排除 residual 被 ego translation speed 单变量完全支配。转弯、深度、遮挡、类别和 scene
仍可能纠缠；真正的 claim 需要 A1 representation controls 和 A2 held-out rollout safeguards。

### 6.5 “为什么 machine pass 仍不 promotion？”

A0 panel 中需要人工确认 3D box、actual/static arrow、遮挡与背景 flow 的语义对应；机器门槛无法发现全部
标注/投影错位。A1 machine scan 与该人审独立，可以继续；A1 confirm/A2 的最终 promotion 仍受人审约束。

## 7. 人工复核材料

```text
/root/autodl-tmp/runs/route-pivot-a0-real-motion-s20260718-v3/review/REVIEW_PROMPT.md
/root/autodl-tmp/runs/route-pivot-a0-real-motion-s20260718-v3/review/cases.jsonl
/root/autodl-tmp/runs/route-pivot-a0-real-motion-s20260718-v3/review/reviews.template.jsonl
/root/autodl-tmp/runs/route-pivot-a0-real-motion-s20260718-v3/panels/
```

共 12 个 panels；模板所有 verdict 保持 `null`，不得由 autoresearch 代填。

## 8. 可复现证据

```text
/root/autodl-tmp/runs/route-pivot-a0-real-motion-s20260718-v3/
  manifest.json
  resolved.yaml
  clips.jsonl
  schema_audit.jsonl
  actor_residual_targets.jsonl
  background_ego_flow_audit.jsonl
  panel_manifest.jsonl
  metrics.jsonl
  result.json
  summary.json
  review/
  panels/
  COMPLETE
```

RAFT checkpoint SHA256 为
`ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322`；正式 run 在 clean commit
`45cb279` 上完成，`generation.fps=7` 继承 R1 冻结协议，但 A0 本身没有生成视频或训练模型。

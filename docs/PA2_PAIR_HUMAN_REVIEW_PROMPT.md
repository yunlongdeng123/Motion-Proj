# PA2-PAIR-03 两阶段人工偏好评测提示词

你是 PA2 的人工视频偏好评测员。目标是检验：由 RAFT-chain + P-UNC 给出的 pair 偏好，是否与人对短时驾驶视频的物理合理性判断一致，并检查自动 chosen 是否通过“少动/静止”或灾难性画质作弊。本评测不训练模型、不修改自动分数，也不要求判断论文贡献。

评测目录固定为：

```text
/root/autodl-tmp/runs/autoresearch-pa2-pair-s20260715-v1
```

共评测 `pa2-pair-review-000` 至 `pa2-pair-review-047`，必须完成 48 个 case。每个 case 是同一 condition 下的 A/B 两条真实 SVD rollout；A/B 顺序已按固定 seed 随机化。

## 1. 允许和禁止使用的证据

阶段 A 只允许查看：

```text
review/stage_a_blind/*.mp4
reviews.template.jsonl
本提示词
```

阶段 A 完成并保存后，阶段 B 才允许查看：

```text
review/stage_b_overlay/*.mp4
review/stage_b_diagnostics/*.json
```

整个评测期间禁止打开或据此填写 verdict：

```text
review_cases.private.json
preferences.jsonl
constructor_pairs.jsonl
constructor_segments.jsonl
candidate_diagnostics.jsonl
constructor_diagnostics.jsonl
machine_summary.json
summary.json
任意 candidate score.json / trace.json / latent 文件
```

禁止根据文件名、candidate ID、正负 branch、constructor 名称或自动 global winner 猜答案。阶段 B JSON 中的 P-UNC 分解和 local segment label 是允许的 adjudication 证据，但不是标准答案；CoTracker3 overlay 也可能因遮挡、低纹理或 identity switch 失败，必须结合原视频判断。

## 2. 明确的非目标

- 不评估视频是否与数据集 future GT 一致；评测材料不提供 future GT。
- 不偏好“运动越少越好”或“画面越锐越好”。
- 不因为 A/B 运动量不同就自动选择运动更小的一侧。
- 不把小幅可接受的随机未来差异判成 identity/composition failure。
- 不代替机器修改 P-UNC、CoTracker3、margin 或 feasibility 阈值。

## 3. 阶段 A：只看盲视频

按 case_id 顺序打开 `review/stage_a_blind/<case_id>.mp4`。可循环、暂停和逐帧查看。先独立完成全部 48 个阶段 A 字段并保存 `reviews.jsonl`，再打开任何阶段 B 文件。

综合判断优先级如下：

1. 灾难性无效：严重黑屏、大片破碎、主体/道路不可辨认、首帧明显错误。
2. 身份与几何连续性：车辆、行人、道路边界和主要物体不得跳变、复制、融化或换身份。
3. 运动物理合理性：速度/加速度变化应连续；不应出现无因瞬移、方向突变、强烈抖动、漂浮或不合理形变。
4. 相机—背景一致性：背景与自车视角的共同运动应连贯，静态结构不应独立滑动。
5. 视觉质量：模糊、闪烁、纹理破坏等只在影响可用性或运动判断时决定胜负。
6. 运动量：用于识别静止化或运动压制，不作为“越少越好”的奖励。

`stage_a_verdict` 合法值：

- `a_better`：A 在上述综合标准上有可重复辨认的优势，且 A 本身不是灾难性无效。
- `b_better`：B 有对应优势，且 B 本身不是灾难性无效。
- `tie`：两侧均有效，差异不足以稳定决定，或各有优劣且总体无法排序。
- `both_invalid`：两侧都有灾难性质量/身份/运动失败，无法形成有意义偏好。不要把“两侧都普通”写成该值。
- `uncertain`：由于压缩、遮挡、运动过小或证据冲突，人工确实无法判断。它不是省略评测的选项。

边界例：

- 一侧略锐但车辆明显瞬移，另一侧稍模糊但运动连续：通常选择后者。
- 一侧几乎冻结而画面稳定，另一侧保持合理运动且无灾难性伪影：不得因“更稳”选择冻结侧。
- 两侧都保持同一场景，仅远处纹理有轻微差异且运动同样合理：`tie`。
- 一侧有局部短暂 tracker overlay 未来可能失败，但阶段 A 原视频连续：阶段 A 不得预判 overlay，更不能据此判 invalid。

阶段 A 还必须填写：

- `stage_a_motion_plausibility.a/b`：`pass`、`fail`、`uncertain`。
- `stage_a_visual_quality.a/b`：`pass`、`fail`、`uncertain`。
- `stage_a_identity_consistency.a/b`：`pass`、`fail`、`uncertain`。
- `stage_a_motion_amount`：`a_more`、`b_more`、`similar`、`neither_moves`、`uncertain`。这里只描述运动量，不表达偏好。
- `stage_a_failure_reasons.a/b`：零个或多个以下值；无明显失败时用空数组：
  - `physics_implausible`
  - `temporal_jitter`
  - `geometry_deformation`
  - `identity_switch`
  - `camera_motion_inconsistent`
  - `low_motion_or_frozen`
  - `blur_or_artifact`
  - `other`

## 4. 阶段 B：诊断 adjudication

确认全部阶段 A 已保存后，对同一 case 查看：

```text
review/stage_b_overlay/<case_id>.mp4
review/stage_b_diagnostics/<case_id>.json
```

overlay 只来自已通过 E0 的独立 CoTracker3 first-frame grid，不复用 RAFT/P-UNC 训练 scorer。诊断 JSON 提供：

- A/B 的 CoTracker3 coverage、survival 和 camera-compensated dynamics；
- A/B 的 P-UNC projection energy、confidence、coverage、motion 与 displacement；
- 4 帧滑窗 local segment labels；
- 不提供 private candidate ID 或自动 global winner。

检查 overlay 点是否贴合相同纹理、遮挡后是否合理消失、是否发生 identity switch。P-UNC projection energy 越低只表示该轨迹需要的置信度归一化修正较小；如果较低分来自冻结、track 丢失、低 coverage 或灾难性画质，不得盲从。

填写 `stage_b_verdict`，合法值与阶段 A 完全相同。允许维持或改判：

- 若 `stage_b_verdict == stage_a_verdict`，`stage_b_change_reason` 可为空。
- 若二者不同，`stage_b_change_reason` 必须具体说明哪些 overlay/segment/coverage 证据导致改判；只写“看了诊断”不够。
- tracker 明显失败时，应在 reason/notes 中记录，并以原视频为主，不要把 tracker failure 当视频 failure。

阶段 B 还必须对 A/B 各填写两个布尔值：

- `low_motion_collapse.a/b`：仅当视频相对场景应有运动呈现不合理冻结、运动被压制到失真，或以少动明显规避物理错误时为 `true`。自然低动态但合理的视频为 `false`。
- `catastrophic_quality_failure.a/b`：仅当视频因严重画质、身份、几何或时序破坏而不可用于偏好训练时为 `true`。

## 5. JSONL 填写规则

先复制模板，严禁由 Codex/脚本填写人工 verdict：

```bash
cd /root/autodl-tmp/motion_proj
cp /root/autodl-tmp/runs/autoresearch-pa2-pair-s20260715-v1/reviews.template.jsonl \
   /root/autodl-tmp/runs/autoresearch-pa2-pair-s20260715-v1/reviews.jsonl
```

每行对应一个 case，不得增删、重复或改写 `case_id`。JSON key 中的小写 `a/b` 对应视频上的大写 A/B。最终每行格式为：

```json
{"case_id":"pa2-pair-review-000","stage_a_verdict":"a_better","stage_a_motion_plausibility":{"a":"pass","b":"fail"},"stage_a_visual_quality":{"a":"pass","b":"pass"},"stage_a_motion_amount":"similar","stage_a_identity_consistency":{"a":"pass","b":"pass"},"stage_a_failure_reasons":{"a":[],"b":["temporal_jitter"]},"stage_b_verdict":"a_better","stage_b_change_reason":"","low_motion_collapse":{"a":false,"b":false},"catastrophic_quality_failure":{"a":false,"b":false},"reviewer":"你的人工标识","notes":"可为空；若 other 或 tracker failure，需在此解释"}
```

规则：

- `reviewer` 必须是真实人工标识，不得为 `pending`、`Codex` 或 `AI`。
- `notes` 和 `stage_b_change_reason` 必须是字符串。
- 所有 `pending` 和 `null` 必须被替换为合法值。
- failure reason 不得重复；使用 `other` 时在 `notes` 解释。
- 不得让脚本根据自动 label 补全任何人工字段。

## 6. 预注册聚合与门槛

完成后运行：

```bash
cd /root/autodl-tmp/motion_proj
/root/autodl-tmp/envs/motionproj/bin/python \
  -m motion_proj.diagnostics.physics_dpo_pair \
  --config configs/diagnostics/physics_dpo_pair.yaml \
  --aggregate-only
```

聚合只读取已有 machine artifact 与人工 `reviews.jsonl`，不加载 GPU、不重生成候选、不改变 verdict。

定义 eligible agreement case：机器 global label 为 `a_wins/b_wins`，且人工阶段 B 为 `a_better/b_better`。`tie`、`both_invalid`、`uncertain` 和机器 abstain/invalid 不计入 agreement 分母，但仍计入 48-case 完成数及 failure 审计。

```text
agreement_rate = scorer winner 与 stage_b winner 一致数 / eligible agreement case 数
Wilson = 该二项比例的双侧 95% Wilson 下界
```

全部通过才解锁 PA3：

- 完成 case 数 ≥ 48；
- eligible agreement case 数 ≥ 24；
- agreement rate ≥ 0.75；
- 95% Wilson lower bound 严格 > 0.50；
- 自动 scorer chosen 的 `low_motion_collapse` 总数 = 0；
- 自动 scorer chosen 的 `catastrophic_quality_failure` 总数 = 0；
- 每个阶段 A→B 改判都有非空具体理由。

如果 48 个 case 已完成但 eligible 数不足 24，状态是 `needs_more_reviews`：扩大 review，不降低阈值。如果 eligible 足够但任一质量、agreement 或 Wilson 门槛失败，状态是 `rejected`，PA3/DPO/AWR/训练与双卡均继续阻断。如果全部通过，状态为 `done`，只解锁单卡 `PA3-KERNEL-04` 的 1/8/32-pair 代数与容量测试；仍不直接授权正式训练或切双卡。

# PA1-BRANCH-02 v5 结构对齐盲审：完整人工评测提示词

> **评测对象**：`/root/autodl-tmp/runs/autoresearch-pa1-branch-s20260715-v5`
>
> **任务**：`PA1-BRANCH-02`；本次只评估 common-prefix sibling 是否仍是同一驾驶场景的合法不同未来。
>
> **重要**：这不是物理运动偏好评测，不选择 winner，也不判断哪一列更真实、更平滑、更动态或画质更好。

## 给评审者的提示词

请独立观看 8 个 `panels/*.mp4`。每个 panel 有三列同步视频，列名为 A、B、C；它们来自同一条件的一个 Anchor 和一对 sibling。三列顺序已固定随机化，评测时**不知道也不应尝试推断**哪一列是 Anchor、positive 或 negative。

你的唯一问题是：**三列是否仍表示同一个驾驶场景、同一初始条件和同一批主体；两条 sibling 是否只是该场景在后续时间上的可接受分歧？**

允许 sibling 在后续帧有不同的车辆运动、相机运动、物体位置、遮挡细节或纹理细节。它们不需要逐像素相同，也不需要具有相同的运动幅度。不要把“运动更少”“更平滑”“更好看”“更符合物理直觉”作为本任务的依据。

### 盲法与禁止事项

在完成全部 8 条 verdict 前，只能查看以下内容：

- 本提示词；
- `panels/branch-review-00.mp4` 至 `panels/branch-review-07.mp4`；
- `reviews.template.jsonl`（仅用于填写格式）。

不要查看或使用 `review_cases.json`、`candidate_manifest.jsonl`、`candidate_diagnostics.jsonl`、`pairwise_distances.jsonl`、`profile.json`、`machine_summary.json`、自动 metric、track overlay、候选 ID 中的角色词，或任何自动 winner / physics score。不要要求或尝试把 A/B/C 映射回生成分支。不要和自动指标混合判断。

### 每个 panel 的观看方法

1. 从第一帧到最后一帧完整播放一次；必要时再回放一次。
2. 先比较第一帧和稳定背景：道路拓扑、车道、相机视角、主要车辆/行人/交通设施、主体相对位置是否一致。
3. 再比较后续帧：允许未来运动不同，但应仍能看作同一场景从同一初始条件延续而来。
4. 只给该 panel 写一个 group-level verdict，不能给 A/B/C 分别打分，也不能选择 preferred video。

### Verdict 定义与优先级

按以下顺序选择**唯一**一个值。

1. `invalid`：至少一列存在灾难性渲染或首帧损坏，以至于无法判断它是否属于同一场景。例如黑屏/大面积损坏、首帧明显变形或不对应条件、持续严重撕裂/闪烁导致道路或主体身份不可辨认。轻微压缩噪声、局部小伪影或后续帧正常的轻微纹理漂移不构成 `invalid`。
2. `different_composition`：视频可观看，但至少一列明显不再是同一场景或同一初始条件的合理延续。例如道路/车道/相机位置或朝向明显换景；主要车辆、行人、交通灯、建筑或主体身份被替换；第一帧/早期帧构图明显不同；物体关系发生无法由正常运动或遮挡解释的重排。只要其中一列有这种明确变化，即选择此项。
3. `same_scene`：三列保持同一驾驶场景、相同的条件首帧和可识别主体；后续差异可理解为同一场景的不同未来。即使一列运动更弱、更强、略有局部伪影或画质稍差，只要没有达到前两项，仍选择此项。
4. `uncertain`：完整观看后仍无法可靠判断，且原因不是“我更喜欢另一列”“运动不够物理”或“我不想作决定”。例如长时间遮挡、边界性身份变化或画面信息确实不足。请在 `notes` 简短说明不确定原因。

若同时存在明确换景和严重损坏，但严重损坏使换景本身无法可靠判断，优先选 `invalid`；若换景已清晰可见且画面仍可判断，选 `different_composition`。

### 需要填写的 8 个 case

依次评审并填写：

```text
branch-review-00  → panels/branch-review-00.mp4
branch-review-01  → panels/branch-review-01.mp4
branch-review-02  → panels/branch-review-02.mp4
branch-review-03  → panels/branch-review-03.mp4
branch-review-04  → panels/branch-review-04.mp4
branch-review-05  → panels/branch-review-05.mp4
branch-review-06  → panels/branch-review-06.mp4
branch-review-07  → panels/branch-review-07.mp4
```

## 填写与提交格式

在 run 目录中复制模板：

```bash
RUN=/root/autodl-tmp/runs/autoresearch-pa1-branch-s20260715-v5
cd "$RUN"
test ! -e reviews.jsonl && cp reviews.template.jsonl reviews.jsonl
```

`reviews.jsonl` 必须保留 8 行、每个 `case_id` 恰好一次。不要修改 `case_id`、`rubric` 或删除字段。`verdict` 只能是：

```text
same_scene
different_composition
invalid
uncertain
```

一行示例（示例 verdict 并不代表任何实际 panel 的答案）：

```json
{"case_id":"branch-review-00","verdict":"same_scene","reviewer":"human","notes":"三列首帧、道路和主要车辆一致；后续运动存在可接受差异。","rubric":"三列是否保持同一驾驶场景布局、主体身份和条件首帧；两条 sibling 是否只是同场景的不同未来，而非不同构图或灾难性失真？"}
```

`reviewer` 保持 `human` 或填写你指定的匿名评审标识；`notes` 可留空，但对 `different_composition`、`invalid` 和 `uncertain` 应填写一条简短、可观察的原因。不要在 notes 中猜测 Anchor/sibling 角色或引用自动指标。

## 聚合规则与后果

所有 8 条均填写后，聚合器按以下预注册规则判断：

- `completed_cases = 8`；
- 去除 `uncertain` 后，`same_scene / decisive_cases >= 0.875`；
- `different_composition + invalid = 0`；
- 至少要有一条 decisive verdict（不能全部为 `uncertain`）。

因此，任意一条 `different_composition` 或 `invalid` 都会使本次 PA1 结构对齐 review 不通过。`uncertain` 只在确实无法判断时使用；它不应替代认真判定。

填写完成后请通知 Codex。Codex 将在**不使用 GPU、不重生成视频**的前提下运行：

```bash
cd /root/autodl-tmp/motion_proj
/root/autodl-tmp/envs/motionproj/bin/python -m motion_proj.diagnostics.physics_dpo_branch \
  --config configs/diagnostics/physics_dpo_branch_v5.yaml \
  --aggregate-only
```

通过后才会解锁 PA2 的单卡 pair-legality 工作；未通过则 PA1-BRANCH-02 停止，不会以更多 GPU、更多数据或自动评分绕过本人工门槛。无论通过与否，本评测本身都不会直接产生 chosen/rejected 物理偏好标签。

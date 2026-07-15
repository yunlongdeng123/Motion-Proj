# SAP-DPO PA0：人工复核、scene split 与 schema

> 日期：2026-07-15
>
> 上位计划：[PHYSICS_DPO_AUTORESEARCH_PLAN_V2_AC_REVISED.md](PHYSICS_DPO_AUTORESEARCH_PLAN_V2_AC_REVISED.md)
> 状态：`PA0-REVIEW-00 done`、`PA0-SCENE-SPLIT-01 done`。PA0 aggregation、scene split/schema 与正式 split materialization 均已完成；当前只授权 PA1 的 Base guard horizon profile，不授权 sibling candidate、DPO/AWR 实现、训练、cache 扩展或切换双卡。

## 1. 当前证据与 PA0 边界

PA0 aggregation 的 source worktree 为 `16b6975`（当时 dirty 状态已在 manifest 保留）。正式 scene split 在 clean `5713267` worktree 写入。以下两个已有 run 的机器门槛已通过；用户提供的 review 文件均已完整填写，原始 reviewer ID 与 notes 均已逐字节复制到新的 PA0 decision run：

| 角色 | 冻结 run | 机器证据 | 人工材料 | PA0 所需人工结论 |
|---|---|---|---|---|
| pair label scorer | `autoresearch-p0-projector-s20260714-v1` | P-UNC 是唯一 machine-eligible candidate；无 future-GT | 12 张 `panels/*.png` | 12/12 `valid`；达到 V2 的 11/12 门槛 |
| independent evaluator | `autoresearch-e0-evaluator-s20260714-v3` | CoTracker3 rerun exact、synthetic/perturbation ranking 通过；无 RAFT/P0/P1 输入 | 8 段真实 + 4 段 synthetic `track_overlay/*.mp4` | 12/12 `valid`；达到 V2 的 10/12 门槛 |

历史 P1 RGB/VAE target 已 machine fail，不能被 PA0 review 覆盖；SAP-DPO 只把 P-UNC 用作 no-grad pair scorer，绝不恢复 P1 renderer 或 hybrid target。

## 2. 人工复核操作与不可变性

复核者应只看对应面板，先写自己的 verdict，再查看机器摘要。每一行保留原始 `case_id`，填写：

```json
{"case_id":"...","verdict":"valid | invalid | uncertain","reviewer":"<匿名标识或姓名>","notes":"可复核的失败原因"}
```

`uncertain` 是合法结果，不能改写为 `valid` 以凑门槛；没有有效 tracking 的情形也不能以数值 0 代替。不得修改 `reviews.template.jsonl`、`manifest.json`、`machine_summary.json` 或任何已有 panel。

### 2.1 P-UNC review

材料与模板：

```text
/root/autodl-tmp/runs/autoresearch-p0-projector-s20260714-v1/
  panels/
  reviews.template.jsonl
  REVIEW_README.md
```

逐一判断白色 projected path 是否仍贴合可见局部、是否保留合理加减速/转弯、是否出现无图像支持的跳变。PA0 的 V2 判定要求 12 条均被填写，并且至少 11/12 为 `valid`；同时不得出现系统性静止化、frame-0、visibility 或 support 违规。

### 2.2 CoTracker3 review

材料与模板：

```text
/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v3/
  track_overlay/
  reviews.template.jsonl
  REVIEW_README.md
```

逐一判断大多数 overlay point 是否贴合纹理，遮挡或低纹理点是否消失而不是伪造为连续轨迹。PA0 的 V2 判定要求 12 条均被填写、至少 10/12 为 `valid`，并且 identity switch / occlusion failure 能被识别。

> 说明：历史 E0 v3 配置中的 `minimum_valid_rate=0.875` 比 V2 的 10/12 PA0 规则更严格。不得事后修改历史 run 的 resolved config 或重写其 summary。人工表填写完成后，应创建一个新的 PA0 decision artifact，分别按 V2 的 P-UNC 11/12、E0 10/12 规则聚合，并保留原始 v3 machine evidence；不要直接把旧 v3 的 `--aggregate-only` 输出冒充 V2 决策。

### 2.3 PA0 决策产物（人工输入到位后才创建）

新的、不可复用 run 已保存：

```text
manifest.json                 # 引用两个源 run、源 manifest/summary SHA256
resolved.yaml                 # 只含 PA0 聚合阈值，不含模型或训练参数
reviews.p0.jsonl              # 对原始人工表的逐行只读复制
reviews.e0.jsonl
review_decision.json          # 每个 verdict、decisive/valid 计数、失败原因
summary.json
metrics.jsonl                 # 可为空的结构化 review 事件，不伪造数值指标
COMPLETE
```

产物：`/root/autodl-tmp/runs/autoresearch-pa0-review-s20260715-v1/`。其 `summary.json` 为 `done`，P-UNC/E0 的 valid count 分别为 `12/12` 与 `12/12`；对应 V2 门槛为 `11/12` 与 `10/12`。`manifest.json` 锁定两个源 run 的 manifest/summary/review SHA256，且源 P0/E0 run 没有被回写。任一未来重新聚合若不满足规则即为 `blocked`，不以更多 GPU、更多 candidate 或实现 DPO 绕过。

## 3. 冻结的 scene-level split 设计

本节的预注册已执行：正式 run 只读取两个官方 manifest，不写入数据目录、不生成 candidate，也不改变已有官方 manifest。

### 3.1 来源与确定性算法

来源是已完成的官方 scene-level manifest：

| 官方来源 | 原始 scene / clip | source split fingerprint |
|---|---:|---|
| nuScenes `train` / CAM_FRONT / 8 frames | 700 / 3425 | `5d7cc689e109e07fe3c404face1a276645d1bbc27c9ed455d7f3cba55b9ce83e` |
| nuScenes `val` / CAM_FRONT / 8 frames | 150 / 732 | `72ce633be361945e8af61a639576a32c409250cc08db548c1b33d6635b8f6bf0` |

使用固定 salt：

```text
sap-dpo-v2-pa0-scene-freeze-20260715
```

在每个官方 split 内，按下式对 `scene_token` 排序：

```text
SHA256("sap-dpo-v2-pa0-scene-freeze-20260715|" + scene_token)
```

选择 hash 最小的 scene 进入较小 partition。使用 token 而非 dataset index，避免未来 clip 排序或抽样变化改变 scene 归属。

| partition | 来源 | scene / clip | canonical scene-list fingerprint | 允许用途 |
|---|---|---:|---|---|
| `preference_train` | official train | 630 / 3079 | `39f0572fa114413c5400d69b8bcc0afb345c4e9174c7c8eda3c097f9a67e80f6` | PA1–PA4 candidate、pair 与训练数据 |
| `preference_dev` | official train | 70 / 346 | `0b3da555d947e964141e09d89f7232cbeb901e101a3abe13fd9aedcf883a08c1` | fork/strength、margin、scorer 校准；绝不进入训练 |
| `screen_eval` | official val | 32 / 155 | `d1dc3c3b82e92026ce4d05e49aff7eccf18c2e372b81aa44a850c81ad719b0e7` | PA4 的预注册 16 clip screening；不进入训练或阈值拟合 |
| `formal_test` | official val | 118 / 577 | `92f0cf7f32d3d7298f7cafe3f9fd4f894f9178bb6df79ccc1e2ee5e49f2c63fc` | PA6/PA7；保留至少 256 clip 的正式评估容量 |

> 2026-07-15 定义了此前未指定的 canonical serialization：`scene_list_fingerprint=SHA256(canonical JSON(partition, source split/fingerprint, salt, ordered scene tokens, ordered clip IDs))`。上表替代只读设计阶段未定义 serialization 的预计算 hash；scene selection rule、source fingerprint、salt、scene/clip count 均未改变。四个 partition 的总 split fingerprint 为 `e525edf33bcfec169c0077d2eb2e528d953dbc9930e771c803c889a32983c73a`。

正式 materialization 证据为：

```text
/root/autodl-tmp/runs/autoresearch-pa0-scene-split-s20260715-v1/
  manifest.json                 # git.dirty=false，source file SHA256
  resolved.yaml
  scene_split_manifest.json     # split fingerprint e525edf33bcfec169c0077d2eb2e528d953dbc9930e771c803c889a32983c73a
  metrics.jsonl                 # 四个 partition 的 scene/clip count
  summary.json                  # status=done, next_gate=PA1-HORIZON-01
  COMPLETE                       # SHA256(summary.json)
```

四个 partition 的 scene token 必须两两不重叠；`screen_eval` 与 `formal_test` 均来自 official val 且相互不重叠。每个 run 保存 source manifest fingerprint、salt、完整 scene-token list、clip list、selection rule、scene/clip counts 和生成后的 manifest fingerprint。若任何 source fingerprint、8/14-frame profile 所选 frame 数或 camera 改变，必须新建 split 版本，不能静默沿用本表。

## 4. 条件、candidate 与 preference schema

PA0 后的数据应保存在 run/cache 数据盘而非 Git；所有 JSONL 以一行一个不可变实体保存。训练代码只能读取通过 schema validator 的 records，缺失字段、future-GT、fingerprint 不匹配、重复 ID 或跨 scene split 都必须 fail closed。

### 4.1 `conditions.jsonl`

每个 condition 是统计与抽样基本单位，而不是 generation seed。最小字段：

```text
schema_version, condition_id, scene_id, scene_token, clip_id, split,
camera, conditioning_frame, condition_frame_sha256, num_frames, fps,
generation_protocol=svd_official_v1, scheduler_fingerprint,
base_model_fingerprint, uses_future_gt=false, git_commit, config_fingerprint
```

`condition_id` 由 schema version、scene token、clip ID、camera、frame profile 和 official condition fingerprint 计算；同一 scene 的任何 clip 都不得跨 split。

### 4.2 `candidates.jsonl`

每条记录都是一次真实 SVD 采样并已解码的 RGB video。Base guard 和 sibling 均需要：

```text
candidate_id, condition_id, scene_id, split, candidate_role,
rgb_video_path, vae_latent_path, diagnostics_path,
generation_protocol, base_model_fingerprint, scheduler_fingerprint,
initial_latent_hash, prefix_latent_hash, prefix_trace_hash,
fork_step, branch_family, branch_direction, branch_strength,
perturbation_rms, antithetic_group_id, generation_seed,
guidance_schedule, num_frames, fps, uses_future_gt=false,
git_commit, config_fingerprint
```

约束：一条 `base_guard` 加四条 sibling 共享同一 `condition_id`、initial latent 和 prefix trace；四个 sibling 使用相同 scheduler/condition/decode，成对反向、等 RMS、零均值 perturbation。Base guard 必须由 C0 parity 验证；任何 candidate 都不能引用 P1 RGB target、hybrid latent 或 projected cache。

### 4.3 `preferences.jsonl` 与 `segments.jsonl`

`preferences.jsonl` 只描述同 condition 的两个真实 candidate；`segments.jsonl` 描述同一 pair 的局部窗口，不把 tie/abstain 强行转为 winner。最小字段：

```text
pair_id, condition_id, candidate_a, candidate_b, split,
global_label, winner_candidate_id, loser_candidate_id,
feasibility_a, feasibility_b, physics_components, quality_components,
preference_margin, pair_confidence, scorer_fingerprint,
human_review_id, abstain_reason, uses_future_gt=false
```

```text
segment_id, pair_id, start_frame, end_frame, label, winner_candidate_id,
loser_candidate_id, confidence, violation_decomposition,
frame_alignment_pass, abstain_reason
```

只有 `feasible/feasible`、Pareto + non-inferiority 通过、且 `pair_confidence > 0` 的 `winner/loser` record 才能进入 DPO；`tie`、`abstain`、invalid candidate 与任何跨 condition pair 永远不进入 loss。每个 condition 最多一个 global pair，训练 sampler 必须先均匀采 condition，再采 pair/segment，并记录 per-condition exposure p95。

## 5. PA0 实现、测试与正式 materialization

已新增不访问 GPU/RGB 的 `motion_proj.data.physics_dpo_schema`，并在 `tests/test_physics_dpo_schema.py` 覆盖以下 fail-closed 条件：

1. 官方 train/val 与四个子 split 的 token 完全不重叠；
2. 重跑同一 source manifest、salt 后得到 exact scene/clip list fingerprint；
3. condition 不能跨 scene split，candidate/pair 不能跨 condition；
4. `uses_future_gt=true`、P1/hybrid target 引用、缺 provenance、重复 ID 与未知 branch family 均 fail closed；
5. sibling 的 initial/prefix hash、scheduler、condition、antithetic RMS 与 branch direction 约束正确；
6. tie/abstain segment 与 all-tie pair 被 dataset 拒绝，不能以零损失静默吞掉；
7. 所有 split、candidate、pair、segment 的 canonical JSON fingerprint 被写入 manifest。

验证命令为 `PYTHONPATH=. /root/autodl-tmp/envs/motionproj/bin/python -m pytest -q tests/test_physics_dpo_pa0_review.py tests/test_physics_dpo_schema.py tests/test_split_manifest.py tests/test_replay_review.py`，结果 `11 passed`。正式 materializer 已检查 source split fingerprint 并在 clean worktree 完成上述 run；未实现 DPO/AWR、未构造 preference cache、未开始 horizon/branch GPU profiling。

## 6. 下一步

唯一下一步是 `PA1-HORIZON-01`：在冻结的 scene split 内执行 `2 conditions × {8,14} frames × Base guard only`。必须先报告 peak VRAM、generation/score time、track coverage、metric repeatability 与 storage/video；冻结 horizon 和 claim scope 后才进入 `PA1-BRANCH-02`。不因 worktree 或硬件状态绕过这一顺序。

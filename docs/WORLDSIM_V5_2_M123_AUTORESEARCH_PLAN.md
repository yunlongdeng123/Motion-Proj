# WorldSim V5.2 — M1/M2/M3 Autoresearch 正式计划

> 状态：`pending`
>
> 任务族：`WS-V52-R0..R7`
>
> 执行授权：允许单卡 RTX 3090 连续无人值守运行一晚，最长 `12 h`；正常科研分支和工程恢复不再向用户逐项确认。
>
> 机器合同：`configs/worldsim_v52/m123_autoresearch_v1.yaml`
>
> 人工归因合同：`docs/run_manifests/worldsim-v5.2.1-human-review-attribution-v1/`

## 0. 结论先行

V5.2 不推倒 M1/M2/M3，而是补上此前缺失的因果边界：

```text
P0 Base Validity Gate
        ↓ pass
M1：Who owns the Gaussian?
        ↓ uncertainty / actor sidecar
M3：Where should the actor be at t?
        ↓ candidate edit
M2：Is the edit safe to execute?
        ↓
execute / abstain
```

正式定位冻结为：

| 模块 | V5.2 定位 | 本轮允许做什么 | 本轮禁止做什么 |
|---|---|---|---|
| P0 | 实验因果边界，不是论文模块 | 排除 global-collapse 污染，冻结 eligible denominator | 用 P0 结果替换基座或删除失败样本 |
| M1 | 核心研究模块 | 建立 causal bridge；测试 actor track、时序 visibility、canonical model selection 与 Bayesian UNKNOWN | Graph/KNN/BKI/空间扩散；从无观测处制造正负证据 |
| M3 | 已有历史正结果的 downstream 核心候选 | 在相同 badcase 上做 `unwarped / flow-warped / pose-warped` 因果诊断并 replay 冻结 V4 delta | 因 panel 症状相似就直接重写 M3；把 V5 constraint route 复活 |
| M2 | 风险控制与 abstention 安全层 | 对已经产生的 M1/M3 edit 做执行/拒绝策略 | 把 router 当作缺失信息的 repair；在 geometry undefined 时宣称 geometry-safe |

核心论文假设是：

> Driving Gaussian ownership is a temporally incomplete actor-association problem, not a spatial propagation problem.

V5.2 的主方法 working name 为 `TrackBayes-GS`；`TubeBayes-GS` 作为 actor-tube 实现名保留。

## 1. V5.2.1 人工复核形成的新证据层

V5.2.1 原始 `BADCASE_REGISTRY.jsonl` 保持不可变。人工复核只新增 attribution layer，不改变原 threshold、failure predicate、Top-K、Discovery/Confirmation membership 或 case ID。

18 个 case 的冻结归因分母：

| research gate | 数量 | 用途 |
|---|---:|---|
| `BASE_FAILURE` | 9 | 基座失效哨兵，不进入 M123 primary denominator |
| `M123_ELIGIBLE` | 8 | 背景基本可用、动态 actor/boundary/ghost 有研究意义 |
| `ATTRIBUTION_UNRESOLVED` | 1 | global 与 dynamic 混合，先做 Base Validity/局部残差分解 |

8 个 `M123_ELIGIBLE` 全部来自 StreetGS；这不是“选 StreetGS 赢”，而是 AD-GS 的代表性 actor/boundary case 多数已被 global collapse 污染。

### 1.1 Design case（只允许这 5 个参与方法设计）

| Review # | Case ID | nuScenes scene/frame/camera | 主要问题 | 主模块 |
|---:|---|---|---|---|
| 05 | `BC-STREETGS-945caf2fc082` | `scene-0242 / 127 / c1` | 卡车拖影、形态错误 | M3，M1 次级 |
| 10 | `BC-STREETGS-6132ad736366` | `scene-0255 / 27 / c2` | 远距小 actor 弱或缺失 | M1 |
| 11 | `BC-STREETGS-84bf82336ee0` | `scene-0048 / 167 / c1` | 自行车透明/模糊/ghost | M1×M3 |
| 16 | `BC-STREETGS-b363a27e6231` | `scene-0048 / 47 / c2` | 施工人员/设备 boundary leakage | M1×M3 |
| 17 | `BC-STREETGS-4305955afdfd` | `scene-0139 / 47 / c1` | 行人透明拖影/重复 | M3，M1 次级 |

### 1.2 One-shot Confirmation case

| Review # | Case ID | nuScenes scene/frame/camera | 主要问题 | 用法 |
|---:|---|---|---|---|
| 06 | `BC-STREETGS-62640d591ebc` | `scene-0242 / 132 / c0` | 卡车轨迹/visibility/compositing | M3/M1 confirmation only |
| 12 | `BC-STREETGS-68c77ab5bc76` | `scene-0255 / 37 / c2` | 多个远距小 actor | M1 confirmation only |
| 18 | `BC-STREETGS-7e9c9ecf93da` | `scene-0048 / 17 / c0` | 动态边界与 visibility 耦合 | M1/M3 confirmation only |

这 3 个 case 在候选、阈值、指标、判词和 source hash 冻结前不得读取新算法质量，不得用于选 arm。

### 1.3 数据集与存储快照

逻辑数据集均为 `nuScenes trainval`，DriveStudio 10 Hz 三前向相机协议；不同目录只是历史加工快照：

| scene | scene index | 冻结快照 |
|---|---:|---|
| `scene-0048` | 45 | `/root/autodl-tmp/data/worldsim_v4/drivestudio_processed_10Hz/trainval/045` |
| `scene-0139` | 110 | `/root/autodl-tmp/data/worldsim_v4/drivestudio_processed_10Hz/trainval/110` |
| `scene-0230` | 179 | `/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval/179` |
| `scene-0242` | 191 | `/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval/191` |
| `scene-0255` | 204 | `/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval/204` |

每个 case 的 exact target path、split hash、target/prediction/mask/panel/metric SHA256 均在 `cases.jsonl`，后续不得靠 scene/frame 字符串重新猜路径。

## 2. 必须继承的失败事实

本计划不是旧路线重跑。正式 run 的 `failure_ledger_refs` 至少绑定：

- `V4-F39`：M1 development 正结果不能覆盖 scene-disjoint validation rejection；
- `V4-F42/V4-F47`：M2 selective routing 通过不等于 geometry 改善，full-denominator MAE 曾退化 `+3.3908096237 m`；
- `V4-F45/V4-F49`：M3 的可评与 abstain 分母必须同时报告；
- `V5-F31/F32/F33`：观测稀疏、graph 跨场不稳、boundary enrichment 不等于 boundary 是主因；
- `V5-F47/F48/F51`：没有 absolute geometry-safe repair，不得只用相对 proxy 提升复开 M2；
- `V5-F57/F59`：V5 constraint projection 信号不足并已 rejected，不得倒写 V4 M3；
- `V51-F63`：多视角 identity coverage/persistence 不足；
- `V51-F65`：Trace3D faithful alpha 跨 fresh process 不确定；
- `V51-F66`：Stage H/BKI 未执行而被 V5.2 scope 取代，不得写成 empirical reject；
- `V52-F01/F02`：global collapse 污染归因；8 个 eligible case 仍只有症状一致性，因果桥尚未通过。

因此 V5.2 的新增变量必须是 `temporal actor evidence / identity / visibility / canonical motion`，不能只是新的空间 kernel、更多 KNN 或 threshold 搜索。

## 3. 数据纪律与自动执行授权

### 3.1 数据层级

```text
Tier H：V4/V5/V5.1 历史证据，只作引用
Tier D：5 个 M123 Discovery design cases
Tier C：3 个 M123 one-shot Confirmation cases
Tier F：fresh validation/test/KITTI，整轮锁定 unread
```

Base-failure sentinels不进入 M123 primary aggregate，但每个候选 closeout 必须重放这些 case 的 gate，证明 router 不会错误接受它们。

### 3.2 无人值守规则

- 用户已授权一晚内连续执行，不因普通 package、单 case、CUDA 或缓存异常暂停询问；
- 工程故障最多在新 run ID 下自动最小修复重试一次，原失败 run 保留；
- quality-driven retry、改阈值、删 case、换 checkpoint、改 split 一律禁止；
- 各独立分支 fail-closed：M1 失败不阻止 M3 bridge；M3 失败不阻止 M2 合同审计；
- 任何需要新人工 verdict 的阶段只生成完整 review package 和 prompt，并将该分支标成 `blocked`，不得自动代填；
- 达到 `12 h`、输入 hash 漂移、split leak、NaN/Inf、Confirmation attempt 重用或第二次 OOM 时自动停止并原子写 terminal。

## 4. 正式 run 合同

每个 task 使用不可复用 run ID：

```text
YYYYMMDDTHHMMSSZ__<task-slug>-s<seed>-r<nnn>
```

每个 run 至少保存：

```text
resolved.yaml
manifest.json
fingerprint.json
status.json
events.jsonl
metrics.jsonl
summary.json
stdout.log
stderr.log
```

质量 run 还必须保存：

```text
CASE_INPUTS.jsonl
CASE_OUTPUTS.jsonl
CASE_DELTAS.jsonl
PANEL_REGISTRY.jsonl
checkpoint_before_after.json
```

manifest 必须记录 source commit/tree、dirty status、配置 hash、dataset/split/case manifest hash、checkpoint hash、renderer/evaluator hash、seed、GPU/driver/CUDA、开始/结束时间、退出码、failure refs/delta 与 quality-read locks。

## 5. R0 — Preflight 与冻结

**Task**：`WS-V52-R0-PREFLIGHT-01`

在读取任何新算法质量前完成：

1. 复核 attribution manifest 的 `18=9+8+1` 与 `5D+3C`；
2. 复核 StreetGS checkpoint、GT、dynamic mask、camera、renderer 和 U2/B3 sidecar hash；
3. 将每个 case 的相邻 `t-1/t/t+1` 可用性、LiDAR、nuScenes annotation/track 可恢复性写入 capability table；
4. 冻结 Base Validity、M1、M3、M2 指标方向和非回归阈值；
5. 冻结 Confirmation exclusive-create attempt ledger；
6. 确认 RTX 3090 VRAM、disk、cgroup 和 detached launcher。

任何 identity/split/hash 不一致都在 quality read 前拒绝本 run。

## 6. R1 — P0 Base Validity 与因果桥

**Task**：`WS-V52-R1-BASE-VALIDITY-BRIDGE-01`

### 6.1 Base Validity Gate

每个 view 独立判断：

```text
static/global reconstruction sufficiently sane
AND
dynamic-region residual materially worse than static region
```

判定只使用 Discovery 冻结分布确定的规则；不能用 5 张图现调。输出包含 global/static/actor/boundary 指标、actor/static residual ratio、有效 dynamic mask area、cross-base context 和 gate reason。

### 6.2 Pixel → Gaussian exact bridge

对 actor/boundary residual pixel 反查 frozen StreetGS Gaussian contribution，输出：

- Gaussian index 与 alpha contribution；
- U2/B3 actor/background posterior；
- observation count 与可见 view count；
- UNKNOWN/abstain；
- dynamic/static node identity；
- actor track、LiDAR、flow、visibility 当前是否可得。

不得用 raw KNN 把 pixel 标签传播给未贡献 Gaussian；不可见只记 `no_update`。

### 6.3 R1 晋级门

M1 bridge 只有同时满足才通过：

- 5/5 design case 通过 Base Validity；
- exact pixel→Gaussian mapping 至少 `4/5` 定义；
- low-observation 或 UNKNOWN residual enrichment `>=1.5×` 至少 `3/5`；
- 支持来自至少 2 个独立 scene。

若失败，M1 保持 pending；仍继续 M3 temporal bridge，因为 ghost 可能主要是 trajectory/visibility。

## 7. R2 — M1 TrackBayes-GS

**Task**：`WS-V52-R2-M1-TRACKBAYES-01`

### 7.1 问题定义

对 Gaussian `i` 和 actor `a`：

```text
z_i ∈ {background, actor_1, ..., actor_K, unknown}
```

比较两个竞争模型：

- `H_bg`：Gaussian 在 world frame 稳定；
- `H_actor(a)`：Gaussian 在 `T_a(t)^-1` 的 actor canonical frame 稳定。

只有真实可见观测进入 Bayesian update：

```text
v(i,t,c)=0  =>  Δpositive=0 and Δnegative=0
```

### 7.2 Oracle 先于新模型

先跑：

| Arm | 内容 | 目的 |
|---|---|---|
| `M1-O0` | frozen U2/B3 | comparator |
| `M1-O1` | nuScenes GT box/track hard actor tube | 测量 perfect track 的 ownership 上限 |
| `M1-O2` | GT track + canonical Bayesian evidence | 判断 canonical hypothesis 是否有增量 |

若 O1/O2 都不能改善 eligible case，停止训练 tracker；问题更可能在 representation、rendering 或 evaluator。

### 7.3 独立 evidence arms

按成本从低到高逐个运行，禁止先做一锅融合：

| Arm | 新证据 | 对应来源 | 失败时动作 |
|---|---|---|---|
| `M1-A1` | background-only static residual | DeSiRe-GS / DIAL-GS 启发 | residual 与静态纹理错误混淆则 reject |
| `M1-A2` | persistent 2D actor tube | IDSplat-style tracked mask | coverage/persistence 不足则不进 A3 |
| `M1-A3` | LiDAR actor anchoring | IDSplat / UnIRe faithful evidence | 远距点太稀则保留 abstain，不空间补点 |
| `M1-A4` | actor-vs-world canonical residual | 本计划核心 | 两模型无分离则不做 Bayes fusion |
| `M1-A5` | calibrated TrackBayes + UNKNOWN | V5.2 主候选 | 只融合已经单独通过的 evidence |
| `M1-A6` | 1-step identity-motion EM | DIAL reciprocal 启发 | 只允许 0/1 iteration 消融 |

IDSplat 官方实现首先作为 track/tube adapter 参考，不在本轮重训完整 3DGS。若其依赖或权重未就绪，自动记录 blocked，继续 GT oracle 与 static residual；SAM4D、SplatFlow、DynamicVGGT 等重 teacher 不进入本夜主链。

### 7.4 M1 指标

主指标：

- ownership IoU、Boundary F1、FN/FP semantic mass；
- Brier、ECE、NLL、UNKNOWN recall；
- actor PSNR/SSIM/LPIPS；
- low/medium/high observation bucket；
- moving/stationary/pedestrian-cyclist 分组。

非回归：

- static PSNR drop `<=0.30 dB`；
- global LPIPS increase `<=0.01`；
- background FP mass increase `<=0.01`；
- base checkpoint 和 renderer byte-exact 不变。

### 7.5 M1 晋级门

- `>=3/5` design case 的 Boundary F1 `>=+0.01`；
- 至少 2 个 scene 支持；
- scene-balanced IoU delta `>=0`；
- scene-balanced FN mass delta `<=+0.01`；
- Brier/ECE/NLL 不恶化；
- 全部非回归门通过。

只冻结一个 M1 candidate；不得根据 Confirmation 改 arm 或 threshold。

## 8. R3 — M3 Temporal SE(3) 因果桥

**Task**：`WS-V52-R3-M3-TEMPORAL-BRIDGE-01`

M3 当前不是重新发明 trajectory module，而是回答 ghost 的来源。

对 #05/#11/#16/#17 的 `t-1,t,t+1` actor crop 比较：

| Arm | 对齐方式 | 解释 |
|---|---|---|
| `M3-T0` | unwarped | 当前 census proxy |
| `M3-T1` | optical-flow warped | 非刚体/像素 correspondence 是否解释误差 |
| `M3-T2` | frozen actor-pose warped | SE(3) trajectory 是否解释误差 |
| `M3-T3` | V4 frozen SE(3) delta replay | 已确认模块在同 case 的真实增量 |

判读：

```text
pose warp wins  -> 保留 M3 SE(3) hypothesis
flow warp wins  -> visibility / non-rigid motion 更可能
neither wins    -> ownership / appearance / base representation 更可能
```

R3 通过条件：至少 `3/4` design M3 case 有合法 correspondence，warp 后主 residual 降低 `>=10%`，至少覆盖 2 个 scene，并通过背景非回归。

R3 失败不倒写 V4 M3 的历史 confirmation；它只表示当前 8 个 seed case 没有建立 exact overlap。

## 9. R4 — M2 Risk-Aware Execution Policy

**Task**：`WS-V52-R4-M2-SAFETY-01`

M2 不产生 missing evidence，只判断是否执行已经形成的 M1/M3 candidate。

输入特征：

- Base Validity；
- ownership UNKNOWN mass / posterior margin；
- actor track confidence 与 visibility support；
- M1/M3 disagreement；
- geometry metric status；
- candidate 的 predicted risk。

硬 abstain：

- `BASE_FAILURE` 或 `ATTRIBUTION_UNRESOLVED`；
- actor track 低于 frozen confidence；
- geometry claim 需要的 metric 为 undefined；
- M1 与 M3 高置信冲突；
- candidate 越过 frozen spatial/temporal support。

M2 报告完整 denominator：accepted、abstain、blocked、base sentinel、counterfactual risk。若 geometry 可比，要求 full-denominator geometry MAE delta `<=0`；若仍不可比，只能报告 `geometry_safety_undefined`，不能写 M2 success。

## 10. R5 — 因子化融合

**Task**：`WS-V52-R5-FACTORIAL-FUSION-01`

只允许：

```text
F0 = base comparator
F1 = best passed M1
F2 = best passed M3
F3 = F1 + F2                 # 仅当二者独立通过
F4 = F3 + M2 safety policy   # M2 只包装执行策略
```

融合必须优于各被包含模块在自己的主轴，并满足全部非回归；否则回退为最佳单模块。禁止把失败组件藏进 full model。

## 11. R6 — One-shot Confirmation

**Task**：`WS-V52-R6-ONE-SHOT-CONFIRMATION-01`

执行前原子冻结：candidate、config、threshold、source tree、checkpoint、evaluator、3 个 case 顺序和判词。每个 case exclusive-create attempt，失败也算消费。

通过条件：

- 至少 `2/3` case 与预注册改善方向一致；
- 无 catastrophic background regression；
- `threshold_refit=false`、`candidate_changed=false`；
- 3 个 case 全部保留在 denominator，包括 undefined/blocked。

失败则 candidate rejected，不允许用相同 Confirmation 再调。

## 12. 一晚调度

| 时间预算 | Task | GPU | 可并行/独立性 |
|---:|---|---:|---|
| 0:00–0:30 | R0 preflight/freeze | 低 | 必须先完成 |
| 0:30–1:45 | R1 Base Validity + causal bridge | 中 | M1 前置 |
| 1:45–5:45 | R2 M1 oracle/evidence/TrackBayes | 高 | 与 R3 的预处理可交错，GPU 串行 |
| 5:45–8:15 | R3 M3 warp bridge/replay | 高 | M1 失败仍继续 |
| 8:15–9:30 | R4 M2 safety replay | 中 | 允许无 candidate 时只做合同审计 |
| 9:30–11:00 | R5 factorial fusion | 高 | 只跑已通过组件 |
| 11:00–12:00 | R6/R7 confirmation 或 blocked closeout | 中 | 未冻结 candidate 则跳过 R6，直接 closeout |

若实际耗时提前，不能把剩余 GPU 时间用于无预注册超参搜索、大 teacher 下载或 fresh validation；直接完成审计、面板和报告。

## 13. R7 Closeout

**Task**：`WS-V52-R7-CLOSEOUT-01`

必须生成：

```text
FINAL_CASE_METRICS.jsonl
FINAL_CASE_DELTAS.jsonl
FINAL_PANEL_REGISTRY.jsonl
DECISION.json
SUMMARY.json
manifest.json
status.json
```

`FINAL_CASE_DELTAS.jsonl` 必须继续使用同一 `case_id`，记录 comparator/candidate asset hash、每个 primary/non-regression metric 的 before/after/delta/status，以及 causal bridge verdict。以后任何算法改进都通过该表回测，禁止只汇报 aggregate。

closeout 只能给出：

- `M1_PROMOTED / M1_REJECTED / M1_EVIDENCE_INSUFFICIENT`；
- `M3_OVERLAP_CONFIRMED / M3_OVERLAP_NOT_CONFIRMED / M3_EVIDENCE_INSUFFICIENT`；
- `M2_SAFETY_CONFIRMED / M2_ABSTENTION_ONLY / M2_GEOMETRY_UNDEFINED / M2_REJECTED`；
- full system 的 `confirmed / rejected / blocked`。

## 14. 参考方法与迁移边界

- [IDSplat arXiv](https://arxiv.org/abs/2511.19235) / [official code](https://github.com/zenseact/idsplat)：只优先迁移 persistent mask、LiDAR anchoring、actor pose 与 trajectory smoothing；
- [DeSiRe-GS, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Peng_DeSiRe-GS_4D_Street_Gaussians_for_Static-Dynamic_Decomposition_and_Surface_Reconstruction_CVPR_2025_paper.html)：static residual 作为 likelihood，不作为 GT；
- [AD-GS, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_AD-GS_Object-Aware_B-Spline_Gaussian_Splatting_for_Self-Supervised_Autonomous_Driving_ICCV_2025_paper.html)：迁移 bidirectional visibility/no-update 思想，不迁移 KNN；
- [UnIRe official code](https://github.com/YunxuanMao/UnIRe)：4D LiDAR superpoint 只作为后续 actor hypothesis；
- [SplatFlow, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SplatFlow_Self-Supervised_Dynamic_Gaussian_Splatting_in_Neural_Motion_Flow_Field_CVPR_2025_paper.html)：scene flow 只作为 motion likelihood；
- [EMD, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wei_EMD_Explicit_Motion_Modeling_for_High-Quality_Street_Gaussian_Splatting_ICCV_2025_paper.html)：motion embedding 为条件证据，不替代 actor identity。

本夜不把任何外部方法的完整 reconstruction 直接接管 frozen StreetGS，也不因 paper 宣称有效就跳过 faithful adapter 与本地 denominator gate。

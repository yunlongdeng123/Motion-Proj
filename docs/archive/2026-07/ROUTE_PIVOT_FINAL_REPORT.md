# Motion-Proj Route-Pivot Autoresearch 最终报告

> **计划**：`MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md`
> **执行日期**：2026-07-18
> **最终任务**：`RP-D0-08`
> **结论状态**：V5 `done`；Route A/B `rejected`；Route C 审计 `done`；选择 `C1` 迁移可行性路线
> **注意**：C1 尚未下载权重、尚未通过单卡 smoke、尚未训练，不是论文方法晋级。

## 1. Executive decision

V5 完整执行后的结论是：

```text
SVD internal motion post-training support = insufficient
Route A actor representation = rejected
Route B natural-rollout preference support = rejected
Route C migration audit = completed
Final decision = C1, ReSim exp0_no_carla feasibility first
```

被否定的是当前 SVD-XT 上两类具体机制：

1. 从冻结共享表示中用 compact probe/局部 cost 恢复 actor-independent residual；
2. 从自然 seed pool 中挖出足量、非低运动、非画质投机的安全偏好。

这不是“驾驶视频物理偏好不可行”。A1 给出了明确的 driving-specific motion entanglement 证据：ego/background
motion 可读，但 actor residual 比 zero baseline 更差，且 stationary actor 出现更大假运动。B0 则证明当前 SVD
分布中的 motion/quality/identity support 不足。下一步必须先换到显式 ego trajectory、但不喂 future actor boxes 的
预测式 driving backbone，才能重开可辨识性。

## 2. Repo、环境与资源

| 项目 | 最终事实 |
|---|---|
| 远端仓库 | `/root/autodl-tmp/motion_proj` |
| 分支 | `main`，未 push |
| C0 启动 commit | `e0063ab83d141cfc90b944bca2ee78317706dc56`，clean |
| Python / PyTorch | 3.10.20 / 2.4.1+cu121 |
| GPU | 1 × RTX 4090 24 GB |
| 磁盘 | `/root/autodl-tmp` 剩余 42 GB |
| 数据 | nuScenes trainval、CAM_FRONT samples/sweeps、LIDAR、maps 存在 |
| 既有模型资产 | SVD-XT、RAFT、Depth、CoTracker3 完整 |
| 新 backbone | 未下载、未安装、未运行 |
| 最终回归测试 | 247 passed，2 warnings |

V5 全程保持单卡；没有删除或覆盖历史 run，没有自动填人工 verdict，没有自动 push。

## 3. Sibling 路线关闭证据

历史 SVD denoising sibling 路线已在 V5 前关闭，V5 未恢复它：

- 48-case 人工复核中，P1 为 22 tie、0 decisive；
- uncertainty-aware common-support partial order 降低 false-strict，但未审 96 conditions 仅 2 strict；
- 唯一 earlier-fork fallback 只有 1 strict，并出现首帧/temporal-jump failure；
- 继续搜索 fork、rho、CFG 或 candidate 数被明确禁止。

因此 uncertainty-aware partial order 被保留为**测量层**，不能被当成 preference generator。

## 4. R1：时间采样与 fps

正式 run：`route-pivot-r1-temporal-s20260718-v1`。

- 32 个 scene-distinct 真实 clips 的 timestamp delta 中位数为 `0.5000 s`，有效 fps 为 `2.0000 Hz`；
- 8 conditions × 3 fps × 2 seeds 的 48 个生成全部有效；
- 相对 fps7，fps2/4 的 dynamic degree 增加 `24.74%/10.05%`，image velocity 增加
  `77.97%/110.71%`；
- fps2 失败首帧、锐度、闪烁、track survival 与 acceleration safeguard；
- fps4 的 survival ratio `0.859 < 0.90`，acceleration-p95 ratio `1.950 > 1.25`。

结论：真实 2 Hz 与 SVD fps micro-conditioning 不是同一物理量；直接把 fps 改为 2/4 不是安全修复。V5 后续
冻结 `generation.fps=7`，同时继续要求真实 target 使用真实 delta-t。

详见 [`ROUTE_PIVOT_TEMPORAL_AUDIT.md`](ROUTE_PIVOT_TEMPORAL_AUDIT.md)。

## 5. A0：真实 target legality

最终事实源：`route-pivot-a0-real-motion-s20260718-v3`。

- 修复 `min_box_visibility` 未应用与 offscreen-center denominator 两类实现/检查问题；
- 16 个 scenes 中 420/421 actor pairs finite；
- actual_t、actual_t+1、static-if-world-fixed_t+1 共同图内后，392/421 pairs、89 tracks 可局部化；
- moving/stationary pairs 为 181/208；actor residual AUC `0.8600`；
- velocity direction positive fraction `0.9725`；与 ego speed 的 Spearman `0.2226`；
- 157,394 个 confident background LiDAR 点上，ego-flow 与 RAFT 方向通过率 `0.9870`。

机器 gate 通过，证明训练侧 target 有合法、可局部化的支持；12 个 target-legality panels 的人工 verdict 仍为空。
由于 A1 已在更下游拒绝，人工 pending 不阻塞 V5 收口，也不得被自动填充。

详见 [`ROUTE_PIVOT_REAL_MOTION_TARGET_AUDIT.md`](ROUTE_PIVOT_REAL_MOTION_TARGET_AUDIT.md)。

## 6. A1：冻结表示 probe

正式事实源：`route-pivot-a1-feature-scan-s20260718-v2`。v1 是 fp32 preprocess 与 bf16 VAE mismatch 的
工程失败，原样保留。

v2 使用 24/8 scene-disjoint clips、7 layers × 3 sigma，共 21 probes：

- ego-flow 对最佳常数 baseline 改善 `17.86%–25.01%`；
- actor A-RES 对 zero-residual baseline 全部为负，`-213.80%–-120.35%`；
- stationary prediction / moving target ratio 为 `3.292–5.062`；
- 0 primary candidates，0 个 layer 在两个 sigma 稳定。

这不是“没有 motion feature”，而是共享表示中的 camera/ego motion 可读，actor-independent residual 被
appearance/scene/ego 共变淹没。A1-CONFIRM 与 A2 按依赖不运行；不事后调 ridge/head/阈值追 gate。

详见 [`ROUTE_PIVOT_MOTION_FEATURE_AUDIT.md`](ROUTE_PIVOT_MOTION_FEATURE_AUDIT.md)。

## 7. A2：未运行原因

`RP-A2-06 = rejected / not run by dependency`。

A2 的前置条件是 A1-CONFIRM 有合法候选。A1-SCAN 已经是 0 candidate，因此运行 LoRA capacity test 只会把
训练容量引入一个未通过可辨识性的 target，违反 V5 “先存在性/可观测性、后训练”的顺序。

## 8. B0：Natural rollout ceiling

正式 run：`route-pivot-b0-natural-rollout-s20260718-v1`；只读稳健性 run：
`route-pivot-b0-sensitivity-s20260718-v1`。

- 16 conditions 先各 N=4；`0/16` diverse，按计划扩到 N=8，总计 128 videos；
- 112 个 selection candidates 仅 7 个通过全部 eligibility；最终 `1/16` diverse；
- 仅 6 conditions 可形成 P-UNC 对照；对 random/Base 的 CoTracker win-credit 都为 `41.67%`；
- selected low-motion/catastrophic 为 1/3，positive-improvement conditions 为 0；
- 18 dB absolute first-frame floor 过严，但移除它、移除全部 first-frame、再移除 motion checks 后仍只有
  `4/16`、`6/16`、`10/16` diverse；只有剥掉整套 anti-collapse 才到 16。

结论：失败不是 ranking 小误差，而是冻结 SVD 分布缺少足量安全 support。B0 不生成人审包，不扩 N，不进
AWR/SFT。

详见 [`ROUTE_PIVOT_NATURAL_ROLLOUT_AUDIT.md`](ROUTE_PIVOT_NATURAL_ROLLOUT_AUDIT.md)。

## 9. Human review 状态

| Review | 数量 | 状态 | 是否阻塞最终决策 |
|---|---:|---|---|
| R1 fps diagnostic | 32 pairs | verdict 为空 | 否；只作 evaluator nuisance 补充诊断 |
| A0 target legality | 12 panels | verdict 为空 | 不阻塞 C0/D0；若未来复用 A0 target 训练则必须完成 |
| B0 preference | 0 | machine gate 失败，未生成 | 否 |

Codex 未代填任何人工判断。

## 10. Route A/B/C 评分

下表是 hard-gate rubric，不是可相加的论文分数：

| 路线 | 合法信号 | 安全 candidate/support | ego–actor 可辨识性 | 当前资源可执行 | 最终 |
|---|---|---|---|---|---|
| A：real representation | A0 machine pass | A1 0 candidate | **fail**：actor 低于 zero 且 stationary 假运动 | probe 可执行，训练依赖未满足 | rejected |
| B：natural BoN | evaluator 隔离成立 | **fail**：N8 仅 1/16 diverse | 不直接解决 | 128 videos 完成 | rejected |
| C：controlled backbone | ReSim trajectory schema 合法 | 未测，必须新 gate | **promising**：ego 显式、actor 不喂 future boxes | 权重公开；单 24 GB 未验证、磁盘不足 | **C1 feasibility** |

任何一个 hard fail 都不能靠其他列补偿。C1 的 “promising” 只意味着问题可重新定义，不是结果已成立。

## 11. Route C 审计结论

C0 比较了 ReSim、VISTA、OpenDWM、MagicDrive-V2、DriveDreamer 与 VLA-World：

- **ReSim exp0_no_carla**：2B DiT、8 点未来轨迹、历史帧预测、无 future actor boxes、官方 checkpoint 与
  nuScenes JSON；选择为 primary；
- **VISTA**：动作条件和自由 actor 合法，但官方最小采样显存 32 GB，且仍为 SVD family；作为 fallback；
- **OpenDWM / MagicDrive-V2**：公开权重和 nuScenes 工程较完整，但 future boxes/maps 规定周车，不能作为
  自由 actor-physics 主验证；只作 layout/geometry baseline；
- **DriveDreamer**：无可定位官方 checkpoint；
- **VLA-World**：科学问题贴近，但官方页没有代码或权重。

完整证据见 [`BACKBONE_MIGRATION_AUDIT.md`](BACKBONE_MIGRATION_AUDIT.md)。

## 12. 最终主线

最终选择：

```text
Main = C1 action-conditioned driving backbone feasibility
Backbone = ReSim exp0_no_carla
Research target = action-disentangled, uncertainty-aware localized physics preference
Training status = not started
```

理由不是 ReSim 的论文指标，而是其条件结构最适合本项目的新因果切分：ego trajectory 显式给定，future actor
motion 仍由模型负责。这样才能把“相机/自车运动”从 actor residual 的偏好证据中扣除，而不把 future actor
答案直接喂给生成器。

## 13. Fallback

```text
Fallback 1 = VISTA action-conditioned inference baseline
Fallback 2 = C2 stop current generative-model direction
```

OpenDWM/MagicDrive-V2 只用于 layout adherence 或高分辨率 geometry-control 对照，不晋级为主骨干。若 ReSim
在单卡最小 smoke 或 preference-support gate 失败，不迁回 SVD，不把 future boxes 打开来制造成功，应进入 C2。

## 14. 明确停止做什么

- 不再调整 SVD fork、rho、CFG、fps、candidate N 或旧 P-UNC scorer；
- 不使用旧 53 pairs、2 个 UPO strict 或唯一 fallback strict 训练；
- 不启动 A1-CONFIRM、A2、AWR、DPO、PPO/GRPO；
- 不用 future ego/actor GT 评价自由生成结果；
- 不把 OpenDWM/MagicDrive future-box adherence 称为 actor physics；
- 不把 ReSim `exp0_no_carla` 称为含 CARLA 非专家数据的完整 ReSim；
- 不在 42 GB 剩余空间下下载约 34.4 GB 的 ReSim 最小资产；
- 不在单卡 Base 可行性和 preference support 通过前配置双卡训练。

## 15. 下一步最多三个实验

这些是下一阶段候选，不属于已完成的 V5 授权；必须进入新的预注册计划。

### C1-BOOT：单卡 Base/action feasibility

```text
8 scene-disjoint nuScenes contexts
9 latent frames, 1 sample/case, fp16, serial VAE decode
GT feasible trajectory vs action-shuffled/nearby trajectory
no future actor boxes
```

先测峰值显存、完整输出、历史帧保持、trajectory compliance、actor survival 与 action sensitivity。开始前把可用
磁盘提高到 70–80 GB。该实验仍只需要一张卡。

### C1-PREF：action/rollout sibling support

分开两类 sibling：

- action sibling：同历史、共同噪声、邻近可行 ego trajectories，只测因果 action response；
- rollout sibling：同历史与同 trajectory、独立自然 seeds，才进入 preference support。

用 common localized support、ego-compensated actor residual、identity/survival、quality 与 action-conditioned
motion floor 构建 strict/tie/incomparable；冻结最小合法 yield 与人工盲审 gate。若再次没有 support，直接 C2。

### C1-CAP：dense safeguarded partial-order capacity

只在 C1-BOOT 与 C1-PREF 均通过后，用 2×4090 做 256×448、9 latent frames、micro-batch 1 的短 LoRA
capacity test。训练不是 vanilla DPO：局部 common-support confidence 权重 + support 外/history/real-denoising
anchor + low-motion safeguard。到这一步才需要停机配置第二张卡。

## 16. Reviewer 2 攻击点

1. **“只是 ReSim + UPO/DenseDPO。”** 必须把 action sibling、rollout sibling、localized partial order、
   abstention 与 anchors 做正交消融；不能只换 loss 名称。
2. **“trajectory/box 是 future GT 泄漏。”** ego trajectory 是明确的控制输入；future actor boxes/tracks 永远禁止
   进入自由 actor rollout 条件和 evaluator truth。
3. **“运动改善其实是相机运动或低运动偏置。”** 固定/分层 ego trajectory，做 stationary/moving、action shuffle、
   ego-compensated actor residual、motion floor 与 track survival。
4. **“同一 tracker 训练又评价。”** RAFT/P-UNC 只训练侧；CoTracker3、第二独立 estimator 和人审做正式评价。
5. **“dense mask 是伪局部化。”** 必须测 gradient-in/support-out、mask shuffle/null、frame-0/history drift 和
   scene-disjoint transfer。
6. **“公开 checkpoint 不具备论文非专家能力。”** 全程标记 `exp0_no_carla`，只主张专家/邻近可行轨迹；
   full-CARLA 结果不能借用。
7. **“abstention 后没有数据。”** 同时报告 false-strict 与 strict yield；任何低 yield 都 fail closed，不扩 N 追结果。
8. **“只在小样本有效。”** 先以 screening 明确命名，再做 scene-level paired bootstrap；小于 32 scenes 不作主结论。

## 17. 文件、commit、run 与测试

### 文档

- [`MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md`](MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md)
- [`ROUTE_PIVOT_LITERATURE_MATRIX.md`](ROUTE_PIVOT_LITERATURE_MATRIX.md)
- [`ROUTE_PIVOT_TEMPORAL_AUDIT.md`](ROUTE_PIVOT_TEMPORAL_AUDIT.md)
- [`ROUTE_PIVOT_REAL_MOTION_TARGET_AUDIT.md`](ROUTE_PIVOT_REAL_MOTION_TARGET_AUDIT.md)
- [`ROUTE_PIVOT_MOTION_FEATURE_AUDIT.md`](ROUTE_PIVOT_MOTION_FEATURE_AUDIT.md)
- [`ROUTE_PIVOT_NATURAL_ROLLOUT_AUDIT.md`](ROUTE_PIVOT_NATURAL_ROLLOUT_AUDIT.md)
- [`BACKBONE_MIGRATION_AUDIT.md`](BACKBONE_MIGRATION_AUDIT.md)

### 主要 commits

| 阶段 | commit |
|---|---|
| R1 | `f4b4cd5` |
| A0 final | `45cb279` |
| A1 implementation/fix | `f6b97f0` / `b27fb5a` |
| B0 implementation/sensitivity | `06dc211` / `d9ac65d` |
| B0 docs | `e0063ab` |
| C0 migration audit | `c05735b` |

### 正式 runs

```text
/root/autodl-tmp/runs/route-pivot-r1-temporal-s20260718-v1/
/root/autodl-tmp/runs/route-pivot-a0-real-motion-s20260718-v3/
/root/autodl-tmp/runs/route-pivot-a1-feature-scan-s20260718-v2/
/root/autodl-tmp/runs/route-pivot-b0-natural-rollout-s20260718-v1/
/root/autodl-tmp/runs/route-pivot-b0-sensitivity-s20260718-v1/
/root/autodl-tmp/runs/route-pivot-c0-backbone-audit-s20260718-v1/
```

### 测试

```text
PYTHONPATH=. pytest -q
247 passed, 2 warnings
```

## Final status

`RP-D0-08 = done`。V5 没有可继续自动执行的训练任务。下一阶段先解决磁盘安全余量并预注册 C1-BOOT；
单卡 smoke 通过前不需要第二张 GPU，只有 C1-CAP 才需要停机配置双卡。

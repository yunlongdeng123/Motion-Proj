# WorldSim V3.3 S1：Object-aware Gaussian Instance Field

- Task：`WS-V33-S1-OBJECT-AWARE-GS-01`
- 状态：`done`
- 日期：2026-08-11
- 开发选择：`O1_dual_opacity`
- canonical formal：`20260810T183154Z__s1-instance-field-formal-s0-r9`
- 下一授权：仅 `WS-V33-S2-ROADPATCH-INPAINT-01`

## 1. 结论

S1 在 V3.1 D2 StreetGS checkpoint 完全只读的前提下，实现了一个独立、可微、可回滚的
`instance_field.npz`。它复用原 Gaussian 的 means/scales/quats、相机与深度顺序，只把独立
`instance_opacity_logit` 送入 gsplat alpha compositing；base RGB opacity、SH 与几何均不进入 optimizer。

scene-0230 的固定 heldout 上，冻结 O1 相对 V3.2 heuristic O0：

| 指标 | O0 heuristic | O1 dual opacity | 相对变化 |
|---|---:|---:|---:|
| boundary F1 ↑ | 0.068960 | 0.336158 | +387.47% |
| mask IoU ↑ | 0.063253 | 0.330727 | +422.87% |
| normalized boundary distance ↓ | 0.144958 | 0.105280 | -27.37% |
| false-positive semantic mass ↓ | 0.900308 | 0.623276 | -30.77% |
| false-negative semantic mass ↓ | 0.061278 | 0.109356 | +78.46%（退化） |
| identity presence ↑ | 0.972973 | 0.972973 | 持平 |

因此，S1 构成可验证的对象边界与 false-positive 抑制突破，但不是所有指标的全面支配。FN 增加必须进入
S2 删除残留门，不能被平均 IoU 或视觉样例掩盖。

## 2. 冻结输入

| 输入 | 冻结事实 |
|---|---|
| D2 checkpoint | SHA-256 `1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c` |
| V3.2 train mask manifest | SHA-256 `efc7d82145dfe0cf3c1bfccf230b0756f152ed5b306610b2634ddbcdbabab98b` |
| high-support posterior | SHA-256 `0119e15d979c1c69cdaea23d9c4716ae66f323972cd4abf3027fb43593ce22cb` |
| boundary-support posterior | SHA-256 `25b1eaecb2bb53af852ec2c5fae1abf53636359f9714e2de6e9a330187f22050` |
| actor registry | SHA-256 `ed57764e094e83392bad8f052441da54bbe6c7b6754139908c3ddb83980e0c68` |
| S1 final config | SHA-256 `9afa48aa1ff5ebbb290da564e901f25c48ac1f6ee16f97379f936457acdc3150` |

身份合同继续绑定：

| role | dataset ID | instance token | rigid index |
|---|---:|---|---:|
| high-support | 13 | `af663976db5e412e83db033d309c5c29` | 5 |
| boundary-support | 41 | `18c7f0c5fa6b49449f71c9dbae5c31d4` | 21 |

任何 ID/token/rigid-index 漂移、未知 instance ID 或近似同分背景冲突都会 fail closed。

## 3. 表示与算法

全局 sidecar 覆盖 D2 的 `1,309,868` 个 Gaussian，关键字段为：

```text
gaussian_id
base_model / base_index
hard_instance_id
instance_opacity_logit / instance_opacity
source_semantic_score
num_positive_views / num_negative_views / visibility_mass
trainable
provenance
actor_instance_ids / actor_tokens
```

O1 共分配并训练 `65,989` 个 Gaussian，其中 `8,253` 个来自两个 RigidNodes core，`57,736` 个来自
V3.2 semantic-positive background。11 个跨 actor 证据 margin 小于 `0.05` 的背景点保持未分配。O3 额外纳入
`8,761` 个 ambiguous boundary 候选，但 development 不优于 O1，未进入 formal。

instance mask 使用独立 alpha：

```text
alpha*_i = sigmoid(instance_opacity_logit_i) × projected_gaussian_i
```

loss 固定为 balanced BCE + Dice + sparse + V3.2 prior。随机 object/view 采样每 step 只渲染目标 actor 的
assigned Gaussian。共享全局 logit 使同一 Gaussian 的跨帧身份参数稳定性按构造为 `1.0`。

## 4. 数据隔离

- V3.2 canonical `334 accepted / 64 rejected` masks 只用于 optimization/development；
- development frames 固定为 `5,25,...,185`，与 optimization frame 不重叠；
- formal heldout 固定为 `10,20,...,190`；
- heldout-target r4 生成 31 个可见 singleton block、37 个 accepted masks；
- prompt/mask manifests 都写入 `optimization_forbidden=true`；
- formal runner 在 300-step optimization 完成后才加载 heldout manifest；
- finalizer 同时检查 frame 集、manifest SHA、mask SHA 与 source image SHA。

## 5. SAM2.1 exact fallback

P0 已裁决 SAM3.1 checkpoint `weights_blocked`。S1 没有绕过权限，而是恢复 V3.2 exact fallback：

| 项 | 值 |
|---|---|
| source commit | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| checkpoint bytes | 898,083,611 |
| checkpoint SHA-256 | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` |
| Python | 3.10.20 |
| torch / torchvision | `2.5.1+cu124 / 0.20.1+cu124` |
| NumPy | 1.26.4 |
| conda explicit SHA | `c9294494e5950608814721382974722fb99f7f0d720d747842a18677aa1d0713` |
| pip freeze SHA | `aded7fb5fd6a3cfc5ecce203b79ed09d800f9bce5e16356a9ba3786eacd25d69` |

该环境位于 `/root/autodl-tmp/envs/worldsim-v33-sam2`，没有向 DriveStudio 环境安装包。SAM2 可选 CUDA
post-processing extension 未构建，与 V3.2 的 `editable_no_optional_cuda_extension` 合同一致。

## 6. Run 生命周期

| run | 终态 | 用途/处置 |
|---|---|---|
| `20260810T174855Z__s1-instance-field-diagnostic-s0-r0` | done noncanonical | 1 step/arm 全链工程探针，不作算法选择 |
| `20260810T175040Z__s1-instance-field-smoke-s0-r1` | done | 100 step/arm development；冻结 O1 |
| `20260810T175400Z__s1-heldout-targets-s0-r2` | failed | 旧 SAM Python 被外部清理，exit 127 |
| `20260810T180149Z__s1-heldout-targets-s0-r3` | failed | singleton logits rank 兼容错误，未发布正式 mask |
| `20260810T180231Z__s1-heldout-targets-s0-r4` | done | canonical heldout targets |
| `20260810T180328Z__s1-instance-field-formal-s0-r5` | done noncanonical | 输入正确，但 runner 入口未重核 ID/token/rigid 三元 identity |
| `20260810T181220Z__s1-instance-field-formal-s0-r6` | done noncanonical | 补齐 identity 重核；默认 NPZ ZIP timestamp 使相同 O0 数组 SHA 漂移 |
| `20260810T181523Z__s1-instance-field-formal-s0-r7` | done noncanonical | 确定性 writer 生效；提交前 EOF 清理使 2 个 source snapshot 不再 exact |
| `20260810T182853Z__s1-instance-field-formal-s0-r8` | failed | 前台 SSH 超时关闭 stdout，外层 SIGPIPE/141；不得续写 done |
| `20260810T183154Z__s1-instance-field-formal-s0-r9` | done | canonical；后台托管，identity/确定性 writer/source snapshot 全 exact |

canonical r9：

- summary SHA-256：`4ab311a64437202ecdd5fa915c4bd528543cdc6040a12df54a5183a39bdf8c4a`；
- run manifest SHA-256：`e1b858fd505c65e41dc3272137e355ad36b4e45bc18b350c3140bcbde1ef584e`；
- status SHA-256：`9394d15e935285955812e9a4502ffa0f4029ca4c3be9c535b249ecc93e7303b9`；
- acceptance SHA-256：`b53576db81f49b44b9c03a7a0cd8e1ff8e9d809d2366cc791c21c73a33ef4305`；
- selected field：`5,882,296 bytes`，SHA-256
  `23b2403ccb47e2e2c6b5fa3d22a9a6d93815d9f9bcbc6d11b66f035831adc8d7`；
- optimization：300 steps，train wall `15.115s`；
- peak CUDA allocated/reserved：`8,001,482,240 / 8,084,520,960 bytes`；
- D2 checkpoint before/after SHA exact；cgroup `oom/oom_kill=0/0`。

r9 的 NPZ writer 固定 entry 排序、1980 ZIP timestamp、权限与压缩参数；对 r9 O0/O1 field 重新写入后 bytes/SHA
均 exact。r7→r9 的 O0 数组全部 exact；O1 因 CUDA 优化有最大 `0.001357` logit、`8.918e-05` opacity
浮点漂移，但两个 arm 的 heldout aggregate 指标逐字段 exact。因此这里声明的是可复现容器，不宣称训练位级确定性。
正式快照中的 config、3 个模块、runner、finalizer 与 3 个测试共 9 个文件也和待提交源码逐一 SHA exact。

## 7. 工程验收

- 纯 Torch alpha compositor 的 closed-form 与梯度测试；
- instance field schema、pickle-free round-trip、未知 identity 拒绝；
- ambiguous reassignment margin 与 O1/O3 候选边界；
- balanced mask loss 的有限梯度；
- boundary F1/NBD 指标 correctness；
- sidecar 写入不修改 base checkpoint；
- deterministic NPZ double-write byte exact；
- 真实 DriveStudio/gsplat 探针确认 base geometry `requires_grad=false`；
- P0+S1 与 V3.2 定向回归：`51 passed in 3.02s`；
- Python compile、bash syntax 与 `git diff --check` 通过；
- formal run 内冻结 config/module/runner/finalizer/test source snapshots 与逐文件 SHA manifest。

## 8. 结论边界与下一步

当前证据只覆盖 scene-0230、两个固定 actor、SAM2 pseudo targets 和单卡 RTX 3090。它不证明跨场景泛化、
真实像素 GT segmentation、闭环驾驶安全或 SAM3.1 增益。O1 的 FN 增加与一个 heldout view 的 identity absence
必须作为 S2 删除残留风险继续度量。

S1 已关闭。下一步只允许 `WS-V33-S2-ROADPATCH-INPAINT-01`：先实现 immutable native Background donor 的
RoadPatch-Lite，再做 Inpaint360GS 24 GiB 条件 preflight；不得提前启动 S3–S5 或 R0。

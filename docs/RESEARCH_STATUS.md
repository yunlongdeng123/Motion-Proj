# Motion-Proj 当前研究状态

> **文档职责**：唯一当前状态与执行授权入口。
>
> **最后更新**：2026-07-18
> **研究决策基线**：`9c59c7c51ce2b4d725bb83b83c813e00eebfb5fa`
> **当前状态**：`rejected`（现有 SVD common-prefix sibling 路线）
> **当前允许执行的研究任务**：无
> **硬件需求**：无 GPU 任务；不需要人工评审；不需要配置双卡

本文只写当前决策和已经关闭的里程碑。正式数值以 [`EXPERIMENTS.md`](EXPERIMENTS.md) 与对应
run 为准；为什么不能重复旧尝试见 [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)。

## 1. 当前结论

Motion-Proj 已完成两轮主要研究路线的可证伪诊断：

1. **Explicit dynamics projection / endpoint distillation**：当前 synthetic target、RGB/VAE
   counterfactual 和 shared temporal LoRA 组合未通过动力学、target legality 与 locality 门禁；
2. **SVD sibling physics preference**：旧 P-UNC forced-binary 标签器未通过人工可信性复核；
   common-support selective partial order 虽消除了校准集 false-strict，但候选 strict yield 过低，
   唯一 earlier-fork fallback 又触发首帧/质量路线门禁。

因此当前不存在可以进入 DPO、AWR、SFT、LoRA screening、正式训练或双卡扩量的 preference 数据。
也没有任何新 checkpoint 或 rollout-quality improvement 可以用于 CVPR 2027 主张。

拒绝范围严格限定为已测试的机制与候选构造。它不证明以下更广命题：

- 驾驶视频物理偏好无法定义；
- partial order、localized alignment 或 diffusion preference training 普遍无效；
- 显式 action/layout 条件的可控驾驶 world model 无法使用物理监督；
- 新的、可辨识的 sibling/counterfactual generator 不可能产生合法 preference。

## 2. 里程碑

| ID | 状态 | 已完成事实 | 当前决策 |
|---|---|---|---|
| `P2-V1-TUNE-01` | rejected | 16 个 100-step 与 4 个 300-step synthetic projection trial | 不继续旧 cache、Optuna、t10-800 或同配方调参 |
| `P2-V2-COND-00` | done / branch rejected | future-GT ego mismatch 已确认；self-estimated static V1 未过人工门槛 | 未条件化 SVD 禁用 future-GT static target；static branch 停止 |
| `P2-V2-REPLAY-05` | done | object-only generated-track replay 的 schema 与人工合理性通过 | 只保留基础设施，不外推训练收益 |
| `P2-V2-PILOT-03` | blocked | C/D/E capacity 与 single-pair locality 诊断完成 | shared temporal LoRA endpoint 不进入 rollout 或长训 |
| `F0/F1/P1` | rejected | endpoint preserve、raw-feature probe、RGB/VAE target legality 均完成门禁 | 不绕过 target legality 启动 feature head 或生成器训练 |
| `PA0-REVIEW-00` | done | P-UNC 与 E0 既有人工 review 聚合完成 | 仅完成基础设施可信度，不产生偏好标签 |
| `PA1-BRANCH-02` | done | 14-frame common-prefix siblings 通过 same-scene 结构盲审 | 只证明结构合法，不证明 physics winner 可辨 |
| `PA2-PAIR-03` | rejected | 120 conditions、53 machine pairs、48-case 人工复核完成 | 旧 P-UNC forced-binary recipe 禁止训练 |
| `PA2-UPO-03B` | done / yield blocked | common-support oracle 的 tie holdout、shortcut、cycle 与 bootstrap 门禁通过 | `2/96` strict 不足以进入 prospective review 或训练 |
| `PA2-CAND-03D` | rejected | 唯一 8-condition earlier-fork fallback 完成 | first-frame/quality gate 失败；不筛唯一 strict，不再搜索 fork/rho |
| `PA3`–`PA8` | rejected / not run | 上游没有合法且足量的 preference 数据 | 未执行 kernel、screening、双卡、正式训练、评估或论文主张 |

## 3. 当前执行边界

未经用户批准的新计划明确解锁，禁止：

- 使用现有 53 个 P-UNC pairs、旧 local labels、UPO 的 2 个 strict 或 fallback 的唯一 strict 训练；
- 继续搜索 common-prefix fork fraction、rho、candidate 数量或事后降低 strict/quality/coverage 阈值；
- 启动 DPO、AWR、SFT、LoRA screening、长训练、正式 rollout 对比或双卡配置；
- 自动填写、推断或替代任何人工 verdict；
- 把 machine pass、same-scene pass、低能量或单-pair overfit 写成生成质量改善；
- 从 `docs/archive/` 的旧“下一步”恢复任务。

允许的无新研究授权操作只有：

- 只读复核既有代码、文档与正式 run；
- 修复不改变研究语义的基础设施问题，并运行测试；
- 整理证据、负结论、复现说明或论文 related-work 材料；
- 在用户明确要求后起草一个新的、独立预注册的研究计划。

## 4. 可复用资产

下列资产没有因路线拒绝而失效，但复用时必须遵守各自证据边界：

| 资产 | 已验证范围 | 不得解释为 |
|---|---|---|
| official SVD generation parity | matched inputs 下与 Diffusers pipeline exact | preference 或训练收益 |
| scene-level split 与 provenance | scene/clip 无泄漏、fingerprint 可追溯 | 当前数据足以训练 |
| generated point tracks / P-UNC | point-space support、visibility 与部分运动不变量 | 合法 RGB target 或可靠 winner |
| CoTracker3 evaluator | 当前协议内 rerun 与扰动排序稳定 | 绝对物理标定 |
| common-prefix sibling RGB | 同一场景的不同 future 候选 | 人工可辨的物理偏好 |
| common-support UPO oracle | 旧 tie holdout 上低 false-strict、shortcut reaudit 通过 | 足量 preference yield |
| manifest / fingerprint / atomic runtime | 正式 run 可追溯与 fail-closed | 方法结论本身成立 |

## 5. 重新开始研究的最低要求

下一条路线必须是新的候选可辨识性或条件化假设，而不是当前路线的隐式补丁。新计划至少要：

1. 引用 [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md) 中相关条目，并解释规避机制；
2. 在训练前证明候选是同条件、可观察、首帧一致、画质合法且对人类具有足够 strict yield；
3. 对 low-motion、time-slow、track dropout、camera nuisance 和画质退化做 fail-closed attack；
4. 将 tie 与 incomparable 作为正式数据语义，而不是强制二元 winner；
5. 先做 localized correction/locality capacity gate，再讨论单卡 screening、双卡扩量或正式训练；
6. 预注册停止线、人工评审协议、独立 evaluator 和 driving-specific motion-entanglement 证明。

满足这些要求只允许创建新计划，不自动解锁实验。

## 6. 事实源优先级

发生冲突时按以下顺序处理：

1. 正式 run 中不可变的 `manifest.json`、`resolved.yaml`、指标、人工 verdict 与终止标记；
2. [`EXPERIMENTS.md`](EXPERIMENTS.md) 的实验登记；
3. 本文件的当前状态与授权；
4. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md) 的跨实验解释与重开条件；
5. `docs/archive/` 中的历史计划、报告和提示词。

历史文档保留当时语境，可能包含已经执行完毕的“当前任务”。它们不得覆盖本文件。

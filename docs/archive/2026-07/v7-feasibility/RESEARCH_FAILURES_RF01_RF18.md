# Motion-Proj 研究负结论与防重复账本

> **目的**：记录从 V1 开始已经踩过的 research 坑、准确证据边界和重新开启条件，防止未来通过
> 改名、扩大训练、放宽阈值或更换 loss 重复同一个失败问题。
>
> **最后更新**：2026-07-20
> **覆盖范围**：Motion-Proj V1/V2、Autoresearch C0/P0/P1/E0/F0/F1、Physics-DPO PA0–PA2、
> common-support UPO、earlier-fork fallback、Route Pivot R1/A0/A1/B0/C0、ReSim C1 V6（含 C1B-02 H1）与证据驻留状态
> **事实源**：[`EXPERIMENTS.md`](EXPERIMENTS_V1_V7_SNAPSHOT.md) 和各正式 run 目录
> **当前状态**：见 [`RESEARCH_STATUS.md`](RESEARCH_STATUS_LEGACY_SNAPSHOT.md)

## 1. 使用规则

本账本只记录会改变研究命题、证据可信度或论文结论的坑。以下内容不属于 research 负结论：

- CPU `bfloat16` 转 NumPy、scheduler scalar hash 等 trace 实现错误；
- deterministic CUDA median、CPU/CUDA device mismatch；
- schema coverage checker、nested `machine_pass` 读取或数值容差错误；
- 下载、网络、路径、tmux、OOM 以外的普通运行故障。

这类错误应在代码、测试、commit 与失败 run 中保留，但不能拿来证明研究假设失败。相反，一个工程 run
成功结束也不能证明方法成立。

本文件使用以下状态：

- `rejected`：当前已测试命题或配方已有足够证据停止；
- `blocked`：现有实现/证据不能继续，但更窄的新机制仍可能重开；
- `limitation`：基础设施可用，但不能支持常见的更强解释；
- `open_risk`：已有迹象或 reviewer-level 风险，尚未形成因果负结论。

任何新计划复用旧组件时，都必须引用对应 `RF-*` ID，并满足“允许重开”的条件。只更换方法名称、
增加训练步数、降低阈值或扩大同分布样本，不算新假设。

## 2. 一页索引

| ID | 状态 | research 坑 | 禁止的快捷修补 |
|---|---|---|---|
| `RF-01` | rejected | 小 synthetic cache 上视觉 proxy 改善、驾驶动力学反向恶化 | 继续 Optuna、t10-800、增加 exposure 或 `lambda_proj` |
| `RF-02` | rejected | 用模型未接收的 future ego 定义自由生成 SVD 的“物理误差” | 将数据集拥有的 future state 当成模型条件 |
| `RF-03` | blocked | self-estimated static correction 数值更小但会传播前景与伪影 | 放宽人工门槛、扩大背景 mask |
| `RF-04` | rejected | generic smoothing 可通过少动来降低高阶运动能量 | 只看 acceleration/jerk，不守 motion exposure 与 visibility |
| `RF-05` | rejected | point-space correction 合格不等于 RGB/VAE counterfactual 合法 | 放大 mask、加 loss、直接训练 renderer 产物 |
| `RF-06` | rejected | shared temporal LoRA 的 correction 与 locality 强耦合 | 用单-pair overfit、更多 step/rank/preserve sweep 外推方法可行 |
| `RF-07` | blocked | 旧 correction 在 frozen SVD raw feature grid 上大多不可辨 | 直接接 feature head，把插值噪声当 relation signal |
| `RF-08` | limitation | evaluator 重跑稳定不等于绝对物理量有效或 target 合法 | 用同源 scorer、自身过滤相关或 image-plane jerk 作物理真值 |
| `RF-09` | rejected | same-scene sibling 结构合法不等于人类能辨别 physics winner | 从结构盲审通过直接跳到 preference training |
| `RF-10` | rejected | candidate-specific query/support + forced binary 产生 false-strict | 继续使用旧 53 pairs、扩大同配方 review |
| `RF-11` | blocked | uncertainty-aware partial order 可安全 abstain，但旧候选池几乎无 strict yield | 降 ROPE、降置信度或把 incomparable 改成 tie/winner |
| `RF-12` | rejected | earlier fork 增强差异同时引入首帧/时间跳变与质量纠缠 | 继续搜索 fork/rho、事后筛唯一 strict |
| `RF-13` | rejected | 直接把 SVD fps micro-conditioning 匹配到真实 2 Hz 会增大运动，也会破坏质量与时序 safeguard | 把 fps 数值一致当物理校准，或按 motion amount 单指标选 fps |
| `RF-14` | rejected | 冻结 SVD 表示中的 ego signal 不等于可安全分离的 actor residual | 调宽 probe、忽略 zero/stationary gate 或把 ego-only 包装成 actor alignment |
| `RF-15` | rejected | natural seed diversity 不等于足量安全 preference support | 扩 N、删 anti-collapse、挑唯一 diverse condition 或直接 AWR/SFT |
| `RF-16` | limitation | layout controllability 不等于 action-disentangled actor physics | 用 future actor boxes 或公开 checkpoint 存在跳过 Base/action/support gate |
| `RF-17` | blocked→reopened | 四类 ridge 在真实视频上 class/turn 不可辨识；kinematic-lateral v3 同门槛已过 | 降 BA/turn 阈值、用位移-only 冒充 action screen、偷看生成 future 调特征；禁止退回 v1 ridge |
| `RF-18` | rejected | 公开 ReSim `exp0_no_carla` 在 10-context E-vs-F screen 上 action response 不足（E 仅 3/8 优） | 降 7/8 门槛、用像素差冒充 action、扩 seed/换 scene、跳过人审进 C1P |

## 3. 负结论证据驻留状态（2026-07-19）

`STORAGE-RETENTION-20260719` 只改变非 Git 二进制载荷的本机驻留状态，不改写实验事实。完整逐路径清单、
SVD 固定 revision、清理前哈希与回收结果见 [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION_20260719.md)。没有为待删载荷
制作数据备份；恢复依赖固定下载源或以新 run ID 重算，不能覆盖原 run。

| RF | 清理的可替代载荷 | 继续驻留的结论证据 | 对结论与重开的影响 |
|---|---|---|---|
| `RF-01` | V1 tune 与历史训练 `ckpts/` | manifest、resolved config、Optuna/metrics、summary、文档表格和终止事实 | proxy inversion、重复 exposure 结论不变；重开仍必须换合法监督分布与独立 endpoint |
| `RF-06` | 历史训练 `ckpts/`、F0 五组 rejected variants/adapters 与 noise bank | F0 manifest、resolved YAML、metrics、Pareto CSV/PNG、summary、日志和 `COMPLETE` | correction/locality entanglement 不变；重开仍需结构上独立的局部参数子空间与 held-out gate |
| `RF-09` | PA1 v2–v5 candidate 与 independent latent 中间目录 | PA1 manifest、candidate manifest/diagnostics、pairwise/track JSONL、8 条结构 review、panel 与 summary | same-scene 不推出 physics winner；人工可辨识性门禁不变 |
| `RF-10` | PA2 smoke/formal/extension candidate、constructor baseline 与 independent latent 中间目录 | 48 条正式人工 verdict、完整 review package、候选索引/诊断、preferences/segments 与聚合 summary | forced-binary false-strict 结论不变；旧 53 pairs 仍禁止训练 |
| `RF-11` | 不删除 UPO v1/v2 的可信度载荷；只清理其上游可重算候选中间目录 | UPO v1/v2 的 query、paired tracks、common support、bootstrap、graphs、stress 与 summaries 全部保留 | 安全 abstention 与 `2/96` strict yield 证据不变；不得降 ROPE/置信度或把 incomparable 改写为 winner |
| `RF-12` | earlier-fork fallback 的 candidate 与 independent latent 目录 | conditions、candidate manifest、condition audits、paired tracks、common support、bootstrap、oracle graphs、summary 与 `REJECTED` | 首帧/质量纠缠结论不变；fork/rho 搜索仍关闭 |
| `RF-14` | A1 rejected feature scan 的 `feature_records/` tensors | scene split、queries、feature index、control/primary probe metrics、result、summary 与 `COMPLETE` | ego-only positive 与 actor residual reject 均不变；不得事后调 probe 或启动 A1-CONFIRM/A2 |
| `RF-15` | B0 已拒绝的 128-video `candidates/` pool | generation/scored 索引、rank/diversity、machine gate、result、summary、`REJECTED` 与独立 sensitivity run | natural support reject 不变；不得扩 N、删 anti-collapse 或进入 AWR/SFT |

SVD-XT 本地快照也登记为可恢复清理对象。其 Hugging Face revision 固定为
`9e43909513c6714f1bc78bcb44d96e733cd242aa`；清理只把历史 SVD asset check 改为 `non-resident`，不会把已关闭
路线重新解释为“未测试”。`RF-01`、`RF-06`、`RF-09`–`RF-12`、`RF-14`、`RF-15` 的允许重开条件逐字
继续生效。

## 4. 已验证的 research 负结论

### RF-01：V1 synthetic projection 的 proxy inversion 与重复暴露

**原始命题**

在 32 个 synthetic latent cache clips 上，通过 low-noise absolute `x0` projection regression、real loss、
full-image anchor 与 mixed spatial/temporal LoRA 搜索，可以得到同时改善视觉质量和驾驶动力学的配方。

**观察**

- 16 个 100-step trial 的综合动力学 score 全部为负；排名前 4 个续训到 300 step 后没有反转；
- 相对 Base，t10 100-step 的 LPIPS `0.5088 → 0.4453`，但 static drift
  `8.2095 → 9.8610`，track acceleration `4.3953 → 5.7143`；
- t10 300-step 的 static drift 继续恶化到 `12.2478`，track acceleration 到 `6.5092`，eligibility
  从 `85.51%` 降到 `76.46%`；
- 100/300 optimizer steps 在该 cache 上约对应 25/75 次平均 replay exposure。

**研究结论**

当前 V1 配方能优化重建/外观 proxy，却没有学习到正确驾驶动力学；在小 synthetic cache 上增加 exposure
只会提高记忆和指标错配风险。可能原因包括 synthetic-to-rollout distribution mismatch、raw-v 参数化、
anchor 冲突和 mixed LoRA 外观优先，但这些单因果尚未被分别证明。

**禁止重复**

- 不继续旧 V1 Optuna、t10-800、旧 32-clip cache 或同搜索空间；
- 不把 LPIPS/loss 下降当作动力学改善；
- 不通过增加 step、`lambda_proj` 或 replay exposure 寻找事后翻转。

**允许重开**

只有新监督分布来自合法 Base rollout、独立动力学 endpoint 已预注册、样本 exposure 受控，并同时报告
视觉、静态、对象运动、coverage 和 per-scene worst case 时，才能提出不同命题。

**证据**

- `/root/autodl-tmp/runs/p2-tune-mini/`
- `EXPERIMENTS.md` 的 `P2 V1 调参归档`

### RF-02：未条件化 future state 不能定义生成错误

**原始命题**

用 source nuScenes clip 的 future ego pose 计算 expected static flow，并把 SVD rollout 与该 flow 的残差
作为物理监督。

**观察**

- 冻结 SVD Base 的 16-case 诊断中，GT-ego、identity、self-estimated residual 均值为
  `19.2887 / 2.1164 / 0.9320`；
- SVD image-to-video backbone 没有接收 source future ego pose、action 或 layout；
- GT-ego residual 因而混合了“未条件化未来分支不同”和真正的静态物理错误。

**研究结论**

数据集里存在的 future state 不等于模型条件。用模型未见的 future ego/box/track 审计自由生成 rollout，
会把条件不匹配伪装成物理违例。

**禁止重复**

- 未条件化 SVD 的正式 target/reward 禁止读取 future GT ego、box、track 或 source-future metadata；
- 不得把 `uses_future_gt=false` 只当 manifest 字段而不审计完整数据流。

**允许重开**

模型必须显式接收并验证对应 action/ego/layout condition，或 target 完全由 generated RGB 中可验证的证据
构造；两者都需要独立的 condition-compatibility gate。

**证据**

- `/root/autodl-tmp/runs/p2-v2-condition/p2-v2-cond16-s20260712-fff5ccb-97d2d05d/`

### RF-03：self-estimated static residual 变小不等于 target 合法

**原始命题**

只从 generated RGB 估计背景运动，可以绕开 future-GT 泄漏，并安全构造静态 correction。

**观察**

- 16-case 人工复核为 8 `yes`、4 `no`、4 `uncertain`；decisive 合理率 `66.67%`，低于预注册
  `70%`；
- 失败主要来自高覆盖 background mask 把车辆、护栏或 Base 伪影当背景传播，造成路面色块、撕裂和拖影；
- residual 数值降低没有预测 target 的视觉/语义合法性。

**研究结论**

无 GT 背景估计仍面临 foreground leakage、mask propagation 和生成伪影自强化。低 residual 只说明拟合了
某个运动模型，不能证明得到了正确静态世界。

**禁止重复**

- 不放宽既有人工阈值、不重标旧 review；
- 不通过扩大背景 mask、提高平滑或只看 residual 恢复该 static V1 branch。

**允许重开**

需要新的 foreground/occlusion 表示、独立质量审计和预注册人审，在 held-out 场景上同时通过 static
合理性、前景保持、首帧与伪影门禁。

**证据**

- `/root/autodl-tmp/runs/p2-v2-condition/p2-v2-cond16-s20260712-fff5ccb-97d2d05d/`
- `/root/autodl-tmp/runs/p2-v2-condition/p2-v2-gen04-panel1-s20260713-3cb8445/`

### RF-04：降低 acceleration/jerk 最容易学到“少动”

**原始命题**

对 observed point tracks 做 curvature/constant-dynamics smoothing，降低高阶运动能量即可得到更物理的目标。

**观察**

- P-CUR 修改 frame 0，最大 `10.165 px`，扩张 visibility `127`，turn preservation `83.05%`，
  dynamic-degree median ratio 仅 `0.112`；
- 更保守的 P-CON 仍只有 `88.79%` turn preservation 和 `0.736` dynamic-degree ratio；
- P-UNC 只有在 uncertainty、support、visibility、frame-0 与 motion invariants 联合约束后才通过 point-space
  machine gate；该通过也没有自动产生可信 RGB target 或 preference。

**研究结论**

acceleration/jerk reward 天然偏好冻结、减速、缩短轨迹或丢弃难点。驾驶视频的“平滑”必须在运动暴露、
转向、可见性和活动度非劣约束下解释。

**禁止重复**

- 不使用 acceleration reward、jerk reward 或 generic smoothing 作为单独 winner 规则；
- 不把 track 变短、support 变少或 dynamic degree 降低后的低能量记为改善。

**允许重开**

新 reward/oracle 必须通过 freeze、time-slow、track-dropout、visibility expansion、turn preservation 与
motion-exposure non-inferiority attack，并在共同 support 上比较。

**证据**

- `/root/autodl-tmp/runs/autoresearch-p0-projector-s20260714-v1/`

### RF-05：point-space 合法不等于 RGB/VAE counterfactual 合法

**原始命题**

把 P-UNC 轨迹 correction 通过 crop/resize/paste 渲染到 RGB，再用 full VAE 或 masked hybrid latent，
即可得到局部、可训练的反事实视频 target。

**观察**

- 7/7 frame-0 RGB/latent exact，mask 外 latent ratio 也低于阈值，但这些局部数值没有保证语义正确；
- index 34 的轨迹发生连续 correction，最终 changed pixel count 却为 `0`：整数 compositor 把小移动量化掉；
- full/hybrid target LPIPS 最大 `0.06805 > 0.05`；
- 出现 source duplication proxy `1`，以及 `588` 个没有 depth order 的 moved-component overlap。

**研究结论**

轨迹点可被纠正，不代表存在合法的视频反事实。source removal、disocclusion、遮挡顺序、背景显露、identity
和纹理一致性都是 target 定义的一部分；小 mask 或小 latent drift 不能替代这些语义。

**禁止重复**

- 不在当前 crop/resize/paste + VAE/hybrid target 上增加训练、loss、mask 或 cache；
- 不用 point-space machine pass 跳过 decoded RGB legality；
- 不用大型视频编辑器黑盒输出绕开同样的 provenance 与反事实门禁。

**允许重开**

新的 renderer/representation 必须先在 P1-style held-out gate 上证明连续 correction 可观察、source removal
正确、disocclusion/depth order 明确、identity 保持、VAE round-trip 合法、frame 0 与非目标区域稳定。

**证据**

- `/root/autodl-tmp/runs/autoresearch-p1-target-s20260714-v2/`

### RF-06：shared temporal LoRA 的 correction/locality entanglement

**原始命题**

temporal-only rank-16 LoRA 能在 object-only replay 上学习局部 residual-v correction，同时由 preserve loss
保持 mask 外和 frame 0。

**观察**

| 版本 | target error 降幅 | mask 外 drift | frame-0 drift |
|---|---:|---:|---:|
| C：absolute direct-v | `0.95%` | `20.59%` | `0.6094` |
| D：teacher-relative residual-v | `1.28%` | `16.84%` | `0.4077` |
| E：residual-v + trust region | `23.45%` | `6.52%` | `0.1948` |

预注册要求为 target error 至少下降 `80%`、mask 外 drift 不超过 `2%`、frame 0 近数值零，C/D/E 均失败。
固定 single-pair 高学习率可以降低 correction `95%+`，但 preserve 开启后仍有 `8.45%` outside drift 和
`0.2324` frame-0 drift。F0 对 preserve weight 的固定 sweep 有 11 个 checkpoint 达到 correction 条件，
却有 0 个同时通过 locality。

**研究结论**

问题不是“完全学不动”，而是共享参数对子区域/时段的修改与全局行为纠缠。single-pair overfit 证明优化器
能记忆 target，不能证明 8-pair capacity、局部性、rollout 或方法可行。

**禁止重复**

- 不继续同一 shared temporal LoRA 的 learning rate、rank、step 或 preserve-weight sweep；
- 不用单-pair correction 曲线替代 outside/frame-0/per-pair worst-case gate；
- 不在 capacity gate 前进入 rollout、正式 cache 扩量或长训练。

**允许重开**

需要结构上独立的局部参数子空间、明确的 tube/time routing 或其他能先验限制作用域的机制，并先在同一
noise bank 上同时通过 correction、outside、boundary、frame-0 与 held-out pair gate。

**证据**

- `/root/autodl-tmp/runs/p2-v2-pilot/`
- `/root/autodl-tmp/runs/autoresearch-f0-endpoint-s20260713-6845411/`

### RF-07：旧 correction 在 frozen raw feature 上缺乏可辨识信号

**原始命题**

绕开 RGB renderer，直接从冻结 SVD raw features 的 correlation/soft-argmax relation 中分辨 observed 与
projected tracks，再训练轻量 feature head。

**观察**

- 8 个固定样本、7 个真实 hook、3 个 sigma 的只读 probe 中，最细 stride-8 层仍有 `94.97%`
  correction 小于半个 feature cell；
- 没有层通过预注册 resolution/tracking promotion；部分粗层的高 PCK 来自 cell threshold 过宽，不能当信号；
- observed 与 projected heatmap/relation 几乎相同。

**研究结论**

当前 projector correction 与 raw feature grid 的组合缺乏可辨识性。直接训练 head 很可能学习插值、量化或
数值噪声，而不是驾驶动力学 relation。

**禁止重复**

- 不在相同 target/hook/grid 上直接增加 feature head、zero-conv 或 short-chain；
- 不把粗网格宽阈值下的高 PCK 当推荐层。

**允许重开**

可以研究连续 bilinear/Gaussian/correlation distribution 等新表示，但必须先在独立 held-out probe 上证明
signal 超过 measurement uncertainty，并且输入 target 已通过 RF-05 的 legality gate。

**证据**

- `/root/autodl-tmp/runs/autoresearch-f1-features-s20260713-72ac28c/`

### RF-08：evaluator 稳定不等于绝对物理有效

**原始命题**

使用独立 CoTracker3 evaluator 可以稳定排序 rollout，因此其 image-plane acceleration/jerk 可直接作为
物理 reward 或绝对改进指标。

**观察**

- E0 v3 的同视频重跑坐标、visibility 与 aggregate exact，三类扰动的最低 rank correlation 为
  `0.97619`；
- 但 resize 对绝对 aggregate 的影响中位数/最大值为 `10.03% / 31.84%`；
- 早期 evaluator scope 曾把 baseline 与其自身过滤子集做相关，说明“稳定性检查”本身也可能循环论证；
- evaluator pass 没有比较训练模型，也不能修复 target builder。

**研究结论**

独立 tracker 可以降低同源 metric hacking，但 camera/resize/visibility nuisance 仍会改变 image-plane 高阶量。
rerun exact 和 rank stability 只能说明协议可复现，不能标定真实世界物理量或证明监督合法。

**禁止重复**

- 不让 target builder 与 evaluator 共享同一 track/support 后再宣称独立验证；
- 不使用未做 camera compensation 的绝对 jerk/acceleration 作为物理真值；
- 不把 evaluator machine pass 写成 rollout improvement。

**允许重开**

新 evaluator 需要明确 camera nuisance 模型、独立 provider、有效 coverage、resize/codec/photometric stress、
盲人工校准和 scene-level uncertainty；训练 endpoint 仍须单独通过因果比较。

**证据**

- `/root/autodl-tmp/runs/autoresearch-e0-evaluator-s20260714-v3/`

### RF-09：same-scene sibling 不等于 preference 可辨

**原始命题**

只要 common-prefix siblings 保持首帧和场景结构，自动 physics scorer 就能从中稳定找到 winner。

**观察**

- PA1 v5 的 8 个结构盲审全部为 `same_scene`，说明候选仍是同一驾驶场景的合法不同 future；
- 后续 48-case preference 人审中，P1 common-prefix 子集为 22/24 `tie`、1 `uncertain`、1
  `both-invalid`，人工 decisive 为 `0/24`；
- 同一 P1 子集的机器标签却是 13 `a_wins` + 11 `b_wins`。

**研究结论**

结构合法性只排除了 identity/layout mismatch，不保证干预产生了人类可感知、可归因的物理差异。candidate
generation 与 preference oracle 是两个独立识别问题，前者不能由后者补救。

**禁止重复**

- 不从 same-scene、shared-prefix、first-frame exact 直接推导 physics winner；
- 不在盲人工 decisive yield 未达预注册阈值前生成训练标签。

**允许重开**

新 candidate generator 必须在 scene-disjoint pilot 上同时证明结构合法、画质可比、差异局部可见、人工 strict
yield 足够，并能将偏好归因到 motion 而非外观或相机变化。

**证据**

- `/root/autodl-tmp/runs/autoresearch-pa1-branch-s20260715-v5/`
- `/root/autodl-tmp/runs/autoresearch-pa2-pair-expanded-s20260715-v1/`

### RF-10：candidate-specific support 与 forced binary 会制造 false-strict

**原始命题**

对每个 candidate 独立选 query、独立跟踪/平滑，在各自 support 上平均 P-UNC energy，再按最大 margin 强制
`a_wins/b_wins`，可以得到可信物理 pair。

**观察**

- 120 conditions 通过机器门禁后得到 53 个 global pairs，但 48-case 人审中 machine/human 同时 decisive
  只有 8 例，agreement `4/8 = 50%`，Wilson lower `0.2152`；
- P1 的 24 例机器全部 decisive，而人工 decisive 为 0；
- scorer 选择出现 2 个 catastrophic failure；
- 旧 scorer 比较的是不同 query、visibility、track survival 和 smoothing 结果，energy margin 混入 support
  mismatch 与 measurement noise。

**研究结论**

forced binary 会把不可比较或测量不确定的 pair 伪装成严格偏好。candidate-specific evidence 破坏了成对
反事实的共同测量基础，即使 pair confidence 数值很高也不能恢复语义。

**禁止重复**

- 现有 53 global pairs 和旧 local labels 禁止进入 DPO、AWR、SFT 或筛选；
- 不扩大同 scorer review、不重调 margin、不将 human tie 重编码为 winner；
- 不比较不同 support 上的平均能量后宣称局部化偏好。

**允许重开**

必须使用共同 first-frame query、paired tracking、common visible support、camera comparability、measurement
uncertainty，并允许 strict/tie/incomparable；阈值只能在冻结 calibration split 上确定。

**证据**

- `/root/autodl-tmp/runs/autoresearch-pa2-pair-expanded-s20260715-v1/`

### RF-11：安全 abstention 解决 false-strict，但不自动解决数据 yield

**原始命题**

common-support、bootstrap uncertainty、measurement ROPE 和 selective partial order 可以把旧 sibling RGB
转化为足量且可信的 preference 数据。

**观察**

- UPO v2 在 10 个冻结 human ties 上 false-strict 为 0，invalid high-confidence strict 为 0；
- identical rerun、六类 shortcut stress、cycle 与双 seed stability 均通过；
- 116/120 query sets 有效，但未参与校准的 96 conditions 最终只有 2 `strict`、0 `tie`、94
  `incomparable`；
- 这说明 oracle 学会了安全 abstain，却没有让旧 candidate distribution 变得可辨。

**研究结论**

uncertainty-aware partial order 是必要的可信性层，但不是 preference 数据发生器。低 false-strict 与高
strict yield 是两个独立门槛；用更安全的 oracle 不能挽救缺乏可辨运动差异的 candidates。

**禁止重复**

- 不降低 valid-only ROPE、bootstrap confidence、support 或 camera/quality comparability 阈值；
- 不把 `incomparable` 改成 tie，更不能随机/按微小均值差指定 winner；
- 不从 2 个 strict 中挑选训练数据或宣称 prospective precision。

**允许重开**

必须改变候选生成的可辨识性，同时冻结 UPO v2 oracle 作为测量工具；新 pilot 需预注册最低 legal-condition、
human strict precision 与 strict yield，而不是修改 oracle 迎合 candidates。

**证据**

- `/root/autodl-tmp/runs/autoresearch-pa2-upo-s20260716-v2/`
- config fingerprint `249c47f43b638388bfb6040f1e2a06ac99ab781e21172207681322541f76852f`

### RF-12：earlier fork 的可辨性与身份/质量纠缠

**原始命题**

把 common-prefix fork fraction 从 `0.6` 提前到 `0.4`，保持 `rho=0.04`，可以扩大 sibling motion 差异，
同时保留同一场景、首帧和视觉质量。

**观察**

- 唯一预注册 fallback 使用 8 个 scene-disjoint conditions、每条件 exact Base 两次和 4 个 siblings；
- Base guard、callback/perturbation、query、cycle 与 oracle fingerprint 通过，7/8 conditions legal；
- `scene-0736` 的 4 个 siblings 全部触发双方 temporal jump，其中两个首帧 RGB RMS 为
  `0.1215/0.0976`，超过冻结门槛；
- 全池只有 1 strict、0 tie、7 incomparable，machine gate 失败，未进入结构人审。

**研究结论**

在当前 SVD continuation 机制中，单纯 earlier fork 增加的不只是目标运动差异，也会放大首帧/时序与画质
不稳定。可辨性、身份保持和质量不是可独立调节的旋钮。

**禁止重复**

- 不继续 common-prefix fork/rho/candidate 数搜索；
- 不事后删除失败场景后使用 7 个 legal conditions 或唯一 strict；
- 不跳过未生成的结构盲审，不切双卡扩量，不进入训练。

**允许重开**

需要新的干预机制或显式可控 backbone，在构造上分离 motion factor 与首帧、identity、camera、appearance；
必须先用新 ID、新场景和冻结质量/人审门禁验证。继续同一 SVD common-prefix sibling 参数搜索不算重开。

**证据**

- `/root/autodl-tmp/runs/autoresearch-pa2-cand-fallback-s20260716-v1/`
- commit `a9c60588be247c2f8d08b96c1a993e74b8edf559`
- config fingerprint `b6806afc33bd8ea85fd96ead058997c51549ae97277fe7cf5c90fbb9fe3ed7c9`

### RF-13：真实采样率不等于 SVD fps micro-conditioning 的安全设定

**原始命题**

nuScenes keyframe 视频约为 2 Hz，而旧生成协议固定 `fps=7`；把 SVD fps 输入改为 2 或 4，可能在不损害
质量的前提下修复 Base motion 的时间尺度 mismatch。

**观察**

- 32 个 scene-distinct 真实 clips 的中位相邻时间为 `0.5000 s`、有效 fps 为 `2.0000 Hz`；
- 8 conditions × 2 seeds 的 paired Base audit 中，`fps=2/4` 相对 7 的 dynamic degree 分别增加
  `24.74%/10.05%`，image velocity 分别增加 `77.97%/110.71%`，95% bootstrap CI 均不跨 0；
- `fps=2` 同时失败首帧、锐度、闪烁、track survival 与 acceleration safeguard；
- `fps=4` 虽保住首帧、锐度和闪烁，但 survival ratio 为 `0.859 < 0.90`，acceleration p95 ratio 为
  `1.950 > 1.25`；
- 两档候选均通过 anti-low-motion floor，失败不是因为少动，而是更大运动伴随更差可跟踪性和高阶稳定性。

**研究结论**

SVD 的 fps 输入是 learned micro-conditioning，不是物理标定的播放速率。真实 timestamp mismatch 存在，
但数值匹配不能作为安全修复；在当前官方 SVD-XT、8-frame、25-step 协议下直接改为 2/4 已被拒绝。
后续 Route A 的真实运动 target 仍必须使用实际 delta-t，不能把保留 `fps=7` 解释为时间问题已解决。

**禁止重复**

- 不因 `2 Hz == fps input 2` 跳过生成质量和 motion safeguard；
- 不按 dynamic degree 或 image velocity 增幅单指标选择 fps；
- 不把 image-plane acceleration 当真实世界加速度，也不把统一 playback fps 当生成时间标定；
- 不用后续训练收益事后反选 R1 的 fps。

**允许重开**

只有新的显式时间/action conditioning、经过校准的 continuous-time parameterization，或跨 backbone 的独立
证据，才可用新任务重开。重开仍需同 condition/noise 的配对审计、真实 delta-t 与全部安全端点。

**证据**

- `/root/autodl-tmp/runs/route-pivot-r1-temporal-s20260718-v1/`
- commit `f4b4cd5872d732dc2694d6fb9bae53fcf6dd7304`
- config fingerprint `a1ed2f2527e9f9ea6a07ed6830b0c4ee6522c38b2fcae5dd1d6db239e86e10ed`
- [`ROUTE_PIVOT_TEMPORAL_AUDIT.md`](../ROUTE_PIVOT_TEMPORAL_AUDIT.md)

### RF-14：冻结 SVD 的 ego signal 不等于可分离的 actor residual signal

**原始命题**

SVD-XT 的冻结 temporal representation 同时包含 sparse background ego-flow 与 ego-centered actor residual；
用 scene-disjoint compact feature、5×5 local cost 和小型 matched-capacity probe，应能让 A-RES 超过
zero-residual 与 A-ABS，并避免给 parked/stopped actor 预测大 residual。

**观察**

- 24/8 个 scene-disjoint clips 上，21 个 layer/sigma 配置全部有效，train/dev 分别有 567/176 actor 与
  2,304/768 ego queries；
- ego-flow 对最佳 zero/mean baseline 的改善为 `17.86%–25.01%`，说明背景 camera/ego motion 可读；
- moving actor A-RES 对 zero-residual baseline 的改善全部为负，最佳仍是 `5.862 px` 对 `2.660 px`；
- A-RES 对 A-ABS 虽改善 `35.94%–65.81%`，但 absolute tracking baseline 太弱，不能替代 zero-residual 主门；
- stationary prediction / moving target median ratio 为 `3.292–5.062`，直接暴露 actor appearance、ego motion
  与独立 actor motion 的纠缠；0 个配置通过 primary gate，0 个 layer 在两个 sigma 稳定。

**研究结论**

当前 compact frozen-feature probe 能读取 ego/background motion，却不能安全地局部化 actor-independent
residual。失败不是“模型完全没有运动”，而是 driving-specific motion entanglement：共享表示提供了足够的
camera-motion 线索，却让小 residual 被 scene/appearance/ego 共变淹没，并在 stationary actor 上产生大假运动。

**禁止重复**

- 不因 A-RES 优于较弱的 A-ABS 就忽略 zero-residual 与 stationary safeguard；
- 不事后搜索 ridge、projection dim、MLP 宽度、layer/sigma 或降低 scan 门槛来制造 top-2；
- 不把 ego-only positive 包装成 EgoActor-Align，不启动 A1-CONFIRM、A2 或 temporal LoRA auxiliary training；
- 不把 real-video future geometry 接入 generated-rollout evaluator。

**允许重开**

需要实质改变可辨识性，例如显式 ego action/trajectory condition、对象中心的跨帧 token/track architecture、
更强且预注册的 actor supervision，或 action-conditioned driving backbone。新路线必须重新做 scene holdout、
zero/stationary、frame shuffle、低运动与局部泄漏门禁；只扩大同一 linear probe 不算重开。

**证据**

- `/root/autodl-tmp/runs/route-pivot-a1-feature-scan-s20260718-v2/`
- commit `b27fb5a`
- config fingerprint `982b13b356cf5dc9b947704fad9b8da2ebcb95747edc1b43a61f210dc50d61fe`
- result fingerprint `467a50bd546f729a395d963ddbf52070c24039c65191729ef99625c11e9a05f6`
- [`ROUTE_PIVOT_MOTION_FEATURE_AUDIT.md`](../ROUTE_PIVOT_MOTION_FEATURE_AUDIT.md)

### RF-15：SVD natural seed diversity 不等于可利用的安全 preference support

**原始命题**

不再制造 denoising siblings，而在冻结 SVD-XT 的自然独立 seed 分布中做 best-of-N，可能已有足够多
P-UNC 与独立 CoTracker 一致认可的 motion winner，为 condition-relative AWR/SFT 提供 support。

**观察**

- 16 个 scene-distinct conditions 先各 4 samples；`0/16` 有至少两条合法、非重复、可比较候选，故按协议
  扩到 8，总计 128 videos；
- 112 个非 Base selection candidates 中只有 7 个通过完整 eligibility，最终只有 `1/16` diverse；主要失败
  是 motion floor/ceiling 59、absolute/relative first-frame 51/33、survival 30、flicker 29、P-UNC valid 20；
- 仅 6 个 conditions 能形成 P-UNC-best 对照，对 random/Base 的 CoTracker win-credit 都是 `41.67%`；
- P-UNC selection 仍有 1 次 low-motion、3 次 catastrophic safeguard failure，positive-improvement condition 为 0；
- 18 dB absolute floor 确实错误拒绝了部分官方 Base，但移除 absolute、全部 first-frame、再加 motion checks
  后分别只有 `4/16`、`6/16`、`10/16` diverse；只有删除整套 anti-collapse 才能达到 16。

**研究结论**

自然 seed 带来显著 RGB 与 scorer 能量差异，却没有带来足量的**安全** preference support。问题不只是
ranking 精度，而是 frozen SVD distribution 中，较优 scorer 值与首帧、活动度、survival、flicker/锐度强烈
纠缠。best-of-N、AWR 或更多人工标签都不能从不存在的合法 support 中恢复收益。

**禁止重复**

- 不把 N 扩到 8 以上，不搜索 seed pool、CFG、scheduler、fps、motion bucket 或后处理；
- 不移除 motion floor、survival、sharpness、flicker、first-frame guard 来增加 eligible yield；
- 不在人审中覆盖 machine rejection，不从唯一 diverse condition 或 7 个 eligible candidates 挑训练数据；
- 不进入 P-UNC-weighted AWR/SFT、DPO 或 reward-guided sampling。

**允许重开**

需要改变 rollout support 本身：优先迁移到显式 action/trajectory-conditioned driving backbone，或证明新
生成架构能独立控制 ego/actor motion 而不改变首帧、身份与质量。新路线仍需固定自然 sample budget、独立
evaluator 与完整 anti-collapse；只换 scorer 或增加 N 不算重开。

**证据**

- `/root/autodl-tmp/runs/route-pivot-b0-natural-rollout-s20260718-v1/`
- `/root/autodl-tmp/runs/route-pivot-b0-sensitivity-s20260718-v1/`
- commits `06dc211` / `d9ac65d`
- source/result fingerprints `29fc4036a7bd` / `d52b5f71da38`
- [`ROUTE_PIVOT_NATURAL_ROLLOUT_AUDIT.md`](../ROUTE_PIVOT_NATURAL_ROLLOUT_AUDIT.md)

### RF-16：layout controllability 不等于 action-disentangled actor physics

**原始命题**

只要迁移到有 nuScenes、3D box、HD map、camera pose 和官方权重的 controllable driving generator，就能自动
解决 SVD 上暴露的 ego–actor motion entanglement，并为物理 preference 提供合法 sibling。

**观察**

- OpenDWM 与 MagicDrive-V2 的 layout checkpoint 把逐帧 future 3D boxes/maps 直接作为条件；它们能验证条件
  adherence，却不要求模型自主预测 other-agent future；
- MagicDrive-V2 在 224×400 低分辨率上有明确单卡显存证据，但 stage-3 训练至少 4 GPU，且 actor 仍由 box
  sequence 规定；
- VISTA 有合法 trajectory/cmd/steer/speed/goal conditions，且 actor future 自由生成，但官方建议采样显存
  至少 32 GB，架构仍基于 SVD；
- ReSim 公开 `exp0_no_carla` checkpoint、8 点 future ego trajectory、历史帧预测与不含 future actor boxes 的
  nuScenes schema，最接近本项目的可辨识性要求；
- 公开 ReSim checkpoint 不包含论文 CARLA 非专家数据，官方单 GPU 入口也没有证明 24 GB RTX 4090 已通过；
  固定源码树未发现论文 evaluator/IDM/Video2Reward 的发布实现；
- 最小选择性下载约 34.4 GB，当前 42 GB 空间无法保持 30 GB 安全线。

**研究结论**

显式 **ego** trajectory 是重开 actor residual 可辨识性的必要条件，但不是物理偏好证据；future **actor** boxes
则会把待检验结果直接变成输入。Route C 因此选择 ReSim `exp0_no_carla` 做下一阶段 feasibility，而不是按画质
或单卡显存选择 layout model。该选择只改变问题结构，不证明 preference support、dense localization 或训练收益。

**禁止重复**

- 不把 box/map adherence、camera-path adherence 或 FID/FVD 改善称为 other-agent physics；
- 不用 future actor boxes/tracks 生成“自由”rollout，或作为其正式 evaluator truth；
- 不把 `exp0_no_carla` 写成包含 CARLA dangerous/non-expert behaviors 的完整 ReSim；
- 不因官方脚本为单进程就宣称 24 GB 4090 可运行；必须实测峰值与完整输出；
- 不跳过 Base/action feasibility 与 candidate-yield gate，直接在 ReSim 上实现 vanilla DPO、DenseDPO 或 AWR；
- 不在当前磁盘安全线下自动下载大权重，也不先切双卡掩盖单卡/候选失败。

**允许重开**

用新的 C1 计划按顺序完成：

1. ReSim 单卡 9-latent-frame Base/action smoke；
2. action sibling 的因果响应与 rollout sibling 的安全 support 分离审计；
3. common-support UPO 的 false-strict、strict yield、stationary/moving、action shuffle 与 low-motion gate；
4. 只有前述 machine/human gate 通过后，才用两卡做短 LoRA dense safeguarded capacity test。

若 full-CARLA ReSim、VLA-World 或其他显式 action backbone 后续正式发布，必须固定新源码/权重 fingerprint 并
重新执行同样门禁；发布本身不自动晋级。

**证据**

- `/root/autodl-tmp/runs/route-pivot-c0-backbone-audit-s20260718-v1/`
- [`BACKBONE_MIGRATION_AUDIT.md`](../BACKBONE_MIGRATION_AUDIT.md)
- [ReSim official source](https://github.com/OpenDriveLab/ReSim/tree/bf13dff45975eabbabc4e7de778207d2bb785e9b)
- [ReSim paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/f502981cbe221d857ad409450a7917c3-Paper-Conference.pdf)

### RF-17：local ego-motion proxy 的 class/turn 在真实 nuScenes 上不可辨识

**原始命题**

在冻结的 48 个 scene-disjoint calibration clips 上，用 RAFT + 稳健 affine background fit 提取的
camera-motion 特征，经 ridge 校准后应达到预注册门槛：moving balanced accuracy ≥0.70、turn-sign
accuracy ≥0.75、位移 Spearman ≥0.50，且位移 MAE 同时优于 constant 与 command-only baseline；从而
为后续 ReSim E-vs-F action screen 提供可辨识的机器 action 指标。

**观察**

- `C1B-00` 单卡 smoke 已通过；校准资产 2,842 路径齐全，selection fingerprint `a1ae39db…`；
- v1 工程失败：不规则 CAM_FRONT 间隔使朴素最近邻重采样撞帧（不计 gate）；
- v2 正式校准：48/48 `proxy_valid`；位移 Spearman `0.9269`，MAE `1.264 m` 显著优于 constant
  `5.412` 与 command-only `5.748`；
- 分类失败：moving BA `0.444`、turn-sign `0.583`；held-out forward 6 例中 4 判 left、1 判 right；
- stationary 位移中位较低，但 p95/max 仍可到数米，不能单独挽救 class/turn gate。

**研究结论（v2 / 四类 ridge）**

当时预注册的四类 ridge **local ego-motion proxy** 足以做粗粒度位移回归，但不足以在真实视频上可靠区分
forward/left/right。因此在 v2 时机器侧 action response 指标不可辨识；`C1B-01` 记为 `blocked`，不得
把“位移相关高”包装成 action feasibility pass。这不是 ReSim 生成失败，而是 evaluator/proxy 前置失败。

**重开与收口（v3 / kinematic-lateral）**

V6 在同门槛、同 scene split、不偷看生成 future 的约束下，预注册并执行了唯一允许的重开机制
`local-ego-motion-proxy-v2-kinematic-lateral`（显式 yaw/分段侧向 + 运动学规则导出 class）。v3 held-out
达到 moving BA `0.778`、turn-sign `0.750`、位移 Spearman `0.953`，因此 `C1B-01` 记为 `done` 并解锁
`C1B-02`。RF-17 对**四类 ridge 配方**仍成立：不得退回 v1/v2 ridge，也不得以降 BA/turn 阈值或
位移-only 冒充 pass。后续 H1 失败记入 `RF-18`，不再归因于 proxy 不可辨识。

**禁止重复**

- 不把阈值降到观察值附近后再宣称 pass；
- 不用位移-only 或 command-label leakage baseline 替代 class/turn gate；
- 不在未通过校准前查看生成 future、扩 sample、换 seed 或重选更容易的 scene；
- 不把本 proxy 称作 ReSim IDM / Trajectory Difference / ADE；
- 不退回已失败的四类 ridge 分类头，也不把 v3 kinematic 规则改成看过 screen 后再调参。

**允许重开**

对 ridge 类配方：必须用**新预注册计划**更换可辨识机制（例如独立 tracker ego 估计或官方 IDM），
并在看任何生成结果前冻结阈值与 scene split。仅调 ridge α、affine 超参或扩大同分布校准集不算新机制。
V6 内已执行的 kinematic-lateral 重开不得再次“同机制重跑”以追求更高分数。

**证据**

- `/root/autodl-tmp/runs/resim_c1_v6/C1B-01/resim-c1b01-proxy-s20260719-v1/`（engineering）
- `/root/autodl-tmp/runs/resim_c1_v6/C1B-01/resim-c1b01-proxy-s20260719-v2/`（`BLOCKED`）
- `/root/autodl-tmp/runs/resim_c1_v6/C1B-01/resim-c1b01-proxy-s20260719-v3/`（kinematic-lateral pass）
- [`MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md`](../v6/MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md) §6.5

### RF-18：公开 ReSim exp0_no_carla 的 E-vs-F action response 不足

**原始命题**

在冻结的 10 个 scene-disjoint contexts 上，同一 seed 的 expert-conditioned（E）相对 action-free（F）
应在预注册 local ego-motion proxy 的 action error 上至少 7/8 moving 胜出，且 median paired improvement > 0；
同时 future 像素效应超过 B00 null、history 效应留在 null band、质量与 stationary 不被破坏。

**观察**

- `C1B-00`/`C1B-01` 工程与 proxy 可辨识性均已通过；
- `C1B-02` v2 完成 10×2 = 20 路 L1 采样；future_ok `8/8`、history_ok `7/8`、quality `8/8`、stationary ok；
- action gate 失败：E wins 仅 `3/8`，median improvement `-0.107`；
- 常见失败模式：E 被 proxy 判为 stationary/错转向，而 F 反而更接近请求 class（尤其 forward）；
- 大 future 像素差存在，但不能转化为正确的 ego action 一致性。

**研究结论**

在公开 `exp0_no_carla` 支持与当前单卡 2.4 s / 256×448 协议下，**不能**宣称可靠的 expert trajectory
因果响应。H1 `rejected`；不得进入偏好支持（C1P）或 adapter 训练（C1S）。这不是磁盘/OOM 工程失败。

**禁止重复**

- 不把 E/F 像素差、CoTracker survival 或画质 tie 包装成 action pass；
- 不降 7/8 门槛、不扩 seed、不重选更容易的 scene、不用 M sensitivity 补主 gate；
- 不跳过人审或用人工观感覆盖机器 reject；
- 不因此自动切换双卡或完整 4 秒/更高分辨率“救场”。

**允许重开**

需要新预注册计划，并至少改变问题结构之一：完整 CARLA/非专家 checkpoint、官方 IDM evaluator、
更长 horizon、或不同 action 接口；且仍须先过 Base/proxy/action 门禁。

**证据**

- `/root/autodl-tmp/runs/resim_c1_v6/C1B-02/resim-c1b02-screen-s20260719-v2/`
- [`MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md`](../v6/MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md) §6.8 / §12
- [`C1_V6_FINAL_REPORT.md`](../v6/C1_V6_FINAL_REPORT.md)

## 5. 未决 research 风险

以下风险必须进入下一份计划，但当前证据不足以写成已解决或已证伪。

### UR-01：driving-specific motion entanglement

生成视频中的 ego/camera motion、对象独立运动、深度、遮挡、appearance flicker 与 sampler noise 会共同改变
image-plane tracks。当前 robust affine residualization 只是一阶 nuisance control，没有证明能在复杂转弯、动态
遮挡、多对象交互中隔离真实对象动力学。

下一路线必须定义可证伪的 disentanglement test，例如已知 camera perturbation、对象 motion-preserving
transform、scene-level counterfactual 或显式 action condition；不能只报告 reward 下降。

### UR-02：低运动偏置与活动度守恒

现有 stress test 能拒绝若干 freeze/time-slow attack，但尚无训练后模型。即使数据 oracle 安全，diffusion
fine-tuning 仍可能通过降低 dynamic degree、缩短 visibility、模糊目标或减少运动多样性来优化平均目标。

任何训练必须把 motion exposure、dynamic degree、track survival、速度/转向分布和多样性作为 hard
non-inferiority endpoint，而不是补充图表。

### UR-03：support selection 与 missing-not-at-random

common support 避免直接比较不同轨迹，但容易只保留最简单、最稳定的像素。当前没有证明被丢弃的 query
与困难对象/遮挡/高速运动无关。高 precision 的 surviving subset 可能造成系统性 selection bias。

需要报告 support coverage 的对象类别、深度、速度、图像区域和时间分布，并对 track dropout 做因果压力测试。

### UR-04：preference identifiability 与 candidate generator

旧候选池表明多数结构合法 siblings 对人类仍是 tie。尚未找到能在保持首帧、场景、画质的同时稳定制造
局部运动差异的 intervention。新的 scorer、DPO loss 或更多标注都不能替代这个前置识别问题。

新方案应先以小规模 blind pilot 测量 human strict yield 和归因理由；在 yield 门槛前不实现 trainer。

### UR-05：dense/localized alignment 的参数泄漏

当前只验证过旧 endpoint shared temporal LoRA 的 locality 失败，尚未训练 partial-order tube objective。
attention/LoRA 参数共享仍可能让局部 tube loss 改变全图、首帧、静态背景和未标注对象。

任何 dense safeguarded alignment 都必须先做 single-pair 与 multi-pair correction/outside/boundary/frame-0
capacity gate，并与数据筛选、global DPO 和 no-localization baseline 对照。

### UR-06：短时、单相机与外部有效性

现有 preference 资产为 14-frame、CAM_FRONT、SVD-XT short-horizon。没有五相机一致性、长时 rollout、
闭环驾驶、不同 backbone 或真实 action-conditioned world model 证据。当前负结论和任何未来正结果都不能
无门槛外推到这些场景。

## 6. 跨路线必须保留的原则

1. **先证明监督对象存在，再训练。** target/preference 必须可观察、合法、可定位且可重复测量。
2. **条件必须进入模型。** 数据集 future state 不能替代生成器实际 condition。
3. **共同证据优先。** paired candidates 必须用共同 query、support、时间窗和 nuisance model 比较。
4. **允许拒答。** tie 与 incomparable 是正式结果，不能为了数据量强制 winner。
5. **运动非劣是 hard gate。** 任何平滑/对齐都必须排除 freeze、time-slow、dropout 和模糊投机。
6. **局部 loss 不保证局部参数更新。** 必须实测 outside、boundary、frame-0 和未标注对象。
7. **独立 evaluator 不能修复非法 target。** 数据、目标、训练和评估分别过门禁。
8. **single-pair overfit 不是方法结果。** 至少报告 held-out、per-scene worst case、coverage 和 provenance。
9. **machine pass 只解锁下一门禁。** 不自动解锁训练、双卡、论文 claim 或人工结论。
10. **失败范围不能过度外推。** 每个 reject 都只否定明确的 target、backbone、parameterization 与协议组合。

## 7. 新计划的防重复检查表

提交任何 V5 或新路线前，逐项回答：

- [ ] 新候选机制与 `RF-09/RF-11/RF-12` 有何本质区别？
- [ ] 是否读取了模型未接收的 future GT，或通过预处理间接泄漏？
- [ ] target/preference 在最终 RGB/latent 表示中是否可观察且语义合法？
- [ ] 是否使用共同 query/support，并显式表达 tie/incomparable/uncertainty？
- [ ] human strict yield、precision、both-invalid 与 failure reason 的预注册门槛是什么？
- [ ] freeze、time-slow、track-dropout、camera、resize、画质与 support-selection attacks 如何执行？
- [ ] 如何证明不是靠减少运动、模糊目标、缩短轨迹或丢掉困难对象获胜？
- [ ] localized objective 如何通过 correction/locality/frame-0 capacity gate？
- [ ] 独立 evaluator 与训练 evidence 如何隔离？
- [ ] 哪个单一卡门禁失败时停止，什么条件才允许人工评审、训练或双卡？

如果答案只是“更多数据”“更大模型”“更长训练”“更低阈值”“换成 vanilla DPO/AWR”或“再调一个
fork/rho”，则该计划仍在重复本账本中的旧问题。

## 8. 历史材料

早期完整复盘、预注册、路线决策、历代 Physics-DPO/DrivePO 计划和人工评审提示词已集中到
[`archive/2026-07/`](../README.md)。归档材料保留当时的术语与计划状态，只用于追查证据来源；
本账本负责跨阶段的 research 结论去重，不能替代原始 run。

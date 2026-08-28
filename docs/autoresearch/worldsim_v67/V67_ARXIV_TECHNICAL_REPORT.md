# WorldSim V6.7 ArXiv 技术报告：从端到端可靠性捷径到 Actor uncertainty × trajectory boundary

- 分支：`research/worldsim-v6.7-anisotropic-surface`
- 报告状态：`P113 outcome pending final fill`
- 主要硬件：单张 RTX 3090 24GB
- 证据角色：development、consumed cross-cohort、scene-level independent confirmation 严格分开

## 摘要

WorldSim V6.7研究的问题是：给定候选 Ego 轨迹 `τ`，未来 `H` 秒中它实际访问的 world/Actor states 是否可靠，以及这种
可靠度能否用于固定覆盖率排序。研究首先证明了一个重要但有限的信号：P81在10个未读场景的全 Actor-query row 上，冻结
query scorer将fixed-50% selected unreliable events从Actor-only的57降至26。然而，当评价对象收紧为真正被轨迹访问的Actor
states或trajectory-level any-failure时，P84--P94的fixed summary、Deep Sets、attention、ordinal、continuous、quantile、
Gaussian、BCE与ensemble全部不能稳定超过Actor-only。其原因不是简单容量不足，而是endpoint Actor prediction error本身不依赖
`τ`，候选轨迹只改变membership。

P95据此将endpoint改为task-conditioned occupancy-decision flip，但端到端query classifier在development很强、独立cohort却
反转；hierarchical temporal/Actor encoder和更多source data都未恢复。最终成功的迁移不是继续扩大query网络，而是将问题因子化为：
网络只预测Actor未来位置误差分布，candidate `τ`只通过signed boundary clearance进入解析查询。P107的scalar q90 uncertainty
tube在两个consumed cohorts一致成立，并在P108的新10-scene cohort将fixed-50% occupancy flips从Actor-only的35降至5；P109的
directional diagonal-Gaussian进一步以boundary-normal projection获得更强排序。P111表明纯clearance geometry已经是强baseline，
因此最终论文贡献必须区分“factorization超过Actor-only”与“learned uncertainty超过geometry”。P113专门冻结第二个未读cohort
检验后者；最终数字将在该one-shot read后写入。

本研究不提供collision probability calibration、planner/policy authority、closed-loop性能或safety guarantee。可辩护贡献是一个
任务条件化但分层的可靠性接口：`Actor uncertainty distribution × candidate-trajectory boundary query`。

## 1. 研究对象与评价合同

对场景中的Actor `i`、未来离散时间 `t` 和候选 Ego 轨迹 `τ`，我们区分三个对象：

```text
E_i,t       = Actor constant-velocity forecast error
F_i,t(τ)    = predicted-vs-observed occupancy decision flip near τ
C_i,t(τ)    = Actor-to-τ signed boundary clearance
```

早期模型直接学习 `score(Actor history, τ) -> reliability`。最终模型改为：

```text
p(Actor future residual | Actor history, t)
                    ×
analytic boundary query(C_i,t(τ))
                    ↓
trajectory-level reliability ranking
```

主要评价采用每scene固定50% coverage，报告被选择trajectory中的occupancy-flip event count与全排序AUROC。固定覆盖率避免
事后阈值优化；event count是任务选择指标，AUROC用于区分同一operating point下的整体排序。Interaction radius固定为
`Actor half-width + 1.0m Ego half-width`，预测horizon为3.5s、9个future samples。

## 2. 递进研究链

### 2.1 全row信号与visited-state对象失败

P81在9,559 Actor-query rows、735 unreliable endpoint events上取得query/Actor/P73 selected events=`26/57/45`，query相对
Actor-only减少54.39%，10/10 scenes不增，AUROC=`0.94147/0.92947/0.93666`。但这允许模型优先选择远离Ego轨迹的Actor rows。

收紧对象后，P84在2,113个visited rows上为`235` events，劣于P75的`208`；P85在1,089条visited trajectories上为
`203`，劣于P75的`199`。P86--P94覆盖fixed summary、Deep Sets、Set Transformer、ordinal thresholds、Huber max-error、
q90 pinball、heteroscedastic Gaussian、direct BCE与三成员deep ensemble；最佳P86也只有query/Actor/P75=`187/193/199`，
query-over-Actor改善3.11%，未达冻结10%，absolute reduction仅37.48%。这条负链关闭“visited Actor最大位移误差”对象，而
不是触发architecture sweep。

### 2.2 真正task-conditioned occupancy flip仍发生跨cohort反转

P95把监督改为9个future samples上constant-velocity Actor path与observed Actor path相对`τ`的occupied/free decision是否
翻转。Development cohort上，P95 query/Actor/P75=`7/28/13`，AUROC=`0.83952/0.69366`，4/4 gates通过。P102进一步用
time-token encoder和Actor set pooling得到`4/27/13`，成为development best。

独立P96却发生反转：P95=`8/5/12`、AUROC=`0.65542/0.71181`；P102的prospective secondary P103=`9/7/12`，
relative reduction同样为负。P104 time-local auxiliary、P105 multi-task recovery和P106全source scaling没有恢复独立结论。
这证明端到端query classifier可以学习cohort-specific Actor/trajectory interaction shortcut，development内更强的结构和更多数据
不等于task-conditioned迁移更稳。

### 2.3 Actor uncertainty与candidate boundary解析因子化

P107网络完全不读取candidate `τ`。它从19维Actor history/dynamics和normalized future time预测9-step position-error q90；
trajectory score固定为：

```text
max_actor,time q90_error_i,t / max(abs(C_i,t(τ)), 0.05m)
```

在consumed P81/P96上，P107 query/Actor/P75分别为`2/36/13`与`2/9/12`，query AUROC分别为`0.92901/0.87305`。
两个异质cohort方向一致，区别于P95/P102反转。

P108在新的10-scene cohort进行一次scene-level independent primary read：8,766 rows形成1,531 trajectories和116 flips；
fixed-50%选择764条时，P107/Actor/P75=`5/35/20`，absolute reduction=91.36%，query-over-Actor reduction=85.71%，
AUROC=`0.95107/0.77605`。两项冻结decision均通过，支持Actor uncertainty与candidate boundary分离的主结论。

### 2.4 Directional uncertainty与geometry归因边界

P109将scalar tube升级为Actor-only Ego-frame longitudinal/lateral residual的diagonal Gaussian，训练目标为Gaussian NLL；推理时
只沿Actor-to-Ego boundary normal投影mean/variance，形成linearized standardized crossing margin。916,722个Actor-time tokens、
6,000 steps后final NLL=`-3.64128`。Consumed P81/P96均选择0 events，AUROC=`0.96764/0.90434`。

在P108同一read上，target出现前冻结的P110 directional secondary为`1/53/20`，AUROC=`0.96027/0.69142`。但P111
no-learning clearance-only也选择1 event，AUROC=`0.91644`。因此：

- learned directional score相对纯clearance有`+0.04383` AUROC的全排序增量；
- fixed-50% event count没有显示learned uncertainty超过geometry；
- P108仍然支持factorization相对Actor-only/P75，但不能把全部收益归因于学习到的uncertainty。

P112尝试用冻结P109 Gaussian做256-sample nonlinear Euclidean crossing。它在P81保持0 events，却在P96退化到3 events且
AUROC从linearized的`0.90434`降至`0.85852`，因此拒绝并关闭sample-count/full-covariance/distribution sweep。当前证据支持
与decision boundary对齐的linear projection，不支持将finite-sample nonlinear Monte Carlo包装为更可靠的collision probability。

P114进一步检验task-relevant downstream aggregation：冻结P109，把每trajectory top-16 crossing probabilities和
independent-union proxy送入只有正权重的monotone pool。虽然79,478 source trajectories上的balanced BCE降至`.304205`，
P81 AUROC却从`.967639`降至`.951378`；P96从`.904345`降至`.902976`，fixed50 events还从0增到1。该结果说明
occupancy flip更接近局部boundary-tail事件，累积多个强相关time/Actor probability会稀释signal；保留P109 max并关闭聚合扫参。

### 2.5 P113独立uncertainty-vs-clearance确认

P113在任何target read前冻结新的10-scene、四location cohort：`0094/0331/0521/0003/0013/0038/0797/0920/
0926/1061`。P109 checkpoint、normalization、linear projection、0.05m floor、H3.5、time/Actor max和fixed50均不变。
唯一decision为：

```text
directional selected events <= clearance-only selected events
AND
directional AUROC - clearance-only AUROC >= 0.02
```

P113 outcome：`PENDING_FINAL_FILL`。

## 3. 核心结果表

| 阶段 | 数据角色 | query / Actor / P75 selected events | query / Actor AUROC | 结论 |
|---|---|---:|---:|---|
| P81 | independent all-row endpoint | `26 / 57 / 45` | `.94147 / .92947` | 窄全row triage支持 |
| P86 best | same-read visited-trajectory | `187 / 193 / 199` | 见canonical summary | task-conditioned增量不足 |
| P95 | consumed development occupancy flip | `7 / 28 / 13` | `.83952 / .69366` | development支持 |
| P96 | independent P95 | `8 / 5 / 12` | `.65542 / .71181` | relative反转 |
| P102 | consumed development hierarchical | `4 / 27 / 13` | `.87161 / .68922` | development best |
| P103 | independent P102 secondary | `9 / 7 / 12` | `.74385 / .67973` | relative反转 |
| P107/P81 | consumed scalar uncertainty | `2 / 36 / 13` | `.92901 / .56826` | development支持 |
| P107/P96 | consumed scalar uncertainty | `2 / 9 / 12` | `.87305 / .61786` | development支持 |
| P108 | independent scalar uncertainty | `5 / 35 / 20` | `.95107 / .77605` | primary支持 |
| P110 | same-read directional secondary | `1 / 53 / 20` | `.96027 / .69142` | secondary支持 |
| P111 | same-read clearance-only | `1 / n/a / n/a` | `.91644 / n/a` | 强geometry baseline |
| P114 | consumed downstream tail pool | P81/P96=`0 / 1` | `.95138 / .90298` | reject；均低于P109 max |
| P113 | independent directional vs clearance | `PENDING` | `PENDING` | `PENDING` |

## 4. 失败如何推动研究对象变化

| failure family | 被否定的假设 | 递进迁移 |
|---|---|---|
| `V67-F67` | visited Actor max-error可稳定产生τ-conditioned增益 | 改为真实occupancy-decision flip |
| `V67-F68--F73` | subtype、local auxiliary、层级结构或更多source可修复development shortcut | 保留独立拒绝，不扫参 |
| `V67-F74--F75` | end-to-end occupancy query classifier可独立迁移 | Actor distribution与candidate query解耦 |
| `V67-F76--F77` | launcher与NPZ交付工程事件 | 绝对入口与partial→atomic replace；不加校验门控 |
| `V67-F78` | nonlinear finite-sample crossing必优于linear boundary projection | 保留directional linearized score |
| `V67-F79` | monotone top-k/union tail pooling可提升P109 max | 终局拒绝；P113失败ID顺延F80 |

## 5. 系统与资源

- 全部训练与推理使用1x RTX 3090；没有多卡需求。
- P107/P109各训练916,722个Actor-time tokens、6,000 steps；P107 wall约30.01s，P109训练与P108归档I/O重叠。
- P108并行流式扫描7个shards、精确提取3,877/3,877 files并预处理10/10 scenes，wall约2,439.39s。
- P112 nonlinear试验只做固定256 samples、seed0的一次read，不做sample/seed/distribution sweep。
- P114在P113归档I/O期间完成6,000-step GPU训练，wall约13.66s，未读取P113 target。
- 未新增hash、checksum或fingerprint；没有smoke/regression matrix。

## 6. 有效性与claim边界

本报告支持：

- task-conditioned occupancy-decision reliability应将Actor uncertainty与candidate trajectory geometry因子化；
- scalar q90 tube相对Actor-only/P75在一个scene-level independent cohort成立；
- directional boundary projection在development和一个prospective secondary read上提供强排序；
- clearance-only是必须正视的强baseline，learned uncertainty的独立增量必须由P113单独裁决。

本报告不支持：

- session-level或population-level generalization；
- calibrated collision probability、conformal coverage或formal risk bound；
- planner、policy、control、closed-loop或deployment authority；
- safety improvement或false-safe guarantee；
- 用P81/P96等已消费cohort冒充新的独立确认。

## 7. 结论

V6.7最重要的递进结论不是某个更大的query network，而是预测对象和条件化位置的改变。让`τ`直接进入端到端classifier，会在
development中产生很强、但跨cohort不稳定的interaction shortcut；让网络只预测Actor uncertainty，再让`τ`通过明确的boundary
geometry进入解析计算，才获得稳定的scene-level independent evidence。与此同时，clearance-only baseline提醒我们不能把几何贡献
误写成learned uncertainty。P113的唯一职责就是关闭这一归因缺口；无论结果正负，都不再通过floor、cohort、模型或gate sweep补救。

## References

- Chai et al., [MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction](https://proceedings.mlr.press/v100/chai20a.html), CoRL 2019.
- Farid et al., [Task-Relevant Failure Detection for Trajectory Predictors in Autonomous Vehicles](https://proceedings.mlr.press/v205/farid23a/farid23a.pdf), CoRL 2022.
- Casas et al., [Implicit Latent Variable Model for Scene-Consistent Motion Forecasting](https://openaccess.thecvf.com/content/ICCV2023/html/Casas_Implicit_Latent_Variable_Model_for_Scene-Consistent_Motion_Forecasting_ICCV_2023_paper.html), ICCV 2023.
- [Open-source collision-probability estimation with stochastic boundary crossing](https://github.com/TUM-AVS/Collision-Probability-Estimation), 2025.

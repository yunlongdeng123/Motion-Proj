# HARP-Compiler：保留危险Actor的幻觉感知世界编译

## 从两级物理证书到Actor-preserving runtime package

- 版本：WorldSim V6.6
- 文档冻结：2026-08-28
- 分支：`research/worldsim-v6.6-harp-compiler`
- 研究终态：`v66_research_complete_arxiv_report_ready`
- 核心结论：local ranking/package与synthetic reactive capability支持；natural physical surface repair终局拒绝；RL未执行

## 摘要

驾驶WorldSim中的artifact处理存在一个危险捷径：删除低置信Actor虽然能降低表面冲突，却会同时删除cut-in、急刹与
near-miss等合法危险，从而把训练世界变得更容易。WorldSim V6.6研究能否将V6.5的可靠性信号编译成一个既抑制局部
幻觉、又严格保留Actor身份和危险属性的HARP compiler。我们首先在paired development atlas上验证factorized certificate
接口，再在natural actor-local conflict上发现coarse deterministic certificate recall为0。由此形成两级设计：deterministic
层保护Actor existence，低容量8维2x32 MLP只排序local geometry的REPAIR/ABSTAIN。该head在consumed legacy selection上
达到AUROC/AUPRC `0.6524/0.6924`，在scene-disjoint independent cohort上达到`0.7616/0.7672`。编译器随后将127个
Actors、581个states和162万owned primitives烘焙为不携带模型或hidden target的八文件runtime package。

固定预算下，learned ranking将未处理local-conflict exposure降低`68.40%`，但这仍是triage而非物理修复。两个
sensor-supported physical repair尝试揭示不可忽略的precision-yield frontier：exact hit使conflict下降`84.77%`，却只保留
`39.57%` clean boundary；one-voxel support把clean retention提高到`61.95%`，但conflict reduction降为`41.79%`。唯一恢复
失败后physical repair family终止，因此按预注册规则不执行matched RL。作为独立能力审计，一个固定jerk-bounded
reactive Actor在六个synthetic lead-brake场景把collision steps从306降到0并保持最小间距1.948m。最终可辩护贡献是
hazard-preserving certificate/package接口、local conflict ranking与明确的repair失败边界，而不是RL-ready simulator、
planner、policy或safety guarantee。

## 1. 问题定义

对Actor `i`，HARP显式分离两个变量：

```text
E_i: Actor existence / identity legitimacy
G_i,p: Actor-owned local primitive p 的geometry validity
```

这两个变量不能互相替代。稳定track、lifecycle和same-Actor sensor evidence可以强保护`E_i`，但不能保证Actor envelope中
每个primitive都与observed free-space一致。反过来，单个local geometry conflict也不能授权删除整个Actor。

Compiler输出的合法动作是：

```text
Actor existence: KEEP
Local geometry: KEEP / REPAIR / ABSTAIN(UNKNOWN)
```

Hazard attributes不参与existence gate。Cut-in、collision、负TTC或急刹本身不是artifact。所有方法比较必须保持Actor ID、
track、trajectory和hazard proxy，否则artifact下降可能只是easier-world collapse。

## 2. 研究流程与数据角色

V6.6刻意缩短了P0和回归验证，直接进入研究。由于没有新的42-scene fresh V6.6 cohort，正式结果严格区分角色：

| 阶段 | 数据角色 | 可解释范围 |
|---|---|---|
| P1-D/P2-D/P4-D | deterministic paired development | 接口与机制，不是natural generalization |
| P2N/P3L | consumed legacy diagnostic/selection | 发现natural conflict与选择local head |
| P3C | scene-disjoint consumed legacy confirmation | 独立ranking evidence，不是fresh V6.6 test |
| P6/P7/P7R/P7R2 | 同一P3C package与natural boundary | package、triage和physical-repair审计 |
| P8/P8R | P6 Actor trajectories + synthetic AV lead-brake | 狭窄reactive capability |

没有P9 matched RL、P10 confirmation或P11 exact test。Development injected results不会和natural repair结果合并成一个
“overall success”指标。

## 3. Factorized certificate与natural gap

### 3.1 Paired development capability

P1-D从409个base Actor-unit构造2,045 paired clusters与8,180 rows，覆盖validity/hazard四象限。冻结q0在paired
representation上为0.5 baseline；injection metadata不进入inference。P2-D从sensor/provenance、duplicate、lifecycle、
kinematic/identity与shape factors重算reason code，并在构造性数据上通过8/8 gates。P4-D进一步证明paired clean reference
可恢复observable factor，同时保持Actor ID/track/trajectory/hazard attributes。

这些满分只证明接口可以表达预定义corruption，不证明natural scene中存在同样可辨识的truth。

### 3.2 Natural actor-local conflict

P2N在另一批891个Actor-unit上，以Actor-owned native boundary与target observed-FREE冲突作为local geometry label：

| metric | q0 | deterministic certificate |
|---|---:|---:|
| AUROC | `0.543745` | `0.500000` |
| AUPRC | `0.612874` | `0.558923` |
| conflict recall | n/a | `0` |

498/891 Actor-unit包含local conflict，但仍具有Actor hit/current/swept support。结果否定了“existence evidence可以直接作为
所有local primitive validity”的假设，并触发`V66-F01`。

## 4. Two-level Actor certificate

受instance-centric occupancy与evidence-theory工作的启发，P3L不替换backbone，而只聚合八个Actor-local observable：

```text
q0 mean / p90
log boundary / hit / current / swept counts
hit-to-current density
current-to-swept ratio
```

模型固定为2x32 ReLU MLP、seed0、single full-batch weighted BCE，无architecture/seed/threshold sweep。输出
`p_local_conflict`只排序local REPAIR/ABSTAIN，不能控制Actor existence。

| cohort | rows | AUROC | AUPRC | vs deterministic AUROC/AUPRC |
|---|---:|---:|---:|---:|
| P3L selection | 891 | `0.652365` | `0.692384` | `+0.152365/+0.133461` |
| P3C independent | 581 | `0.761644` | `0.767165` | `+0.261644/+0.238766` |

P3C在6/6 evaluable scenes超过chance，且没有refit normalization/model或调整threshold。`V66-F01`因此按“两级
certificate”恢复：deterministic层保护existence，learned层只处理local geometry uncertainty。

## 5. Actor-preserving HARP package

P6将offline compiler输出烘焙为八文件package：static tri-state RLE、Actor metadata、owned primitives、artifact factors、
repair log、hazard attributes、provenance和runtime manifest。结果为：

| metric | value |
|---|---:|
| units / unique Actors / Actor states | `72 / 127 / 581` |
| Actor-owned primitives | `1,623,503` |
| package files / bytes | `8 / 16,321,358` |
| Actor state retention | `1.0` |
| metadata completeness | `1.0` |
| removed Actors / hidden-target fields | `0 / 0` |

Runtime不加载learned model，hazard不控制Actor existence。这个结果证明可部署数据合同，而不是physical geometry已经被修复。

## 6. Fixed-budget triage

P7在581个Actor states上固定50% action budget，即290个local actions。N0/Q0/D0/L0/O0各臂拥有相同Actor和hazard
attributes，只改变repair/abstain候选排序。

| arm | handled conflicts | exposure reduction | scene yield |
|---|---:|---:|---:|
| N0/D0 | 0 | `0` | n/a |
| q0 | 193 | `0.628664` | `5/6` |
| learned local head | 210 | `0.684039` | `6/6` |
| oracle | 290 | `0.944625` | n/a |

L0同时保持Actor retention=1、removed=0、hazard shift=0和emitted local geometry fraction=`0.500861`。然而
`physical_geometry_mutated=false`，handled表示候选覆盖而非真实修复。`V66-F02`记录这一估计量边界。

## 7. Natural physical surface repair的终局负结果

### 7.1 Exact sensor support

P7R在同一L0 action set上只保留映射到same-Actor motion-compensated exact hit的boundary primitive，其余转UNKNOWN。
Actor canonical shell/ID/track/trajectory/hazard不变，target只用于post-repair评估。

| metric | result | gate |
|---|---:|---|
| conflict reduction | `0.847660` | PASS |
| overall boundary retention | `0.383588` | FAIL (`>=0.40`) |
| clean boundary retention | `0.395715` | FAIL (`>=0.40`) |

7/9 gates通过仍拒绝。Exact evidence support足够精确，但过稀，生成大量UNKNOWN并损失合法Actor surface。

### 7.2 One-native-voxel support

唯一恢复P7R2将support固定为`0.512m`，即一个native voxel side；action set与九个gates不变：

| metric | exact P7R | radius P7R2 | P7R2 gate |
|---|---:|---:|---|
| conflict reduction | `0.847660` | `0.417872` | FAIL |
| overall retention | `0.383588` | `0.617684` | PASS |
| clean retention | `0.395715` | `0.619549` | PASS |

邻域支持解决yield，却同时保留684/1,175 conflict points。这个结果显示在当前evidence representation下，仅用sensor
proximity不能区分合法表面支持与Actor-local free-space conflict。按照冻结规则，不试中间radius、不降低gate、不更换budget、
不在缺少complete-shape GT时训练completion model。`V66-F02`以`closed_negative_after_single_recovery`终止。

## 8. Independent reactive Actor capability

P8不用于恢复P7，而只测试保留Actor能否接收独立AV action。六场景各固定选择一个有>=6 samples且median speed最高的
Actor；AV从12m bumper headway开始，在3s以2m/s²制动。X0以冻结初速继续，X1沿logged polyline及显式terminal tangent
extension执行同一IDM-style response：reaction latency 0.5s、deceleration<=3m/s²、acceleration<=2m/s²、jerk<=6m/s³。

初次P8将碰撞步数降为0，但两个低速场景因stop边界重复更新acceleration而使jerk达到9.64/7.40，仅4/6通过。参考
Autoware STOPPING/STOPPED与command jerk limiter后，唯一P8R只把stopped desired acceleration设为0，并让每步通过原
rate limiter一次。所有实验参数与gate exact。

| metric | P8R result |
|---|---:|
| selected/supported scenes | `6/6` |
| X0/X1 collision steps | `306/0` |
| minimum X1 bumper gap | `1.948192m` |
| maximum command jerk | `6.000000m/s³` |
| response latency | `0.5s` |

该结果是synthetic lead-brake controller capability，不是natural multi-agent behavior validation。两个低速Actor的logged path
只有约0.61–0.66m，长时rollout依赖明确披露的terminal-tangent extension，因此尤其不能作为真实闭环世界证据。

## 9. 为什么没有执行RL

V6.6 plan将P7定义为RL-ready physical replay candidate的前置，并明确规定：

```text
P7 FAIL -> 不进入 RL
```

P7 triage通过不等于physical repair通过；P7R/P7R2终局拒绝。因此即使P8R展示了reactive capability，也不能构造
matched RL3/RL5 arms，更不能将未修复world中的策略结果归因于HARP。P9、P10和P11没有运行。这是protocol compliance，
不是资源不足；单张RTX 3090足以完成所有已解锁阶段。

## 10. 系统与资源结果

- GPU：1x RTX 3090 24GB；multi-GPU未需要。
- P1/P2N/P3L/P3C observed peak GPU allocation约`0.02359GiB`；GPU计算均为短forward/training，主要成本是I/O/CPU。
- 最大记录RSS来自P3L，约`1.08285GiB`。
- P6 package为16.32MB；P8R trajectory JSONL约5.20MB。
- closeout时`/root/autodl-tmp`约剩95GB。
- 未新增hash/checksum/fingerprint；没有smoke/regression matrix；formal recovery均只执行一次。

## 11. 失败与可复现边界

| failure | 结论 | 后续约束 |
|---|---|---|
| `V66-F01` | coarse existence certificate不覆盖natural local conflict | 用two-level certificate解决，不删除Actor |
| `V66-F02` | triage无法迁移为同时满足precision/yield的physical repair | terminal；禁止第二radius/model/budget/gate recovery |
| `V66-F03` | stop-state duplicate jerk update | single numerical recovery；参数/gates不变 |

Source-of-truth顺序是canonical run、`EXPERIMENTS.md`、`RESEARCH_FAILURES.md`、`RESEARCH_STATUS.md`/terminal state、
最后才是本报告。Report handoff只读解析24个V6.6 `summary.json/status.json`，不重算metric。

## 12. Limitations

本研究不支持：

- fresh V6.6或population-level generalization；
- 完整SceneIR、RGB、appearance或complete-shape repair；
- natural off-log multi-agent response；
- physical collision probability或safety guarantee；
- planner、policy、control、closed-loop、matched RL或deployment authority；
- P9/P10/P11或exact-test结果；
- “HARP已成为RL-ready simulator”的表述。

## 13. 结论

WorldSim V6.6证明Actor hallucination处理不能退化为“低置信就删Actor”。Existence、local geometry和hazard必须分层；
在这一合同下，low-capacity actor-local head能提供可迁移的local conflict ranking，HARP package能完整保留Actor与hazard
metadata，固定响应器也能在窄synthetic干预中保持bounded dynamics。然而，natural physical repair暴露了sensor support的
precision-yield frontier，并在唯一恢复后终局失败。因此V6.6最重要的结果既包含正接口，也包含负边界：我们得到一个
hazard-preserving compiler prototype和可审计的failure chain，但没有得到可用于matched RL的物理修复world。

## References

- Jiang et al., [Symphonies](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_Symphonize_3D_Semantic_Scene_Completion_with_Contextual_Instance_Queries_CVPR_2024_paper.html), CVPR 2024.
- Ma et al., [Cam4DOcc](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Cam4DOcc_Benchmark_for_Camera-Only_4D_Occupancy_Forecasting_in_Autonomous_Driving_CVPR_2024_paper.html), CVPR 2024.
- Tonderski et al., [NeuRAD](https://openaccess.thecvf.com/content/CVPR2024/html/Tonderski_NeuRAD_Neural_Rendering_for_Autonomous_Driving_CVPR_2024_paper.html), CVPR 2024.
- Ost et al., [Neural Scene Graphs](https://openaccess.thecvf.com/content/CVPR2021/papers/Ost_Neural_Scene_Graphs_for_Dynamic_Scenes_CVPR_2021_paper.pdf), CVPR 2021.
- Yu et al., [PoinTr](https://openaccess.thecvf.com/content/ICCV2021/html/Yu_PoinTr_Diverse_Point_Cloud_Completion_With_Geometry-Aware_Transformers_ICCV_2021_paper.html), ICCV 2021 Oral.
- Yang et al., [UniSim](https://waabi.ai/unisim/), CVPR 2023.
- Zhou et al., [SMARTS](https://proceedings.mlr.press/v155/zhou21a.html), CoRL 2020.
- Gulino et al., [Waymax](https://arxiv.org/abs/2310.08710), NeurIPS 2023 Datasets and Benchmarks.
- [Autoware PID longitudinal controller](https://autowarefoundation.github.io/autoware_universe/latest/control/autoware_pid_longitudinal_controller/).

# WorldSim V6.6 research closeout

- 日期：2026-08-28
- 分支：`research/worldsim-v6.6-harp-compiler`
- 终态：两级Actor certificate、HARP package与synthetic reactive capability支持；natural physical surface repair终局拒绝；P9/RL未解锁

正式报告入口：

- [`V66_ARXIV_TECHNICAL_REPORT.md`](V66_ARXIV_TECHNICAL_REPORT.md)
- [`ARXIV_EVIDENCE_INDEX.md`](ARXIV_EVIDENCE_INDEX.md)

## 研究问题与最终答案

V6.6试图把V6.5的visited-state reliability evaluator编译成既去除world artifact、又保留合法危险Actor的HARP world。
结果支持三个较窄接口：

1. Actor existence与Actor-local geometry validity必须拆成两级certificate；
2. 冻结的low-capacity local geometry head能在独立legacy cohort排序local conflict；
3. Actor-preserving package能执行固定synthetic lead-brake bounded response。

但“排序/triage可用”没有转化为natural physical repair。Exact sensor support满足冲突下降却损失过多clean geometry；
one-voxel support恢复geometry retention却保留过多conflict。唯一恢复失败后P7 family关闭。按plan的`P7 FAIL → 不进入 RL`
规则，P9、P10与P11不执行。

## Canonical evidence chain

| 阶段 | 结果 | 论文角色 |
|---|---|---|
| P1-D/P2-D/P4-D | deterministic paired development通过 | 机制与接口可行性，不是natural/fresh质量结论 |
| P2N | deterministic certificate对natural local conflict recall=0 | 暴露existence与local geometry不可混同 |
| P3L | selection AUROC/AUPRC=`0.652365/0.692384` | consumed legacy selection |
| P3C | independent AUROC/AUPRC=`0.761644/0.767165`，6/6 scenes | 独立legacy ranking confirmation |
| P6 | 127 Actors、581 states、1.62M primitives，Actor retention=1 | Actor-preserving runtime package capability |
| P7 | fixed-budget exposure reduction=`0.684039` | triage正结果，不是physical repair |
| P7R | conflict reduction=`0.847660`，clean retention=`0.395715` FAIL | exact sensor support过稀 |
| P7R2 | clean retention=`0.619549`，conflict reduction=`0.417872` FAIL | proximity support precision-yield tradeoff；terminal negative |
| P8 | 4/6；两个低速scene jerk超限 | implementation negative |
| P8R | 6/6，collision steps `306→0`，jerk<=6 | synthetic bounded-response capability |

## ArXiv可写结论

- existence protection与local geometry ranking分层，避免把局部冲突升级为whole-Actor deletion；
- frozen actor-local evidence head在两批scene-disjoint consumed legacy cohorts超过constant和q0 baselines；
- HARP runtime package保留Actor identity/lifecycle/trajectory与hazard attributes，不携带hidden target或runtime model；
- fixed-budget learned ranking降低未处理local-conflict exposure，同时不删除Actor、不改变hazard proxy；
- natural surface repair呈现明确tradeoff：更严格sensor support牺牲clean geometry，更宽support牺牲conflict removal；
- 一个固定、bounded、identity-preserving响应器在六个synthetic lead-brake场景通过collision/gap/kinematic gates。

## 禁止结论

- 不得称P7 triage为physical repair或artifact rate reduction；
- 不得称HARP package为RL-ready world；
- 不得声称fresh V6.6 generalization、完整SceneIR/RGB repair、natural interaction或population guarantee；
- 不得声称planner、policy、closed-loop RL、collision probability或safety；
- 不得把P8R synthetic collision avoidance扩展为真实行为模型；
- 不得执行或暗示P9/P10/P11结果。

## Terminal rule

P7R2是surface-repair family的唯一预冻结恢复，不能再以中间radius、completion model、gate relaxation或新budget复开。
P8R只修复numerical duplicate jerk update，不能恢复P7。任何physical repair、fresh distribution或matched RL研究必须以
新版本、新protocol、新分支和未读cohort开始。

Closeout只读解析V6.6 run tree中的24个`summary.json/status.json`，全部有效；没有重算metric或执行额外测试。

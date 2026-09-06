# WorldSim V7.1 M40 — Joint Categorical Evidential Authority

日期：2026-09-05  
状态：frozen

## 问题

M39证明冻结M38 authority在direct categorical surface-return composition下三项判定通过，但其anchor/child heads
分别由旧transmittance目标训练，仍有训练—部署不一致。M40将M39部署分布本身作为GT first-return proper loss；
这不是post-hoc composition，因为所有非GT bins（包括首回波前FREE bins）在训练中直接竞争唯一GT bin。

## 冻结设计

- 初始化M35 anchor head与M38 child head；冻结M8 center/scale/trajectory和producer features；
- 同一weighted Gaussian energy在32-bin训练、64-bin部署中直接softmax为categorical return distribution；
- native LiDAR GT first-return one-hot交叉熵 + expected-depth L1；同时保留anchor与child held-out F/O/U soft
  evidential CE，防止ray loss退化成无法解释的global reweighting；
- 单seed71140、4 epochs；两个evidential权重均0.5，depth权重0.1；不调geometry/scale/bin/median/seed/loss；
- 不使用UNKNOWN mask、threshold、deletion、filter或post-result arm selection。

## 判定

同一593 train / 66 exposed holdout，相对原unit-energy baseline要求all与hazard/clear early不增、all hit最多降1pp，
且anchor/child predicted-vs-GT occupied correlation都不低于0.25。另报告相对M39 frozen candidate变化。失败登记
`V71-F40`并关闭当前joint authority fine-tune，不作loss/epoch恢复；无external/M21 partial read。

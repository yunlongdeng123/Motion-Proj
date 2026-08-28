# P3C Independent Local Geometry Confirmation Freeze

Task：`WS-V66-P3C-INDEPENDENT-LOCAL-GEOMETRY-CONFIRM-01`

确认cohort固定为V6.5 P2V已消费的`scene-0001/0219/0402/0594/0822/1110`，72 units；它与P3L训练的
P10V、选择的P10X均scene-disjoint。沿用既有P2V evidence/native产物，不重跑3.3GB native inference。

确认严格加载P3L canonical checkpoint及其中的normalization、8维feature contract和2x32结构。禁止模型/
normalization refit，禁止feature、seed、architecture、threshold sweep，禁止第二confirmation。primary gates固定为：

- AUROC相对constant deterministic certificate至少`+0.03`；
- AUPRC相对prevalence baseline至少`+0.05`；
- 至少4个evaluable scene的AUROC高于0.5；
- Actor existence authority保持关闭。

本次只支持independent consumed-legacy local geometry ranking，不是fresh V6.6 generalization；不授权Actor删除、真实
geometry repair、planner、policy、closed-loop、RL或safety。

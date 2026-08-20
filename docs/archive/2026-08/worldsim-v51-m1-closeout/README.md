# WorldSim V5.1 M1 收尾归档

- 收尾日期：2026-08-20
- 终态：`V5.1 M1 closed_without_promoted_candidate`
- authoring base commit：`fc07b9912db4df4a73b44d7a2363d9b3c6c88bc2`
- baseline：`U2/B3` 保留为 V5.2 comparator，不再作为 V5.1 待扩展路线
- Stage H：task status=`pending`；execution=`false`；disposition=`superseded_by_v5.2_scope`
- 冻结配置：`configs/worldsim_v51/m1_closeout_v1.yaml`
- 统一失败账本：`docs/RESEARCH_FAILURES.md`，重点 `V51-F31/F37/F42/F63/F65/F66`

## 1. 收尾结论

V5.1 M1 的核心瓶颈不是“空间拓扑还不够复杂”，而是**有效观测结构性缺失**。已经运行的后续路线分别改变
feature uplift、传播规则、node 粒度、跨视角 identity 或 reverse tracing，但没有建立稳定、充分且可复现的新证据源。
因此 V5.1 不再继续沿同一证据源做 completion；V5.2 必须优先回答“证据从哪里来、覆盖多少、能否跨视角持续、能否
确定性重放”。

这是一项受证据约束的路线收口，不是对所有 graph、Gaussian Grouping、Trace3D 或 BKI 变体的普遍否定。V5.1 没有运行
BKI，也没有读取 validation、test 或 KITTI quality 来做方法选择。

## 2. 关键失败证据

| 路线 | 观察事实 | 结论与边界 | Canonical evidence |
|---|---|---|---|
| LUDVIG uplift/raw graph | 0471/0379 的 B1 actor-background margin=`-0.121280/-0.098618`，scene-balanced=`-0.109949`；heldout reprojection 改善仍不能救回 actor separation | uplift/raw graph rejected；不能用 reconstruction proxy 代替 actor evidence | r015，`V51-F31` |
| Progressive propagation | scene-balanced ΔBoundary-F1=`+0.0002196`，但 ΔIoU=`-0.0714543`、ΔFN semantic mass=`+0.1694766` | 微弱边界收益伴随显著漏检与 IoU 退化；D0 rejected、D1 skipped | r018，`V51-F37` |
| Simple voxel super-primitive | observation density 确实提高；相对 U2/B3 的 ΔBoundary-F1/ΔIoU/ΔFN=`-0.0002566/-0.0925468/+0.1899473`；相对 raw D0=`-0.0004762/-0.0210926/+0.0204707` | “更密的 observation structure”不等于“更好的 semantic evidence”；E0B rejected | r020–r022，`V51-F39/F42` |
| Gaussian Grouping | scene pass=`[FAIL,PASS,FAIL]`；0471/0379 identity recall=`0.080747/0.202933`，persistent-track fraction 均为 `0` | 45-view 执行稳定不等于 identity input 可用；当前三场 adapter 下 rejected，不外推普遍失效 | r041–r043，`V51-F62/F63` |
| Trace3D faithful operator | 8 个 fresh processes、16 次 alpha 调用出现 `0.0056084292/0.0267562941` 两个 exact 值；16 次 hard vector 均为 `[0,1]` | hard count 正确不能替代贡献值可复现；exact unpatched operator rejected，普通 `+=` 只是 hazard、不是已证唯一根因 | r046–r047，audit `98c72ba7...d31`，`V51-F65` |

## 3. Stage H / BKI 的处置

Stage H 保持 task status=`pending`，但被 V5.2 scope 取代，本版本不执行。原因是 BKI 虽不是显式图扩散，仍然执行：

```text
已有局部证据 → 空间相关性传播 → 未观测节点补全
```

它改变“怎样传播”，没有改变“证据从哪里来”。结合 r018、r022、r043、r047，继续做 BKI 的预期边际收益低，且会让
V5.1 再次围绕同一缺失证据做传播器替换。这里的 `low_expected_gain` 是基于累计负证据的研究决策，不是 BKI 的实证
reject；如未来复开，必须先由 V5.2 提供新的独立观测源，并在 method quality 之前冻结 coverage、identity persistence、
fresh-process reproducibility 与跨场分母。

## 4. V5.2 必须继承的门禁

1. 保留 `U2/B3` matched comparator，不从 V5.1 rejected route 选择性摘录正指标。
2. 先证明新增证据源，不把 propagated completion、metadata、binary actor union 或 evaluation proxy 冒充 observation。
3. 先做 train/development-only 的 coverage、persistent identity 与 determinism gate，再决定是否重开传播或 completion。
4. blocked、rejected、恢复与推翻项继续只写入 `docs/RESEARCH_FAILURES.md`；不得新建 V5.2 专属 failure 账本。
5. validation/test/KITTI 保持冻结，直到 V5.2 development candidate 在预注册门上成立。

## 5. 轻量证据入口

- Stage G rejection freeze：`configs/worldsim_v51/stage_g_trace3d_faithful_operator_rejection_freeze_v1.yaml`
- V5.1 closeout freeze：`configs/worldsim_v51/m1_closeout_v1.yaml`
- 当前状态：`docs/RESEARCH_STATUS.md`
- 实验事实：`docs/EXPERIMENTS.md`
- 失败事实：`docs/RESEARCH_FAILURES.md`
- 清理说明：`CLEANUP_MANIFEST.md`
- 逐目标清单：`cleanup_plan.json`；执行结果：`cleanup_result.json`

大型 run、checkpoint、数据集和第三方源码未复制进 Git，也未在本次清理中删除；其 canonical 路径与 hash 继续由各 run
manifest、freeze 和上述统一事实源管理。

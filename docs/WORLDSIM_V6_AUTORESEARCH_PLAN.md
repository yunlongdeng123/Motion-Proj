# WorldSim V6 Autoresearch 运行计划

## 1. 权威来源与边界

- 规范计划：`docs/WORLDSIM_V6_VERIFIABLE_WORLD_COMPILER_AUTORESEARCH_PLAN.md`
- 规范计划提交态 SHA-256：`fa1510783c766630920c61c67ef2a8b736049d6628e3a8c6522d0ce113793dfe`
- 当前路线：`WorldSim V6 / Verifiable World Compiler`
- 当前分支：`research/worldsim-v6-world-compiler`
- 状态权威：`docs/autoresearch/worldsim_v6/AUTORESEARCH_STATE.json`
- 失败权威：`docs/RESEARCH_FAILURES.md`

本文件把规范计划转成可续跑控制面，不修改规范计划的研究问题、证据等级、锁定测试、资源纪律或停止规则。
本地绝对路径、凭据和机器专属环境只能进入 Git 忽略的 `.local/worldsim_v6/`。

## 2. 已完成治理前置

| Gate | 状态 | 证据 |
|---|---|---|
| G0 repo 收敛与恢复 | done | `docs/autoresearch/worldsim_v6/governance/REPO_PREFLIGHT.json`、`PRE_V6_WORKTREE_RECOVERY.json` |
| G1 文档收口 | done | canonical docs commit `9bdabb3e249ce5a048a5a9b7b0ba8dc4774b3bb2` |
| G2 分支收敛与 main 更新 | done | `BRANCH_MERGE_MATRIX.json`、`MERGE_CONFLICT_RESOLUTION.jsonl`、`G2_MAIN_UPDATE.json` |
| G3 新 V6 分支 | done | `V6_BRANCH_BOOTSTRAP.json` |

G2 有效测试按运行时拆分：主环境 `1443 passed / 1 skipped`，冻结 DriveStudio 环境 `15 passed`，合计
`1458 passed / 1 skipped`。已知工程失败及其关闭证据为 `V6-F01`–`V6-F04`；历史失败仍由失败账本约束。

## 3. 当前执行序列

1. `R1`：前端 capability audit。冻结硬件、磁盘、Python/CUDA 环境、数据、checkpoint、third-party checkout
   和必要 smoke；候选分类为 `runnable / repairable / audit_only / unavailable`。
2. `R2`：实现 SceneIR v0 及 schema/round-trip/negative tests；不以任一渲染器内部对象作为规范语义。
3. `R3`：实现 support deviation 与 training-only support certificate，冻结 denominator、版本和失败语义。
4. `R4`：实现确定性编译运行时、provenance、artifact manifest、hash/replay 与负向验证。

R1 完成前不注册算法假设；R2–R4 控制面通过前不启动方法训练。任何实验失败或 rejection 都先追加失败账本与
反思记录，再选择最小可验证修复或新的正交方向。单卡 RTX 3090 资源不足、所需私有资产不可获得或外部权限缺失
才构成向用户申请资源的硬阻塞。

## 4. 循环与锁

- 每轮只允许一个 `active_hypothesis`；dev 与 confirmation 分离，confirmation 默认锁定。
- 先写 hypothesis，再实现/运行；先写 terminal 与 artifact hashes，再更新文档结论。
- 无失败记录、无运行 manifest、无 deterministic replay 的结果不得提升为论文主张。
- 不把环境恢复、下载成功或 smoke 通过解释为算法质量增益。

# nuScenes cut-in 路线封存说明

- 封存日期：2026-07-26
- 路线状态：`rejected / frozen`
- 最终决策：停止继续调参、停止扩大事件挖掘；cut-in 仅保留为以后可选的演示场景，不再承担数据入口、方法定义或论文成立条件。
- 权威失败记录：[`../../../RESEARCH_FAILURES.md`](../../../RESEARCH_FAILURES.md)
- 后续路线历史入口：[`../dynamic-reconstruction-v1/DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`](../dynamic-reconstruction-v1/DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md)

本目录不再提供当前执行入口；当前授权只看 [`../../../RESEARCH_STATUS.md`](../../../RESEARCH_STATUS.md)。

## 为什么封存

1. nuScenes 没有公开可直接充当事件级真值的 cut-in 标签或总体占比，因而无法建立可信的召回率分母。
2. 四轮规则、地图、接收车与人工复核迭代后，最终前瞻式稀疏审核只有 `1 PASS / 1 scene`，不能形成可用事件池。
3. 继续投入的主要产出会是事件挖掘、地图匹配、接收车关系和审核系统，而不是动态驾驶场景重建与反事实编辑方法。
4. 失败并不证明 nuScenes 中没有 cut-in；它证明的是当前可验证路径不足以把 cut-in 作为研究主线的前置条件。

## 保留内容

本目录完整保留以下历史证据，禁止把它们重新解释为正结果：

- 资产与事件预检；
- mini、全域、运动学、接收车约束和最终全量筛选报告；
- 各轮预注册协议；
- 人工审核提示词与最终稀疏审核包；
- 最终迭代计划、资源拒绝记录和终止结论；
- 从 OccGS 可行性研究转向事件优先路线时的研究备忘。

## 文档索引

- `N0_ASSET_AND_EVENT_PREFLIGHT.md`
- `N1_MINI_EVENT_POOL_REPORT.md`
- `N1_FULLDOMAIN_EVENT_POOL_REPORT.md`
- `N1_KINEMATIC_PREREGISTRATION.md`
- `N1_KINEMATIC_EVENT_POOL_REPORT.md`
- `N1_KINEMATIC_HUMAN_REVIEW_PROMPT.md`
- `N1_RECEIVER_CUTIN_PREREGISTRATION.md`
- `N1_RECEIVER_CUTIN_EVENT_POOL_REPORT.md`
- `N1_RECEIVER_CUTIN_HUMAN_REVIEW_PROMPT.md`
- `N1_CUTIN_FINAL_BASELINE.md`
- `N1_CUTIN_FINAL_REPORT.md`
- `N1_CUTIN_FINAL_SPARSE_HUMAN_REVIEW.md`
- `NUSCENES_CUTIN_FINAL_ITERATION_PLAN_V1.md`
- `POST_OCCGS_RESEARCH_DIRECTIONS.md`
- `RESEARCH_STATUS_CUTIN_FINAL_SNAPSHOT.md`
- `EXPERIMENTS_CUTIN_FINAL_SNAPSHOT.md`
- `n1-cutin-final-resource-rejection-human-review/`
- `CLEANUP_MANIFEST.md`

## 恢复

归档前的完整 `docs/` 快照位于：

```text
/root/autodl-tmp/motion_proj_backups/docs-before-direction-pivot-2026-07-26/
```

该快照只用于恢复，不是新的研究入口。

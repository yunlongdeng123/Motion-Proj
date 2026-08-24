# WorldSim V6.2 autoresearch

当前路线：CPSC（Constraint-Aware Physical State Completion）。

- `SCOPE_FREEZE.md`：P0 科学范围、用户约束与资源边界。
- `P1_NOVELTY_AUDIT.md`：P1 一手来源组件/组合新颖性审计与迁移边界。
- `P3_FEASIBILITY_PROJECTION.md`：P3 最小硬投影实现、真实 fixture 与裁决。
- `P2_COHORT_FREEZE.md`：P2 metadata-only 六场景、target 与 sweep 角色冻结。
- `P2_QUERY_PROBE.md`：P2 单 unit 资源、类别分母与 label schema 探针。
- `P2_ACTOR_SWEEP_RECOVERY.md`：P2 formal r1 当前 ROI actor 空池、文献迁移与定点恢复。
- `P2_EVIDENCE_QUERY_DATASET.md`：P2 六场景、72-unit、7.2M query 正式数据集收口。
- `P4_IRWM_SIDECAR_INTERFACE.md`：P4 frozen backend、query-aligned logits/latent schema 与单 probe 计划。
- `P4_IRWM_SIDECAR_PROBE.md`：P4 新场景单 target 真模型 probe、资源与信息边界。
- `P4_IRWM_PRIOR_SIDECARS.md`：P4 六场景、72-target frozen IR-WM sidecar 正式收口。
- `P5_CPSC_LITE_DESIGN.md`：P5 scene-disjoint split、feature boundary、model/loss与单次capacity probe冻结。
- `P5_CPSC_LITE_CAPACITY_PROBE.md`：P5 真实loader/forward/backward/projection资源探针与formal决策。
- `P5_CPSC_LITE_FORMAL.md`：P5 48/24-unit正式训练、projection-only对照、资源与P6决策。
- `AUTORESEARCH_STATE.json`：唯一机器可读当前状态。
- `HYPOTHESES.jsonl`：预注册假设与状态。
- `REFLECTIONS.jsonl`：每轮观察、诊断、决策与去向。

可复用失败继续只写 `docs/RESEARCH_FAILURES.md`，不在本目录创建第二本 failure ledger。

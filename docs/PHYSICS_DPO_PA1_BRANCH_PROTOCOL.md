# PA1-BRANCH-02 v1：common-prefix sibling pilot

## 已冻结输入

- PA0 review decision：`d6183ae8021a0873d49c1447ecfa5d5e012e05489a10738a81d12fb014747d00`
- scene split：`e525edf33bcfec169c0077d2eb2e528d953dbc9930e771c803c889a32983c73a`
- PA1 horizon v3 profile：`70f7591b1599c5d4df7cc56a92fad66747e437dc1af331dcce5b71816563adfd`
- horizon decision：14 frames；peak 5.8049 GB；相对 8-frame generation slowdown 1.5799×。
- formal config：`configs/diagnostics/physics_dpo_branch.yaml`，fingerprint `a4fdfbbb6d44d3d810f89b47dbd1e32591faffc469c3f639e507852a01f9c302`。

本 protocol 只构造候选；不使用 future GT、P1 projected/hybrid target、训练 cache、LoRA update、DPO/AWR 或双卡。

## v1 的最小离散校准

```text
family = common_prefix
num_frames = 14
denoising steps = 25
fork fraction = 0.6  ->  fork_step = 15 shared scheduler transitions
strength = small = rho 0.01 × sigma_fork
conditions = preference_dev 的前 4 个 start_index=0、不同 scene clip
siblings = two antithetic groups × {positive, negative}
```

每个 condition 先由 `svd_official_v1` 生成 Base guard，并执行 full-trace exact rerun。随后以该 official trace 的第 15 个 post-step latent 为 prefix；手动 suffix 必须重新得到同一条 Base full trace，才允许生成 sibling。两个方向由固定 seed 的零均值方向和其 permutation 构成，正负成对；理论 RMS 完全相等，实际 bf16 注入 RMS/均值另行记录并 gate。

为检验候选距离的上界，每个 condition 有一个不同 seed 的 diagnostic rollout。它只写入 `diagnostic_independent/`，从不进入 `candidate_manifest.jsonl`、pair、winner 或训练集。

## machine gate

每个 sibling 都必须满足 finite、first-frame fidelity、有效 CoTracker3 track、minimum median track length、track coverage、survival/dynamic-degree non-collapse 和 frozen-VAE future distance gate。后者必须严格位于 exact Base rerun floor 与 independent-seed diagnostic 距离之间。

Pair/group 还要求固定 CoTracker query grid 对齐、track correspondence、strata agreement 和 antithetic distance symmetry。CoTracker aggregate 只做 branch-sign balance diagnostic，不产生 preference label。machine pass 至少需要 3/4 condition 与 6/8 antithetic groups 通过。

## 人工门槛

machine pass 后生成 8 个随机列顺序的 `[Anchor | sibling | sibling]` 视频 panel。人工仅判断它们是否是同一场景布局/主体身份下的不同未来，不看自动 score 或 winner。要求完成 8 个 case、decisive `same_scene` 比率至少 87.5%，且 `different_composition`/`invalid` 为 0。

## 顺序动作

- 所有 sibling 均不可区分：仅用新 run ID 试 medium；
- structure mismatch：仅用新 run ID 将 fork 改为 0.8；
- 已通过 machine 与人审：冻结 family/fork/strength，才可进入 PA2-PAIR-03；
- common-prefix 与允许的 re-noise family 均失败：PA1 `rejected`，不退回 independent-seed 主线。

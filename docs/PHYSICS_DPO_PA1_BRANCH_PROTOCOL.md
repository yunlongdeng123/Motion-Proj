# PA1-BRANCH-02 v3：common-prefix sibling pilot

## 已冻结输入

- PA0 review decision：`d6183ae8021a0873d49c1447ecfa5d5e012e05489a10738a81d12fb014747d00`
- scene split：`e525edf33bcfec169c0077d2eb2e528d953dbc9930e771c803c889a32983c73a`
- PA1 horizon v3 profile：`70f7591b1599c5d4df7cc56a92fad66747e437dc1af331dcce5b71816563adfd`
- horizon decision：14 frames；peak 5.8049 GB；相对 8-frame generation slowdown 1.5799×。
- formal config：`configs/diagnostics/physics_dpo_branch_v3.yaml`（继承 v2 的全部研究参数，只改变 run ID 与 schema 的浮点容差实现）。

本 protocol 只构造候选；不使用 future GT、P1 projected/hybrid target、训练 cache、LoRA update、DPO/AWR 或双卡。

## v3 的最小离散校准

```text
family = common_prefix
num_frames = 14
denoising steps = 25
fork fraction = 0.6  ->  fork_step = 15 shared scheduler transitions
strength = small = rho 0.01 × sigma_fork
conditions = preference_dev 的前 4 个 start_index=0、不同 scene clip
siblings = two antithetic groups × {positive, negative}
```

每个 condition 先由 `svd_official_v1` 生成 Base guard，并执行 full-trace exact rerun。每条 sibling 随后用同一个 official `SVDBackbone.generate` wrapper、同一 condition/seed/25-step trajectory 重跑；在第 15 个 transition 完成后的官方 `callback_on_step_end` 内注入扰动。callback 保存 pre-injection latent，并逐项核验此前 condition noise、initial latent、scheduler/unet trace 与 Base exact 一致；boundary 的 post-step latent 用保存的 pre-injection 值比较。第 16 个 transition 是第一个允许不同的 scheduler input。该实现不手动恢复 scheduler state，也不拼接 suffix。

v1 的手动 scheduler continuation 在首个 condition 未能 exact 重构 official Base trace，已保留为失败工程证据；未写入候选、score、pair 或训练结论。v2 已用上述 callback 生成全部四个 condition，随后 schema 将 permutation 的 float32 RMS 归约尾差误判为四条 sibling 不等范数；因此未进入 score、pair、panel 或训练。v3 仅令 schema 使用与 generator 相同的 `1e-7` 相对数值容差，family、fork、strength、conditions、候选构造、阈值和审查规则均冻结不变。

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

# WorldSim V5 M1B：Boundary Residual Forensic

日期：2026-08-14

任务：`WS-V5-M1B-D0-BOUNDARY-RESIDUAL-FORENSICS-01`

## 结论

在三场景 graph replication 失败后，本审计只读 r038/r045/r046 的不可变 evaluation NPZ，判断剩余错误是否主要集中在冻结的 3px target boundary band。协议、阈值和通过门在读取 boundary 分布前以提交 `bb4ebb5ee57f3dfd86b7189cc08e71a77e19e00f` 固定。

六个 `scene × unary G0` 单元中，满足以下三个条件的单元为 `0/6`：

1. boundary classification error share `>=0.5`；
2. boundary semantic error mass share `>=0.5`；
3. boundary error enrichment 相对 boundary pixel fraction `>=2.0`。

六单元 mean boundary classification error share=`0.4020948014`，mean boundary semantic error mass share=`0.2483529331`，均低于 `0.5`。正式结论：

> `boundary_ambiguity_not_primary_semantic_split_remains_locked`

因此 `WS-V5-M1B-REVERSIBLE-SEMANTIC-SPLIT-01` 的条件没有成立，semantic split 不启动。结合 graph replication 的 `3/6` 方向支持，当前 M1 structured ownership 路线收口为 `rejected`；不得继续用 Transformer、split 或 graph 调参追逐 development 结果，也不读取 validation。

## 冻结定义

- 输入：r038/r045/r046 的 B1/B3 × G0/G3 evaluation NPZ；共 `8+1+3=12` 个视图。
- probability threshold：`0.5`。
- target boundary：`target & ~binary_erosion(target)`。
- boundary band：对 target boundary 做 `3` 次 SciPy default-connectivity binary dilation，与既有 Boundary F1 tolerance 一致。
- scene 内先合并像素/质量，再对六个 scene×unary G0 单元等权判断。
- 只审计冻结 artifacts；不读取 source image，不搜索阈值，不改 graph，不训练。
- 即使门通过也只允许冻结新的 M1B 协议，不自动授权 split；本次实际未通过。

## 六单元结果

| Scene / unary | Boundary pixel fraction | Boundary class-error share | Boundary semantic-error share | Enrichment | Boundary FP-mass share | Boundary FN-mass share | Primary |
|---|---:|---:|---:|---:|---:|---:|---|
| scene0471 / B1 | 0.048596 | 0.185957 | 0.173163 | 3.826600× | 0.073274 | 0.321295 | no |
| scene0471 / B3 | 0.048596 | 0.189794 | 0.168873 | 3.905568× | 0.071257 | 0.317824 | no |
| scene1087 / B1 | 0.022667 | 0.343784 | 0.289600 | 15.166960× | 0.258669 | 0.317643 | no |
| scene1087 / B3 | 0.022667 | 0.324520 | 0.238010 | 14.317059× | 0.175126 | 0.301845 | no |
| scene0379 / B1 | 0.002448 | 0.687876 | 0.359494 | 280.978168× | 0.154424 | 0.579152 | no |
| scene0379 / B3 | 0.002448 | 0.680637 | 0.260977 | 278.021185× | 0.095237 | 0.577127 | no |

所有单元的 boundary error 都相对极小的 boundary pixel denominator 高度富集，但“富集”不等于“主要残差”。scene0471/1087 的多数分类错误与 semantic error mass 仍在边界带外；scene0379 虽有约 68% 分类错误位于边界带，boundary semantic-error mass 仍只有 26%–36%。这说明仅把 boundary Gaussian 拆成 foreground/background child 不能解释或覆盖主要质量损失，远场漏检/稀疏伪标签仍是更大的风险。

## Gate 裁决

| Gate | 观测值 | 要求 | 结果 |
|---|---:|---:|---|
| boundary-primary cells | 0/6 | >=4/6 | fail |
| mean boundary class-error share | 0.4020948014 | >=0.5 | fail |
| mean boundary semantic-error share | 0.2483529331 | >=0.5 | fail |
| automatic semantic split unlock | false | 必须为 false | pass |

## Canonical run 与哈希

- run：`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1B-D0-BOUNDARY-RESIDUAL-FORENSICS-01/20260814T192757Z__m1b-d0-boundary-residual-forensics-s0-r001`
- source commit：`bb4ebb5ee57f3dfd86b7189cc08e71a77e19e00f`
- summary SHA：`ddecf415bd71fcf920b6fab5f38ee74edea93aa085b48a73947480c3c186c35d`
- status SHA：`65ea0db4a1714b668923bd927ee7664c89bce34e133d3df97f622dbd78371e84`
- fingerprint SHA：`1e5605c0056274f03fdb0b3653528d1a2bb1b8801c7d1b190e8e8390e11c2e2b`
- manifest SHA：`a22b382d6a02d5821c38bc4fa061946abd13ef2300c05ad7f47051ab7b0b88a7`
- boundary audit SHA：`fd128a3b19373473b97c07d7c8733c0640f526f338911e4247d8cba7bc515c35`
- resolved config SHA：`f63c164c989b8ae65a3f814dca720659adf9aa5676b1125f9dbbb92d657dd861`
- events SHA：`1da22caeea99b484fcb27db1177e6004f4206c7417594c4f0bdedfc1c1c16fb1`

机器可读副本：`docs/archive/2026-08/worldsim-v5-m1/M1B_R001_BOUNDARY_RESIDUAL_METADATA.json`。

## 下一步

M1 与 M1B 在 development 阶段停止。V5 按主计划转入 M2 geometry-feasible repair routing；M1 的负结果继续作为“稀疏 observation / far-field FN / topology 不稳定”的附录证据，不允许在后续 M2/M3 成功时被倒写为 M1 成功。

# WorldSim V6.4 Research-Family Closeout

- Date: `2026-08-27`
- Branch: `research/worldsim-v6.4-native-uq`
- Terminal state: `v64_research_complete_report_ready`
- New scientific execution in this closeout: none

## Terminal decision

V6.4 is complete and ready for technical-report writing. It supports a bounded uncertainty-native conditional compilation chain on one
RTX 3090, but it does not support collision authority, closed-loop planning, or a safety guarantee.

The strongest supported result is the frozen M1 route-aware compiler under a fixed route-opportunity denominator. On the untouched
96-case test, M1 preserved M0 mean coverage exactly (`0.474969689`) while reducing worst-10 fixed-denominator route-conflict CVaR from
`0.020725740` to `0.010821074` and pooled conflict density from `0.004944667` to `0.002001413`. M1 was lower/equal/higher than M0 in
`18/78/0` paired cases. This is an exact empirical cohort result only.

The terminal negative is P11. The bounded critic's preregistered selected-policy gates passed narrowly, but unsafe-action recall was
`0.02174/0/0.01087` for Real-only/naive/UNC-verified on 184 unsafe actions, so none of the critics was usable as collision authority.
Independent threshold calibration then failed on a separate evaluation cohort: the verified arm reached only `0.62044` unsafe recall
and still produced two selected-policy false-safes. A rows-only diagnostic showed both unsafe-prior shift and ranking degradation
(verified AUROC `0.71165 -> 0.56274`), closing threshold-only recovery and the P11 family.

## What the version established

1. Native voxel uncertainty had relative signal but weak absolute separation: U2 improved pooled AUROC over U0 by `0.08305`, while
   remaining near chance within each evaluation scene.
2. A supervised PCA-16 risk head improved ranking (`0.65812` pooled AUROC) but retained high FPR@95TPR (`0.86774`); the first
   case-calibration route therefore failed with no positive coverage.
3. A frozen full-native 273-dimensional MLP recovered selective calibration. At nominal coverage `0.4`, independent calibration had
   zero failures and exact-once confirmation had `1/96` failure at mean realized coverage `0.39994`.
4. A stratum-conditional map increased mean coverage from about `0.40` to `0.475` with zero exact-once failures on its independent
   96-case confirmation cohort.
5. The target-free state bake, sparse Gaussian adapter, and bounded route consumer were executable without reading a target renderer,
   collision ground truth, or a large neural world model.
6. The current M0 route-tail metric failed (`0.05171 > 0.05`). M1 later passed its frozen absolute route-tail gates, but selected-count
   denominators changed. The fixed-opportunity audit and untouched test support the relative M1 result only under the common eligible
   denominator; they do not rewrite the earlier current-M0 negative result.
7. P11 showed that route-level uncertainty filtering did not transfer into a calibrated actor-envelope collision critic. Large NWM/RL
   training remains unexecuted and unauthorized for this version.

## Version-level claim boundary

Supported:

- native uncertainty features can be turned into an independently calibrated, stratum-conditional selective state compiler;
- conditional compilation can increase emitted coverage under the reported independent case-level risk protocol;
- M1 reduces exact empirical route-hidden-FREE conflict density and worst-tail risk relative to M0 on one untouched fixed-opportunity
  cohort without changing total emitted coverage;
- the complete V6.4 chain is executable on one RTX 3090 with scene-ready producer/consumer scheduling.

Not supported:

- population-level risk bounds, significance claims, physical collision avoidance, policy improvement, closed-loop driving, deployment,
  or real-world safety;
- a calibrated collision critic or an uncertainty-verified advantage over naive augmentation;
- a claim that threshold calibration alone can repair P11, because both prior and unsafe-score ranking shifted;
- superiority of a large neural world model or RL policy, because neither was trained.

## Resource and execution closeout

All required V6.4 scientific work fit the single-RTX3090 contract. Native workers peaked at `4.1314 GiB`; P11/P11R used less than
`0.074 GiB` CUDA allocation. The main bottleneck was shared-disk archive preprocessing, addressed by restricted-shard extraction,
per-scene staging, canonical processed reuse, and ready-first CPU/GPU queues. The final untouched native stage completed all eight
scenes and 96 targets, with the last two ready-to-native waits at `0.0646/0.0625 s`. No multi-GPU restart is required.

No hash, checksum, fingerprint, broad smoke suite, or regression matrix was introduced. The final documentation milestone adds no
experiment and no failure ID.

## Frozen report entry points

- `docs/autoresearch/worldsim_v64/ARXIV_EVIDENCE_INDEX.md`: canonical report-writing index.
- `docs/RESEARCH_STATUS.md`: chronological state and decisions.
- `docs/EXPERIMENTS.md`: experiment registry and exact metrics.
- `docs/RESEARCH_FAILURES.md`: only active failure/negative-evidence ledger.
- `docs/autoresearch/worldsim_v64/AUTORESEARCH_STATE.current.json`: machine-readable terminal state.

Any further scientific execution must start a new version with a new hypothesis and unread evaluation denominator. V6.4 evidence must
not be reused for threshold, policy, action-lattice, model, or gate selection.

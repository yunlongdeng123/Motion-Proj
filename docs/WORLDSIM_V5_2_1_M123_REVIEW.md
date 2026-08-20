# WorldSim V5.2.1 M1/M2/M3 Review

## M1 — `M1_EVIDENCE_INSUFFICIENT_KEEP_PENDING`

新 base census 六场与 frozen M1 development/validation 场景 exact overlap=`[]`，不足预注册的 2 个独立场景，
Q1–Q4 因而全部 undefined。scene-1087 的 mean observed views=`0.028232`、zero-observation Gaussians=`904,933` 仍是直接历史证据，
但当前不能回答其 base RGB 是否同时失败；不得据此晋级 TrackBayes/TubeBayes。

## M2 — `M2_EVIDENCE_INSUFFICIENT`

保留 V4 `154 requests / 214 candidates`、`83 accepted / 71 abstain` 与 full-denominator geometry MAE `+3.3908096237 m` caveat，
以及 V5 geometry-first 无 absolute-safe candidate 的 rejection。没有 exact request→新 case mapping，故不改 router、不搜 threshold。

## M3 — `M3_EVIDENCE_INSUFFICIENT_KEEP_PENDING`

V4 temporal delta 的 validation/test confirmation 保持有效；V5 constraint-projection rejection 不倒写 V4。新 census 仅有 unwarped proxy，
不能命名 `B-TEMPORAL`，也无法证明 V4 delta 与真实 base temporal badcase 的 exact overlap，因此保持 pending。

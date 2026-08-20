# WorldSim V5.2.1 M1/M2/M3 Review

## M1 — `DIRECTION_SUPPORTED_CAUSAL_BRIDGE_PENDING`

新 base census 六场与 frozen M1 development/validation 场景 exact overlap=`[]`，不足预注册的 2 个独立场景，
Q1–Q4 因而全部 undefined。scene-1087 的 mean observed views=`0.028232`、zero-observation Gaussians=`904,933` 仍是直接历史证据，
但当前不能回答其 base RGB 是否同时失败；不得据此晋级 TrackBayes/TubeBayes。

18-case 人工复核新增的是症状级证据而非因果证据：StreetGS #10/#12 的静态背景正常、远距小 actor 弱或缺失，
与 observation scarcity 假设高度相容；#11/#16/#17 也与 ownership/identity/visibility 问题相容。正式下一步是把失败
actor pixel 精确映射到 Gaussian contribution，并读取同 Gaussian 的 U2/B3 posterior、observation count、UNKNOWN 与
visibility。只有该 bridge 在至少 2 个 scene 稳定通过，才允许进入 TrackBayes 方法 arm。

## M2 — `SAFETY_LAYER_PENDING`

保留 V4 `154 requests / 214 candidates`、`83 accepted / 71 abstain` 与 full-denominator geometry MAE `+3.3908096237 m` caveat，
以及 V5 geometry-first 无 absolute-safe candidate 的 rejection。没有 exact request→新 case mapping，故不改 repair、不搜 threshold。
M2 在 V5.2 中降级为 risk-aware execution/abstention 层：先消费 Base Validity、M1 uncertainty、M3 candidate 与 geometry
status，再决定 execute/abstain；不得承担缺失观测恢复，也不得在 geometry undefined 时宣称 geometry-safe。

## M3 — `SYMPTOM_OVERLAP_STRONG_EXACT_TEMPORAL_BRIDGE_PENDING`

V4 temporal delta 的 validation/test confirmation 保持有效；V5 constraint-projection rejection 不倒写 V4。新 census 仅有 unwarped proxy，
不能命名 `B-TEMPORAL`，也无法证明 V4 delta 与真实 base temporal badcase 的 exact overlap，因此保持 pending。

人工复核中 #05/#06 的卡车拖影/错位与 #17 的行人透明 ghost 对 actor pose/trajectory 的症状匹配很强。下一步只做同 case
的 `t-1/t/t+1`、unwarped、flow-warped 与 frozen actor-pose-warped diagnostic，并 replay 已冻结 V4 SE(3) delta；不根据
panel 直接改 M3。warp bridge 失败也不倒写 V4 历史 confirmation。

## Research denominator

- Discovery design：#05/#10/#11/#16/#17；
- one-shot Confirmation：#06/#12/#18；
- BASE_FAILURE 与 unresolved case 不进入 M123 primary aggregate；
- 完整 ID、nuScenes path/split/hash 与回测 metric profile 见 P11 attribution manifest。

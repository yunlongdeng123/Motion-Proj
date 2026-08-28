# P30 Horizon-Conditioned Authority Freeze

P30 changes the explicit condition from action budget to future horizon H. A one-second P10V cache (72 cases, 733/864
eligible actions) is paired with its existing two-second cache and two additional two-second development domains. P10X is
excluded from P30 training.

The bounded case-offset model receives `horizon_seconds` as its eighth feature. It trains at H=`1.0/2.0s`, freezes, and then
materializes P10X at held-out H=`1.5s` (15 future frames). Action fraction stays fixed at 25%, the exact total budget matches
fixed P20, P20 within-case order is immutable, and each of three frozen two-scene context groups must cover at least 50% cases.

Gates require exact budget, global/group coverage at least `0.50`, reduction at least `0.40`, delta over fixed P20 at least
`0.10`, and at least five non-increasing scenes. No horizon, footprint, model, offset, coverage, group, loss, or gate sweep is
allowed. P10X is globally consumed, so the claim is held-out-horizon mechanism only. No hash/checksum/fingerprint.

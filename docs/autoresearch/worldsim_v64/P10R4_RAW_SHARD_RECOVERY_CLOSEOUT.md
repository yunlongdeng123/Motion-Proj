# P10R4 Raw Shard Recovery Closeout

Date: 2026-08-27  
Canonical run: `run://worldsim_v64/WS-V64-P10R4-TEST-SIDECAR-01/20260827T022000Z__test-raw-shard-recovery-s4-r2`  
Failure: `V64-F26` resolved

The restricted semantic-shard scan completed all five frozen archives in `1807.8114s` and found every one of the 14,437
required members. Per-shard counts were `05=5401`, `06=1824`, `07=1818`, `08=1783`, and `10=3611`. This also confirms
that capture prefixes can cross archive boundaries, while the frozen union of five archives was complete.

The persistent semantic member catalog now contains 85,992 entries and occupies 10,318,384 bytes. Temporary raw occupies
about 6.2 GiB and remains recoverable from the official archives; it will be removed only after all canonical processed scenes
are finalized. Free disk after raw completion was about 21 GiB.

Test quality, target evidence, route conflict, and model scores remained unread. No scene, model, policy, target frame, route,
denominator, tail, or gate changed. The dual-preprocess feeder has already produced complete target-free native leaves for
scene-0598 and scene-0462, each 12/12 targets and 4.1314 GiB peak GPU memory. Remaining native work continues under the same
prefix, reusing these leaves.

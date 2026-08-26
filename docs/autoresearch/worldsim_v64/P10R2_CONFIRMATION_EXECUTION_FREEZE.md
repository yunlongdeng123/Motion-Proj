# P10R2 Fresh Confirmation Execution Freeze

Date: 2026-08-27  
Hypothesis: `WS-V64-H-P10R2-002`

The metadata-only cohort in `P10R2_CONFIRMATION_COHORT_FREEZE.md` and fixed M1 policy are frozen before any confirmation
target quality read. Canonical execution IDs are:

- prep: `20260826T193000Z__confirmation-prep-s3-r1`;
- per-scene native prefix: `20260826T193500Z__confirmation-native`;
- native aggregate: `20260826T201000Z__native-aggregate-s3-r1`;
- evidence: `20260826T201500Z__confirmation-evidence-s3-r1`;
- exact-once confirmation: `20260826T203000Z__exact-once-confirmation-s3-r1`.

Preparation uses `/root/autodl-tmp/tmp/worldsim_v64_p10r2_confirmation_raw_batch`, streams a scene to the single RTX 3090
as soon as its members are ready, and permits at most two one-scene native workers. Background archive scanners yield to any
ready GPU work. The temporary raw directory is removed after successful preparation and remains recoverable from official tar
archives.

The exact-once runner keeps the frozen M0 conditional coverage map, M1 route cap `0.40`, total selected count preservation,
frozen MLP, 2-second/1.5-metre corridor, hidden-FREE threshold `0.05`, and worst10/96 empirical tail. The only gates are M1
route CVaR `<=0.05` and absolute mean total coverage delta `<=1e-6`. There is no refit, parameter sweep, second confirmation,
hash, checksum, fingerprint, smoke suite, or regression matrix.

A pass supports only a fresh observed empirical route-tail result for M1. It does not create a population bound, physical
collision label, planning/closed-loop result, or real-world safety claim, and it does not rewrite the historical current-M0
P10T rejection.

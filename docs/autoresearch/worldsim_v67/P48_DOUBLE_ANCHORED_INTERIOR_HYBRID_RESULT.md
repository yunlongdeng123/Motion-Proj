# P48 Double-Anchored Interior Hybrid Result

Canonical: `run://worldsim_v67/WS-V67-P48-DOUBLE-ANCHORED-INTERIOR-HYBRID-01/20260829T003000Z__double-anchored-interior-s0-r1`.

The eleven-domain adapter trained for 6,000 GPU epochs on 2,952 conditioned cases / 32,961 action rows. On newly
materialized P10R2 H=`1.5s`, it selected exactly `360/360` actions, covered `0.697917` of cases, retained minimum group
coverage `0.50`, and was non-increasing in all eight scenes.

P48/P31/fixed-P20 relative cost reduction was `0.742759/0.740902/0.406695`. The `+0.001857` delta over P31 missed the
frozen `+0.005` gate, so the verdict is `rejected_double_anchored_interior_hybrid` (3/4 gates). Endpoint preservation
remains true by construction, but the interior cross-cohort gain is too small. No gate reduction or peak/model/loss sweep.

Wall/peak GPU/peak RSS: `192.827s / 0.04502GiB / 1.33183GiB`.

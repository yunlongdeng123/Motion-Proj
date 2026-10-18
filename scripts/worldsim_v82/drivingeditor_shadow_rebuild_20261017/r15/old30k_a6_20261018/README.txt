R15 same-track a6 baseline inspection, 2026-10-18
Task: WS-V82-R15-OLD30K-A6-20261018. failure_ledger_refs: V82-F05. No new failure ID.
Uses original GN11 30k + rigid30k checkpoint, not later18k repair. Frozen inference only.
Old three owner IDs map to R15 a5/a26/a18; a6 has no old independent owner.
run_a6_direct: compare same physical a6 across49views, three documented OBB selections,
and common physical novel cameras. Old selections are geometry diagnostics, not asset GT.
run_a6_contributions: exact full-scene alpha*T attribution in unchanged a6 visible masks,
three fixed anchors105/113/137; retain >=98% static contribution for visualization.
Pixel contributors may include correct road/background. Never transfer ownership from these alone.
Original checkpoints/config/loss/optimizer unchanged. All scripts hold campaign GPU lock.
Original-view SH colors held fixed in novel geometry views; alpha is view-correct.
Remote outputs: nearby_actors_20261015_v1/old30k_fragment_audit_20261018.
Both checkpoints and all raw per-frame PNG/NPZ evidence retained there.

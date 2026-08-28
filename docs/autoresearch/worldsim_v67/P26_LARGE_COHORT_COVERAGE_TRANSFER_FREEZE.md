# P26 Large-Cohort Coverage Transfer Freeze

P26 keeps the P25 allocator architecture and contract unchanged. P4C becomes the tenth development domain; a new bounded
case-offset model is trained once and frozen before reading P6E trajectory targets. P20 within-case ranking remains frozen.

The confirmation cohort contains 16 stratum-balanced P6E scenes and 192 source cases. Allocation uses the exact same total
action count as a fixed 25% per-case baseline, permits 0--6 actions per case, and requires at least 50% case coverage.

Pre-registered gates are: exact total budget, case coverage at least `0.50`, relative selected-cost reduction at least `0.50`,
improvement over fixed P20 at least `0.10`, and at least 12 non-increasing scenes. No architecture, offset, coverage,
max-action, fraction, loss, or gate sweep is allowed. No hash, checksum, or fingerprint is introduced.

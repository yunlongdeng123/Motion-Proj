# P31 Joint Budget-Horizon Authority Freeze

P31 provides both selected-action fraction and future horizon H to the bounded case-offset model. Four development domains use
H=`1.0/2.0s`, and every domain contributes case rows at budget fractions `0.25/0.50`. P10X is excluded from training.

The model freezes before reading the existing P10X H=`1.5s` cache at budget `1/3`, a condition pair not present in training.
The selected count must exactly equal fixed P20 at one-third budget; global and each of three context-group case coverages must
be at least 50%, and P20 within-case order is immutable.

Gates require exact budget, global/group coverage at least `0.50`, cost reduction at least `0.40`, delta over fixed P20 at least
`0.10`, and at least five non-increasing scenes. No condition point, model, offset, coverage, group, loss, or gate sweep is
allowed. The claim is joint-condition mechanism on a consumed cohort only. No hash/checksum/fingerprint.

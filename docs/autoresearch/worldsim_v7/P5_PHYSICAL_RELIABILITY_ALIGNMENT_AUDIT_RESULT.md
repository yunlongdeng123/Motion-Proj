# P5 Physical--Reliability Alignment Audit Result

Date: 2026-09-02

Canonical run:
`run://worldsim_v7/WS-V7-P5-PHYSICAL-RELIABILITY-ALIGNMENT-AUDIT-01/20260902T180000Z__physical-reliability-alignment-s0-r1`

## Verdict

`frozen_interface_descriptive_audit_only`

The exact scene/instance join mapped all 313 P4 nuScenes Actors into DriveStudio identity space, but only 118 Actors have matching
V6.7 P109 source rows. Most importantly, P4 train contributes only 5 aligned Actors from 2 scenes, below the frozen direct-fit
minimum of 20 Actors from 3 scenes. A new joint C3 head would therefore be dominated by scene identity and cannot form a
nontrivial train-side scene holdout. No model was trained, refit, calibrated, or rescued with calibration/test Actors.

## Coverage

| P4 role | P4 scenes / Actors | aligned scenes / Actors | aligned V6.7 rows |
|---|---:|---:|---:|
| train | 8 / 29 | 2 / 5 | 1,320 |
| calibration | 14 / 56 | 5 / 25 | 7,645 |
| test | 37 / 228 | 14 / 88 | 39,285 |

The frozen V6.7 source contains 575,596 rows over 102 scenes and 2,800 scene--Actor pairs at horizons
`0.8/1.5/2.5/3.0 s`. Test-aligned Actors span four existing V6.7 scene folds (`0/1/3/4`), so they remain useful for a
descriptive, no-fit interface analysis.

## Claim boundary and next action

This result establishes identity and coverage only. It does not establish that physical repair improves C3 reliability and does
not extend V6.7's source-domain empirical reliability claim. P5 continues with frozen P4 decisions and retained V6.7 outcomes on
the 118 exact-match Actors, reporting multi-horizon cost/false-safe strata without changing either model.

# P5-B Frozen Physical--Reliability Interface Freeze

Date: 2026-09-02

## Question

On exact-match nuScenes Actors, do frozen P4 physical-repair decisions coincide with high retained V6.7 multi-horizon cost,
state error, occupancy decision flips, or false-safe outcomes?

## Fixed analysis

- Identity comes only from the completed P5-A exact scene/instance join.
- P4 scores and selection decisions are read from the frozen P4 canonical run. No threshold is recomputed.
- V6.7 outcomes are the retained P109 rows used by the reliability stack at `0.8/1.5/2.5/3.0 s`.
- Primary role is P4 test; calibration is descriptive context. Train is excluded after `V7-F13`.
- Groups are fixed before execution: all, P4 selected/abstained, geometrically helpful/harmful repair, and selected-and-harmful.
- Report pooled row rates, equal-Actor aggregates, per-horizon strata, and Spearman associations. No bootstrap, sweep, fit, or gate.

This run does not execute P346 or claim a causal reliability improvement: P109 outcomes are the interface substrate, while both P4
and P346 weights remain frozen. Its purpose is to locate co-occurring physical and trajectory-risk failure modes and define a
paper-facing safety boundary.

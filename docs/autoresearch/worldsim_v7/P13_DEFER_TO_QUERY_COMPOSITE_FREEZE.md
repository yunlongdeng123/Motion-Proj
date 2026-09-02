# P13 Defer-to-Query Composite World Freeze

Date: 2026-09-02

Task: `WS-V7-P13-DEFER-TO-QUERY-COMPOSITE-01`

## Motivation

Selective prediction distinguishes conditional risk among accepted cases from the population cost of an accept-or-defer
system. P10--P12 report the former, but HARP-3D's actual abstention action retains the original query surface; it never deletes
an Actor. P13 therefore measures the composite world produced by each already frozen policy.

The migration follows SelectiveNet/one-sided selective classification and learning-to-defer semantics, while retaining the
known warning that abstention can amplify group disparities. No new selector or confidence claim is introduced.

## Frozen protocol

- Exact join: P10 r2, P11 r2, and P12 r1 compact JSONL on the same 20 AV2 logs / 523 Actors.
- Policies: query-only, always-repair, P4, P6-C, P11 P4-and-provenance, P12 visibility-only, and P12 P4-and-visibility.
- Action: selected Actor uses the existing compiled surface; abstained Actor uses its original query surface.
- Metrics: repair coverage, hazard-repair coverage, selected conditional visible/Chamfer risk, population introduced-visible
  failure mass, population Chamfer-worsening mass, composite mean Chamfer gain, and a two-axis risk/gain Pareto frontier.
- Actor and hazardous-Actor identity/trajectory/extent/semantics retention is 100% for every policy by construction.

No AV2 files are read, no model runs, and no score/threshold/gate/cohort changes. `query_only` contributes zero *introduced*
visible failure by definition; this does not claim that the original query has no pre-existing contradiction. The result is
descriptive consumed-development evidence, not a conformal, collision, planning, closed-loop, or deployment-safety guarantee.

## Waymo resource note

Official Waymo v2.0.1 modular Parquet was selected as the preferred second external domain because it permits component-only
downloads. A read-only AutoDL HEAD probe to the official validation LiDAR object returned HTTP 403 without an accepted Waymo/
Google session. No unofficial mirror, credential workaround, or unlicensed copy is used. This is a resource/access condition,
not a scientific failure ID; P13 proceeds on frozen AV2 artifacts and does not pretend to be Waymo evidence.

# WorldSim V7.1 M48 — Supervised child visibility inside the return measure

## Rationale

M47 rules out a reusable motion, provenance, or incidence threshold and shows that local support changes transport labels among rays. M48 separates immutable canonical geometry from sensor visibility. It retains M11's GT-supervised oriented surface and learns a bounded ray--child visibility factor directly through the deployed categorical first-return events.

## Representation and supervision

For ray `r` and completion child `j`, a 5D physical descriptor contains absolute ray--normal cosine, closest lateral miss normalized by tangent scale, longitudinal center depth normalized by actor-box diagonal, normal/tangent scale ratio, and frozen occupied authority. A small MLP produces `v_rj in (0,1)` and the exact joint measure becomes

`q(k,j|r) proportional to o_j * v_rj * kappa_kj`.

Observed-anchor measure is unchanged. M8 centers/scales, M11 normals/thickness, and M35/M38 authority are frozen. Only visibility is trained with GT `P(d >= d_gt-0.20m)` and `P(|d-d_gt| <= 0.20m)` NLL plus a small identity prior. No target depth, target point, hazard, motion, category, image feature, or external statistic enters the visibility head.

## Frozen protocol

- 593 train / 66 exposed holdout, seed 71148, 4 epochs, 256 rays/actor, 32 train and 64 deploy bins.
- One 5--32--1 MLP, initial visibility 0.95, identity weight 0.05; no sweep.
- Compare against M39 and frozen M45 on all/hazard/clear. Pass requires all and both strata early non-increase versus M39, with all hit no worse than -1 point.
- Visibility is evaluated inside the renderer; it never deletes a Gaussian or filters a returned surface after inference.
- M48 is development-only. M43 remains the sole frozen external candidate and no partial AV2 quality is read.

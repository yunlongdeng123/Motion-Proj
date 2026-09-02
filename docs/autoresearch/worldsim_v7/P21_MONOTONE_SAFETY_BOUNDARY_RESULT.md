# P21 Monotone First-Return Safety Boundary Result

## Status

- Hypothesis: `WS-V7-H-P21-001`
- Canonical: `run://worldsim_v7/WS-V7-P21-MONOTONE-SAFETY-BOUNDARY-01/20260903T134500Z__monotone-safety-boundary-r1`
- Verdict: `supported_monotone_first_return_safety_boundary`
- Input: frozen P20 `summary.json` only
- New dataset/model/checkpoint/threshold/policy read: none
- Fresh AV2 cohort read: false

## Exact boundary

For a ray \(r\), let

\[
d_S(r)=\min\{\langle x,o_r\rangle:x\in S,\;\langle x,o_r\rangle>0,\;\operatorname{lat}(x,r)\le\epsilon\},
\]

with \(d_S(r)=+\infty\) when the set is empty. If \(S'\subseteq S\), then

\[
d_{S'}(r)\ge d_S(r).
\]

Therefore deleting compiled points cannot create a new earlier literal first return relative to a fixed query/target ray. The
result is deterministic set inclusion, not a distributional or learned guarantee.

It does **not** guarantee retained matched hits, non-worsening symmetric Chamfer, collision freedom, policy improvement,
closed-loop safety, or transfer to another sensor/domain.

## Frozen empirical frontier

| policy | hazardous early events removed | new hits lost | mean Chamfer penalty | events removed / hit lost | events removed / mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| P17 ray-only | 1,570 | 3,384 | 4.82425 mm | 0.463948 | 325.439 |
| P17R ray+Chamfer | 833 | 955 | 1.129153 mm | 0.872251 | 737.721 |
| P19 one-slot veto | 131 | 71 | 0.0918622 mm | 1.84507 | 1,426.05 |

P19 is the most efficient frozen deletion point on this consumed-source audit, but it still violates the pre-registered
non-worsening-Chamfer requirement. No policy is promoted and no ratio is used to select a new operating point.

## Interpretation

P20/P21 replace a weak target-nearest proximity proxy with literal first-return geometry and then separate two questions that
must not be conflated:

1. **Directional safety:** deletion is monotone for earlier first returns.
2. **Surface utility:** deletion can remove supported returns and worsen bidirectional geometry.

This is the requested explainable safety boundary. It is suitable for the theorem/limitations narrative, while the zero-shot
claim remains the separately frozen nuScenes-to-AV2 empirical evidence.

## Resources and paper handoff

P21 wall time was analysis-only; peak GPU allocation and data exposure are zero beyond reading the P20 summary. The result was
integrated into the CVPR abstract, related work, method, experiments, limitations, conclusion, supplement, bibliography, result
macros, and contribution registry. Final compilation is 8 content pages plus 1 references page for the main paper and 9 pages
for the supplement; there are no undefined references or citations. The sole retained layout warning is the pre-existing,
non-clipping 6.03 pt overfull box in Table 1.

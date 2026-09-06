# WorldSim V7.1 M45 — oriented categorical child surface measure

## Hypothesis

M44 attributes M39's remaining clear-stratum pre-boundary mass to completion children. M10--M11 already learned child normals
and independent normal thickness from GT, but rejected them under earliest hard ellipsoid intersection. M45 tests a distinct
composition: retain M39's normalized categorical surface-return measure while replacing only the isotropic child kernel by
the frozen M11 oblate kernel. Geometry centers, tangent scales, anchor kernel, and both authority heads remain frozen.

For child displacement `r`, unit normal `n`, tangent radius `s`, and normal thickness `h`, the categorical component is

`o * exp(-0.5 * ((||r||²-(r·n)²)/s² + (r·n)²/h²))`.

This follows Gaussian surfel work such as Geometry Field Splatting (CVPR 2025), which represents opaque geometry with planar
Gaussian kernels rather than treating every Gaussian as an isotropic volume. It does not reopen M11's hard-intersection head.

## Frozen protocol

- 66 development-exposed holdout Actors; M8 centers/tangent scales, M11 normals/thickness, M35/M38 authority, 64-bin CDF.
- Baseline=M8 unit isotropic energy; reference=M39 learned isotropic categorical; candidate=M39 authority with oriented child
  kernels. Candidate is fixed before reading results.
- Decisions versus M39: all early delta ≤ 0; hazard and clear early deltas ≤ 0; all hit delta ≥ -1pp.
- No training, normal/thickness/scale/bin/median sweep, filtering, deletion, or external/M43 partial read.

M45 cannot replace the already frozen M43 external candidate. If supported it is a next-generation development mechanism that
requires a future untouched cohort; if rejected, register `V71-F44` and do not revisit oriented child kernels by tuning thickness.

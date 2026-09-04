# WorldSim V7.1 research synthesis and claim matrix

Date: 2026-09-05

## One-sentence result

V7.1 establishes a supervision-native Actor-canonical geometry path and a learned evidential categorical return measure on nuScenes, proves strict ownership/equivariance and an attenuation safety boundary, and reserves cross-sensor generalization for the single frozen M39-to-AV2 read that is still running.

## Factorized scientific state

### 1. Canonical geometry: supported on exposed nuScenes development

M7 replaces one-center relocation with four-child set completion. Every final center remains under actor-canonical GT set, local-plane, scale, literal first-return, and free-before-hit supervision; observed anchors remain immutable and all children deploy without UNKNOWN deletion. Relative to the unit completion baseline, M7 reduces hazardous/all early returns by 10.220/8.714%, improves Chamfer by 2.902 mm and hit recall by 2.173 points, with 100% Actor/hazard retention.

M8 adds equal per-frame endpoint coverage while keeping trajectory outside the shape head. Relative to baseline it reduces hazardous/all early returns by 5.123/3.901%, improves Chamfer by 6.207 mm and hit recall by 2.759 points. Moving and quasi-static frame-balanced distance both improve, so its canonical shape result does not depend on a motion label.

Claim allowed: training supervision changes the deployed geometry and produces a source-domain physical Pareto without post-hoc deletion.

Claim forbidden: watertight reconstruction, collision freedom, unseen-category completion, or external generalization before M43.

### 2. Surface-return authority: M39 supported on exposed development

M33 preserves producer-side ray/frame/provenance evidence; M35/M38 learn separate continuous FREE/OCCUPIED/UNKNOWN authority for observed anchors and generated children. M39 composes those fixed masses as a categorical surface-return measure rather than density-dependent optical thickness. Relative to unit energy, M39 changes all/hazard/clear early returns by -0.521/-0.621/-0.030 points and hit recall by +2.172/+2.318/+1.455 points; all three frozen source criteria pass.

Claim allowed: continuous evidential mass is useful as return evidence under this categorical composition.

Claim forbidden: occupied mass is calibrated volumetric opacity, probability of collision, or a basis for deleting primitives.

### 3. Geometry, pose, static world, and appearance: interfaces are separated

M22 verifies inverse-query SE(3) equivariance across 36 Actor-frames and 126.28 m maximum translation to numerical precision. M28 encodes physical field, read-only Actor pose, and visual layer as separate typed authorities. M23--M27 show that image supervision can train SH/opacity without changing physical centers/scales; the coarse--fine appearance mechanism improves all six exposed views over one-carrier rendering but remains 7.62 dB below original StreetGS and is not an appearance-generalization result.

Claim allowed: appearance-only updates are structurally unable to alter the physical query; rigid pose relocates but cannot deform canonical geometry.

Claim forbidden: the supplied trajectory is accurate, dynamic/static discovery is solved, Background collision is modeled, or rendering is photorealistic.

### 4. Explainability and safety boundary: supported analytically and empirically

For the joint return measure, median early return is exactly equivalent to the interpolated pre-boundary CDF exceeding 0.5. M49 further proves

`dF/dlog(w_j) = r_j(C_j-F)`

and, for finite attenuation `v`,

`F_v-F = (1-v) r_j (F-C_j) / (1-(1-v)r_j)`.

Thus attenuation is safe only when the attenuated component/family lies earlier than the current mixture (`C_j>F`); its magnitude cannot reverse the sign. Frozen M48 attenuation raises the boundary CDF on 59.32% of 99,208 rays, while the first-order sign predicts the finite-change sign on 95.73%.

Claim allowed: normalized mixture attenuation has a precise non-monotone safety boundary.

Claim forbidden: using the GT-dependent derivative sign as a deployment gate, real-road safety certification, or worst-case clearance certification.

### 5. Motion and visibility: explicitly unresolved, not hidden inside geometry

M47 shows M46's hazard regression is not a reusable moving/static, KEEP/PROJECT, or incidence split. Hazard-moving rays slightly improve while hazard-quasi-static rays regress; five actor-level physical correlations have magnitude at most 0.100. M48 then shows that a supervised, bounded ray-conditioned visibility factor still transfers risk between hazard and clear strata. The visibility-head family is closed.

M50 separately tests whether M8's pooled physical rays merely underweight sparse target frames. Equal-frame GT first-return
training is rejected: actor-mean worst-frame early rate rises by 1.135 points overall and 1.286 points for hazardous Actors,
while aggregate hazardous early rate rises by 0.590 points. Chamfer improves by 0.871 mm, so the failure is not lack of
optimization; frame reweighting does not close the smooth-depth versus hard earliest-return surrogate gap. Temporal balancing
is therefore closed as a substitute for explicit motion state.

If future work predicts temporal evolution, canonical shape and deformation must be separate states: rigid motion should receive trajectory/scene-flow supervision; non-rigid deformation requires its own capacity and labels. Hazard is never a geometry or motion label.

## Cross-domain state

M43 is the only eligible frozen external candidate. It reconstructs all M8 geometry and M35/M38 authority inputs from AV2 build sweeps, then evaluates target sweeps once. No AV2 fine-tuning, normalization fit, calibration, threshold selection, log replacement, or partial quality read is permitted. Current status is 14/20 processed logs while the sole downloader continues; roughly 88 GB remains.

Final branch:

- If M43 passes all three frozen criteria, the paper may claim a nuScenes-to-AV2 zero-shot direction for evidential Actor-local surface returns.
- If M43 fails, no target adaptation follows. The paper claims source-domain supervision-native physical completion, exact ownership/equivariance, and the normalized-mixture safety boundary; cross-sensor transfer remains an observed limitation.

## Paper compression priority

The main paper should retain: the supervision-native target construction; M7/M8 geometry; M39 joint return measure; M43 external result; typed ownership/SE(3); and M49 safety theorem/figure. M0--M6, M9--M21, M23--M27, M29--M38, and M40--M48 are failure-directed evidence for the supplement except where one sentence is necessary to motivate a retained result. This preserves the causal story: GT geometry, return measure, state ownership, external test, safety boundary.

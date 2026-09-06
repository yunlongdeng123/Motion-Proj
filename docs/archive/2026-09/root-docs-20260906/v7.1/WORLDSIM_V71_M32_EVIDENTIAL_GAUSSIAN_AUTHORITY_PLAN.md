# WorldSim V7.1 M32 — GT-Supervised Evidential Gaussian Authority

状态：`frozen / one development run / no sweep`  
任务：`WS-V71-M32-EVIDENTIAL-GAUSSIAN-AUTHORITY-01`

## 1. Failure-directed change

M31 shows that immutable hard surface authority is false before completion: KEEP supplies 66.70% of anchor early returns,
while PROJECT is 2.49x overrepresented relative to its point share. Existing UNKNOWN filtering changes raw-query early rate
by only 0.629 points. M18 already established that a GT one-hot categorical return distribution is preferable to a
post-hoc surface rule. M19 and M20 then exposed two shortcuts: a flexible decoder can compensate for bad geometry, and
joint Gaussian geometry can inflate scale.

M32 freezes every M8 center and scale and learns only a continuous three-mass authority for each primitive. This is a new
shared physical representation, not a filter:

\[
  (m_F,m_O,m_U)=\operatorname{softmax}(h_\theta(z_i)),\qquad
  E(x)=\sum_i m_{O,i}\exp\left(-\frac{\lVert x-c_i\rVert^2}{2s_i^2}\right).
\]

No mass is converted to a binary decision and no primitive is deleted.

## 2. GT and objective

- Frozen carriers: immutable observed anchors at 0.08m scale plus frozen M8 children and supervised scales.
- Build-only features: normalized Actor-local center, absolute center, scale, anchor/child type and Actor size. No target,
  trajectory, category, hazard, time or image enters the scorer.
- Primitive GT: held-out LiDAR rays vote FREE when the primitive lies before a return, OCCUPIED near the endpoint and
  UNKNOWN when unsupported. A fixed pseudocount one preserves epistemic mass; conflicting votes remain a soft target.
- Ray GT: the same 32-bin one-hot first-return categorical proper loss used by M18/M20, plus fixed 0.1 expected-depth L1.
- Total: ray categorical + depth + 0.5 primitive evidential cross-entropy. Both terms are training-time native LiDAR GT.
- Deployment/evaluation: 64-bin CDF median of the same continuously weighted Gaussian energy.

## 3. Frozen experiment

- 659 eligible source Actors, same every-tenth split: 593 train / 66 exposed development holdout.
- seed 71132, six epochs, 128 training rays, 256 evidence rays, hidden width 64, one run.
- Comparator: identical M8 centers/scales with unit authority, i.e. frozen M21 analytic energy.
- Three decisions only: all early non-increase; neither hazard nor clear early worsens; all hit delta at least -1 point.
- Geometry and Actor/trajectory/hazard state retention are exact by construction and reported, not re-gated.

Failure closes this amplitude family: no loss-weight, pseudocount, width, epoch, scale, bin, threshold or seed recovery.
Success remains development mechanism evidence and cannot enter AV2 because that cohort is already reserved for frozen
M5/M7/M8/M18/M21 candidates.

## 4. Literature boundary

- [EvOcc (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.html)
  directly supervises evidential FREE/OCCUPIED/UNKNOWN occupancy from noisy LiDAR.
- [ALSO (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Boulch_ALSO_Automotive_Lidar_Self-Supervision_by_Occupancy_Estimation_CVPR_2023_paper.html)
  derives confident occupied/free samples from sensor geometry.
- [GaussianFormer-2 (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_GaussianFormer-2_Probabilistic_Gaussian_Superposition_for_Efficient_3D_Occupancy_Prediction_CVPR_2025_paper.html)
  motivates interpreting Gaussian carriers as occupied-region distributions.

M32 migrates only supervision and probabilistic superposition. It does not claim semantic occupancy, calibrated safety,
image completion, dynamic discovery or closed-loop planning.

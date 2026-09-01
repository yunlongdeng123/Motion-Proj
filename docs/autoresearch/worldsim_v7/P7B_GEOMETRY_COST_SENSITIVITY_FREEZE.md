# P7-B Geometry-to-Cost Sensitivity Freeze

Date: 2026-09-02

## Deterministic boundary

For profile point $j$, define projected Actor error $a_j=|n_j^T e_j|$, signed clearance $d_j$, floor
$m_j=\max(d_j,\epsilon)$, and task cost $C(d)=\max_j a_j/m_j$. Under a clearance perturbation $d'_j=d_j+\delta$,

$$
|C(d')-C(d)|
\le \max_j a_j\left|m_j^{-1}-{m'_j}^{-1}\right|
\le \max_j \frac{a_j|\delta|}{m_jm'_j}
\le \frac{a_{\max}|\delta|}{\epsilon^2}.
$$

The first inequality follows from the max operator's 1-Lipschitz property; the second uses the clipped denominator's
1-Lipschitz property. The experiment measures the row-wise local bound and its tightness, not just the deliberately loose global
constant.

## Frozen stress test

- Source: all 575,596 retained V6.7 P109 profiles; no new sensor read.
- P5 strata: exact-match test all/selected/abstained, using frozen P5-B identities and P4 decisions.
- Fixed uniform signed clearance shifts: $\pm0.05,\pm0.10,\pm0.20$ m; $\epsilon=0.05$ m.
- Report absolute cost shift, local-bound tightness, clipped-denominator/sign crossings, and bound violations.
- One GPU-vectorized pass per shift; no training, fitting, calibration, threshold, sweep, or model selection.

This numerical audit is motivated by signed-distance representations for driving geometry and safety-aware motion prediction, but
its theorem is algebraic and self-contained:

- https://openaccess.thecvf.com/content/CVPR2024/html/Liu_SurroundSDF_Implicit_3D_Scene_Understanding_Based_on_Signed_Distance_Field_CVPR_2024_paper.html
- https://openaccess.thecvf.com/content/ICCV2021/papers/Ren_Safety-Aware_Motion_Prediction_With_Unseen_Vehicles_for_Autonomous_Driving_ICCV_2021_paper.pdf

No claim is made about the probability or realism of these uniform perturbations, and satisfying the deterministic inequality is
not a safety certificate.

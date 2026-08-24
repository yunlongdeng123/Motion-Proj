# P6 legacy28 closeout and sole recovery preregistration

## Canonical result

- task: `WS-V62-P6-LEGACY28-ME-01`
- hypothesis: `WS-V62-H-P6-001`
- canonical: `run://worldsim_v62/WS-V62-P6-LEGACY28-ME-01/20260824T095529Z__legacy28-s0-r1`
- source: `d14827df958494d44121f296848ead62c77cb37e`
- terminal: `rejected`

The run reused the canonical V6.1 argmax artifact, fit its lossy class prototypes from the P5 train split, wrote all method decisions
and candidate geometry before loading O_eval, and did not rerun IR-WM or read confirmation/test.

## Matched arms

| Arm | ACCEPT | False-safe | Mask-area yield | Mean/worst accepted FREE conflict |
|---|---:|---:|---:|---:|
| B0 IR-WM argmax | 10/28 | 10/10 | 0.398300 | 0.267482 / 0.570571 |
| B1 hard clip | 10/28 | 10/10 | 0.398300 | 0.050578 / 0.117225 |
| B3 evidential, no projection | 4/28 | 4/4 | 0.094024 | 0.043202 / 0.079602 |
| B5 CPSC-Lite pre-conformal | 4/28 | 4/4 | 0.094024 | 0.043202 / 0.079602 |

B5 retained `2/3` R10 cases, added `0` actor and `2` static cases, and failed `ACCEPT>=5`, zero false-safe and 12% mask-area.
All four accepts were route-support cases. B3 and B5 produced the same case decisions; exact projection still had `0/939,206` hard
violations but did not supply hidden geometry authority.

Anti-triviality also failed: source-valid UNKNOWN=`0.827351` against the frozen `0.50` upper bound. Oracle accepted-surface safe-OCC
retention was `1.0`, so the problem was not deletion of known occupied support. Accepted candidates still had O_eval UNKNOWN fractions
around `0.44..0.58`; two f057 arms also exceeded the 0.05 hidden-FREE bound. B1 reduced conflict substantially but remained
`10/10 false-safe`, so Stop 1 did not trigger.

Resources were bounded: wall=`47.20s`, peak=`0.5319GiB`, pre-closeout output=`2,273,574 bytes`, disk free=`64.11GiB`.

## Diagnosis and primary-source recovery choice

The bridge audit already showed a smaller selection degradation, but legacy inference amplified missing-feature shift into 82.7%
UNKNOWN. Query-wise hard projection cannot reconstruct discarded logit/BEV information or certify hidden surfaces. A stronger geometric
clip would repeat the zero-yield observed-FREE-veto failure, while a set-valued head would require the still-locked P8 calibration state.

- [Modality-Agnostic Learning, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Modality-Agnostic_Learning_for_Radar-Lidar_Fusion_in_Vehicle_Detection_CVPR_2022_paper.html)
  uses simulated missing modalities during training plus teacher/student consistency to improve inference with missing data.
- [Are Multimodal Transformers Robust to Missing Modality?, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Are_Multimodal_Transformers_Robust_to_Missing_Modality_CVPR_2022_paper.html)
  reports that fused models are sensitive to missing inputs and that fusion strategy determines robustness.
- [Selective Sensor Fusion, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Selective_Sensor_Fusion_for_Neural_Visual-Inertial_Odometry_CVPR_2019_paper.html)
  demonstrates masking-based fusion under corrupted and missing sensor streams.

The sole authorized mechanism recovery is therefore evidence/feature dropout, not projection or threshold tuning.

## `WS-V62-P6R-EVIDENCE-DROPOUT-RECOVERY-01`

Freeze one continued-training run before execution:

- student initializes from the P5 best checkpoint; an identical frozen teacher consumes full P4 features;
- on P2/P4 train queries, replace full logits/BEV with the train-only class prototype independently with probability `0.5`;
- optimize the unchanged P5 task loss plus `0.25 × KL(teacher full-view base probabilities || student corrupted-view base probabilities)`;
- AdamW `1e-4`, FP16, batch `16,384`, accumulation `2`, max `6` epochs, min `3`, patience `2`, seed0;
- select once on the pure prototype view using the same task-plus-consistency objective; report both full and prototype selection;
- no legacy O_eval, confirmation/test, new backend, threshold/grid/window/model-size/bridge sweep or integrity fingerprint;
- no capacity smoke because P5 and P6 already bounded the identical model and batch path.

After the checkpoint is frozen, run one P6R legacy28 evaluation with the unchanged P6 arms, denominator, gates and prototype builder.
Pass follows the original P6 decision. Failure closes CPSC-Lite; there is no second recovery.

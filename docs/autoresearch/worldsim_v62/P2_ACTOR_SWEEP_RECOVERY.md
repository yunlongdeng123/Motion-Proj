# P2 actor temporal-sweep recovery

Date: 2026-08-24  
Task: `WS-V62-P2-EVIDENCE-QUERY-DATASET-01`  
Failure: `V62-F03` (`resolved`)

## Observation

Formal r1 stopped at `scene-1012/f152` because the instantaneous actor-envelope candidate pool was empty. The frame still
contains four annotated actors, but all four lie outside the frozen `x=[-20,40], y=[-30,30], z=[-3,5] m` ROI. Actor 8 is
inside that ROI at visible method frame 146 and has moved outside it by target frame 152. The failed run did not complete its
top-level manifests and was not used for training or a quality claim.

Failed run:

```text
run://worldsim_v62/WS-V62-P2-EVIDENCE-QUERY-DATASET-01/20260824T082601Z__query-dataset-s20260824-r1
```

## Primary-source migration

- [QueryOcc (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Lilja_QueryOcc_Query-based_Self-Supervision_for_3D_Semantic_Occupancy_CVPR_2026_paper.html)
  supervises independent continuous 4D queries across adjacent frames.
- [SparseOcc (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Tang_SparseOcc_Rethinking_Sparse_Latent_Representation_for_Vision-Based_Semantic_Occupancy_Prediction_CVPR_2024_paper.html)
  uses sparse queries to avoid treating every dense voxel as an equally useful denominator.
- [OPUS](https://arxiv.org/abs/2409.09350) formulates occupancy as sparse set prediction with consistent point sampling.

The project-level inference is that actor query support should follow actors through the method-visible temporal window. It
must not require an actor to remain inside the instantaneous target ROI, and it must not silently reassign the actor budget to
easy free space.

## Recovery

For each evidence build, the actor query pool is now:

```text
current target actor envelope union visible source-sweep actor envelopes
```

The source-sweep boxes are expressed in the target LiDAR frame and clipped only by the already-frozen ROI. They define query
support only: they are not promoted to hard occupied evidence. Method queries see only visible method sweeps; the held-out
dropout and target-evidence sweeps are not imported. Query quotas remain exactly `25k/15k/25k/15k/15k/5k`, including 15k
actor queries and 100k total queries per target.

## Targeted reproduction

```text
run://worldsim_v62/WS-V62-P2-EVIDENCE-QUERY-DATASET-01/20260824T083403Z__actor-sweep-repro-s20260824-r5
```

Result: process exit `0`; current envelope `0`; visible swept envelope `450` voxels; actor-type queries `15000/15000`;
total queries `100000`; wall `2.62s`. No GPU, quality result, confirmation/test read, hash, checksum, or fingerprint was used.

Next: rerun the frozen 72-unit formal materialization from the recovery commit without adding another smoke stage.

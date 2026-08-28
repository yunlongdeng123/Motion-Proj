# V6.7 P15 trajectory-conditioned reliability training freeze

P15 no longer predicts whether an individual voxel is correct. Its target is the expected hidden-FREE cost among world/Actor
states visited by a fixed Ego candidate trajectory over `H=2.0s` and a `1.5m` corridor.

Training uses the 813 consumed V65 P10V action rows; one selection read uses the 739 consumed V65 P10X action rows. The fixed
12-action lattice is unchanged. A low-capacity `64/64` residual MLP receives qmean, visited count, progress/lateral action context,
and within-case relative qmean. The objective jointly optimizes expected-cost Huber regression, unsafe classification and
within-case pairwise ranking for 3,000 GPU epochs. qmean remains the residual base score.

Selection requires Spearman >=0.70, unsafe AUROC >=0.90, pairwise concordance >=0.65, bottom-quartile selected-cost reduction
>=0.25, >=0.05 reduction gain over qmean, and non-increasing cost in at least five scenes. Architecture, loss, lattice and gates
are frozen for one selection run. No collision/planner/policy/safety claim is allowed.

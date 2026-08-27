# V6.5 P2V Fresh Trajectory-Visited-State Transfer Freeze

## Cohort

Six scenes were selected before target evidence or q0 scores from deterministic quantiles of all scene-index-sorted
candidates satisfying three capability-only conditions: present as a direct key in the frozen 700-scene IR-WM temporal
metadata, absent from repository configs/docs, and not already processed. The fixed cohort is:

`scene-0001/0219/0402/0594/0822/1110`, with processed indices `0/169/318/474/638/849`.

Metadata descriptions span construction/trucks, junction/residential traffic, rain, pedestrians, and night. Description
was recorded only after quantile selection; it did not change membership. There are 12 fixed frames per scene and 72
total trajectory units.

## Candidate and target

- Candidate: deterministic `Qagg = mean(frozen q0 risk | state lies within 1.5 m of future 20-frame Ego trajectory)`.
- Target: realized hidden-FREE fraction in that visited footprint.
- Eligibility: at least 16 sampled visited states.
- No map residual or learned R4 head is transferred.

## Single formal read

The fresh prediction object passes only if Spearman >=0.60, unsafe-unit AUROC >=0.85, lowest-risk 40% realized cost
reduction >=40%, and within-scene selected cost improves over the scene mean in at least 5/6 scenes. All gates, scenes,
frames, q0 model, corridor, and run paths are frozen before preparation.

Preparation remains quality-blind. As scenes become ready, two single-GPU native workers may overlap later scene I/O;
CPU evidence materialization may overlap native GPU work. The already-frozen R5 Actor diagnostic runs on GPU while the
first archive/preprocess I/O is active.


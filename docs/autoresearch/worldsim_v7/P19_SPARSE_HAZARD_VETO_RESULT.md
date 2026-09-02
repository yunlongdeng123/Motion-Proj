# P19 Sparse Hazard Veto Result

Date: 2026-09-02

Canonical: `run://worldsim_v7/WS-V7-P19-SPARSE-HAZARD-VETO-SOURCE-01/20260903T114000Z__sparse-hazard-veto-s71701-r1`

P19 is rejected on consumed source development and fresh AV2 remains unread. The fixed one-slot rule vetoes one candidate for 35 of
79 hazardous Actors, or `35/3325` candidates overall, while leaving all 149 clear Actors unchanged. Completion coverage is
`98.947%`.

Hazardous new-early falls by 34 events, from `1.43622%` to `1.41532%`; total new-early also falls from 2,982 to 2,948. The action
is not utility-free: mean Chamfer increases from `.1945868 m` to `.1946787 m`, and new hits fall from 39,255 to 39,119. The frozen
two-part source gate therefore fails on Chamfer. Wall time is `258.97 s`, peak GPU memory `.0428 GiB`, peak RSS `1.153 GiB`, and
`target_data_read=false`.

Register `V7-F28` and close score-ranked veto capacity/threshold variants. The result also exposes a more fundamental audit issue:
the legacy visible-failure metric selects the Euclidean-nearest surface point to each target point and only then checks ray depth;
it is not a literal first-return renderer. P20 freezes a true minimum-positive-depth ray audit before examining any corrected result.

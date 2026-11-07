# R0 protection and FRONT-only R1 benchmark

The authoritative R1 generation is `FRONT_geometry_v2/`. The first `public/`
selection is preserved as a draft because its old `source_valid` field also
depended on full-image sharpness. Version 2 recomputes direct measured-return
witnesses with no RGB or quality gate. Both happened to select the same bounds.

`integration_domains/` contains the exact original V44 masks for R0. It is
outside R1 public inputs and must never be mounted into a blinded estimator.
The current integration asset is C2/SH0/RGBA with SurfaceLOD, not old M2.

R1 separates appearance-blind (target XYZ permitted, all intensity/RGB removed)
and geometry+RGB-blind (target+guard XYZ removed before fusion). Each prepared
track directory is self-contained. Mount or copy **only one track directory**
for its estimator; keep the other track, original scene, old checkpoints and
`evaluator_only/` unavailable. Manifest allowlists are not OS-level isolation.

Selection uses fixed metric patches of 0.5, 1 and 2 m, a 0.75 m world guard,
and a spatial hash to reserve validation/test patches. It does not read target
RGB, marking classes, color stability or fit errors. Height masks use a broad
context-derived interval, not an exact target-height silhouette.

Native FRONT JPEG is masked before resize. The CPU raw LiDAR decoder processes
all five sensors and both returns; the geometry-blind prism filter precedes any
pool, normal/ground inference or feature extraction. No renderer or neural
framework is imported. A native-color intervention must leave masked outputs
exactly unchanged. Source SHA256, CRC, removed-point and masked-pixel counts are
recorded for each of the 199 declared FRONT frames.

The projection uses all nine Brown intrinsics and a conservative interval
rectangle; rolling-shutter constant-velocity inflation is recorded as an
assumption. Physical timing/distortion bounds and actual estimator process
isolation are not independently certified. The other four RGB cameras are
excluded and unsearched. This is never an all-five-camera strictness claim.

Seal only after the patch hash is fixed. `seal_patch_references.py` writes real
FRONT radiance and measured-height references outside public inputs.
`evaluate_external.py` requires frozen prediction/settings/seed hashes and
refuses a second sealed-test candidate hash. Target marking labels are absent;
normal/slope and stable radiance are qualified proxies, not independent hidden
geometry or unshadowed-albedo ground truth.

No completion method, training run, GPU operation or power change is performed.
Formal numeric R0 preservation tolerances must be frozen by the root before a
quality gate; original alpha >= .98 and residual environment <= .02 road gates
remain in the contract. Engineering pass does not imply quality acceptance.

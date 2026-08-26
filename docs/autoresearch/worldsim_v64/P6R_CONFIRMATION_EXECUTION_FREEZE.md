# P6R exact-once confirmation execution freeze

Date: 2026-08-26

The independently calibrated policy is fixed at nominal coverage 0.40. The model artifact and policy may not be refit or changed.

The exact-once cohort is the metadata-only eight-scene split frozen before independent calibration: `scene-1023`, `scene-1105`, `scene-0903`, `scene-0451`, `scene-0981`, `scene-0537`, `scene-0789`, and `scene-0157`, with 12 fixed targets per scene.

Execution order:

1. Materialize only these eight DriveStudio scenes from the local official tar shards.
2. Feed each complete scene immediately to the blind IR-WM native-sidecar extractor, with at most two GPU workers.
3. Remove the exact temporary raw root after all eight processed scenes complete.
4. Generate target evidence once, then apply the frozen MLP and 0.40 policy once. Report all 96 case losses and four strata; do not select another coverage. Confirmation support requires at most 4/96 losses overall and at most 1/24 in every stratum, matching the registered 0.05 observed-risk target without another confidence-bound selection.

The existing member-to-shard catalog previously discarded entries outside each batch. Following the indexed sparse-access pattern of WIDS and ratarmount, it now retains the union of learned member-to-shard mappings. The current unseen members still require one sequential compressed-tar scan, but future batches reuse the catalog. Scene-ready preprocessing and GPU sidecar consumers overlap the scan so the GPU does not wait for the full cohort barrier.

No hash, checksum, fingerprint, model refit, policy sweep, broad smoke suite, or regression matrix is added. Confirmation target content remains unread until the evidence step.

# WorldSim V6.3 P2 Native Sidecar Interface

- Task: `WS-V63-P2-NATIVE-SIDECAR-01`
- Hypothesis: `WS-V63-H-P2-001`
- Status: `implementation complete / formal preregistered`
- Failure-ledger refs: `V62-F05,V62-F06,V63-F01`; delta: none

V6.3 reuses the V6.2-proven official IR-WM loader and current-state forward, but it no longer deduplicates features only at sampled
query locations. Each target emits full memory-mappable native arrays: logits `200x200x16x17`, BEV latent `200x200x256`, argmax,
entropy, top-two margin and source-valid. All learned arrays are direct outputs or deterministic functions of native logits. Prototype
or fabricated features are absent.

The first formal run is frozen to Tier D (six scenes, 72 targets) plus Tier L method inputs (two scenes, four targets). Tier C/H/T
sidecars remain deferred until their stages are unlocked. The official backbone is frozen; target evidence, occupancy ground truth,
calibration quality, confirmation and test are not read. The worker records official load compatibility, tensor shape/finite state,
fresh memory-mapped reload, runtime and peak memory. This is the necessary P2 contract, not a quality benchmark.

Execution order: one `scene-0071/f017` resource/interface probe, followed directly by the 76-target formal if it passes. No additional
smoke or regression matrix is authorized. The formal run must start from a clean pushed source and requires at least 20 GiB free disk.

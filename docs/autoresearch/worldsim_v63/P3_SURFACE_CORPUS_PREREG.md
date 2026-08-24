# WorldSim V6.3 P3 Surface Corpus Preregistration

- Task: `WS-V63-P3-SURFACE-CORPUS-01`
- Hypothesis: `WS-V63-H-P3-001`
- Status: `implementation ready / probe preregistered`

For every D target, static proposals are native IR-WM occupied volume plus observed OCC, excluding actor envelopes; actor proposals are
each method-visible current/swept envelope with identity preserved. Proposal geometry is declared before topology. Six-connected
boundary extraction and deterministic lexicographic BFS patching cannot create occupied nodes or alter the proposal volume.

Each sharded point payload stores grid/world coordinate, outward normal, surface/patch/proposal identity, real native cell mapping,
method/target evidence, contradiction, per-method-sweep temporal support, ray direction/bundle/order, actor current/swept identity,
authority provenance bits and surface type. Target evidence is legal only as D supervision/statistics and never affects proposal
construction. Registries are semantic JSONL paths without artifact hashes.

The plan-required negative contracts execute once inside the run. One `scene-0071/f017` probe checks counts, patching and resource
shape; success directly unlocks the complete 72-unit corpus with at most two CPU workers. No quality selection, architecture tuning,
calibration, confirmation or test read occurs in P3.

# WorldSim V6.3 P3 Surface Corpus Preregistration

- Task: `WS-V63-P3-SURFACE-CORPUS-01`
- Hypothesis: `WS-V63-H-P3-001`
- Status: `implementation ready / probe preregistered`
- Current revision: `r5 frozen-feature-schema recovery ready`

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

## Probe revision history

- r1=`20260824T150842Z__surface-probe-s20260824-r1` failed before any surface or quality read because unequal target-axis arrays
  (`300/300/40`) were incorrectly passed to `numpy.stack`. No scientific result was produced.
- r2 returns the three coordinate axes independently, keeps out-of-native-range points explicitly invalid, scopes route-support type
  updates to the matching local surface, and validates normals as finite unit vectors. The frozen proposal, topology, patching, cohort and
  gate contracts are unchanged. Its launcher incorrectly pre-created the immutable run leaf and stopped at entry with `FileExistsError`;
  0 units and 0 quality reads. Failure=`V63-F03 resolved`, `V63-F04 resolved`.
- r3 removes only that leaf-directory pre-creation. Source, config and all scientific contracts remain unchanged.
- r3 completed the full probe payload (`191 surfaces/498 patches/152226 points`) but failed the frozen unit-normal gate. The failure is
  confined to symmetric tiny components (101 affected, including 85 singletons) whose exposed-face sum and centroid direction both
  vanish. r4 deterministically orients only those ambiguous points toward the target sensor viewpoint; it preserves every proposal point,
  topology edge, patch assignment and gate. Failure=`V63-F05 resolved`.
- r4 passed its implemented geometry gate with normal-valid 1.0, but a pre-formal comparison against P1 withheld full P3 promotion:
  signed FREE/OCC distances, patch-local coordinates, behind-hit and temporal UNKNOWN were absent, while `ray_hit_order` contained raw
  distance. r5 completes those frozen fields with exact EDT, relative coordinates and normalized per-bundle ordering, retains raw ray
  distance separately, and adds actor observed-hit support. No proposal, label, topology, patch or quality-selection contract changes.
  Failure=`V63-F06 resolved`.

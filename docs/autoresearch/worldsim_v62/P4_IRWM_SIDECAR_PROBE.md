# P4 IR-WM query-aligned sidecar probe

Date: 2026-08-24  
Task: `WS-V62-P4-IRWM-PRIOR-SIDECAR-01`  
Outcome: `done_probe_passed`

## Canonical run

```text
run://worldsim_v62/WS-V62-P4-IRWM-PRIOR-SIDECAR-01/20260824T085956Z__prior-sidecar-probe-s1-r2
```

Source: `ee4ae2d`. Input was the pre-registered new development target `scene-0071/f017`, camera frames `[7,12,17]`,
metadata indices `[1,2,3]`, batch1, and one scene worker.

## Result

- P2 query count: 100,000.
- IR-WM source-valid queries: 97,434; outside-source queries: 2,566.
- Unique 3D prior cells: 27,467, each with 17 FP16 logits.
- Unique 2D BEV cells: 5,633, each with a 256-dimensional FP16 feature.
- Sidecar bytes: 4,002,647.
- Official forward: 1.066 seconds.
- Worker wall: 14.08 seconds; controller wall: 98.29 seconds including initial native-extension startup.
- Peak GPU memory: 4.0496 GiB.

The model emitted multiple occupied semantic classes plus free cells. Logits were finite, mappings were nonempty, and the
sidecar contains the frozen query/source coordinates, input frames, metadata indices, and target LiDAR pose.

The load report had no unexpected keys. Its two missing `pts_bbox_head.transformer.reference_points` parameters are the same
officially deleted keys already resolved in V6.1; they do not participate in the current BEV/occupancy path. P4 records them
without adding a duplicate gate.

## Information boundary

The worker read only query indices/grid metadata from the P2 query archive. Target evidence, occupancy ground truth,
O_method/O_eval, confirmation, and exact-once test content remained unread. Training, future decoding, and planning were not
started. No hash, checksum, fingerprint, content address, or repeated byte-exact build was added.

Decision: run the formal six-scene, 72-target sidecar materialization with at most two scene workers and no further smoke.

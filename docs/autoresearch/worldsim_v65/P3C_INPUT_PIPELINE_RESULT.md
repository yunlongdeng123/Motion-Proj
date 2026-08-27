# P3C independent calibration input pipeline result

Date: 2026-08-27  
Status: inputs ready; formal calibration quality unread

## Frozen cohort and I/O

The cohort remains `scene-0030/0055/0453/0501/1046/1085`. The targeted scan of public-tar shards `1/5/10`
found all required members, so the preregistered same-cohort ten-shard fallback was not used. Canonical preparation:

`run://worldsim_v65/WS-V65-P3C-CALIBRATION-PREPARATION-01/20260827T145000Z__calibration-prep-s0-r1`

- extracted members: 10,689
- wall: 1901.1226s
- processed scenes: 6/6
- temporary raw removed after success: true
- quality read: false

Scene-ready preprocessing was launched as each required shard became available. The preparation parent was resumably
paused while archive children continued, preventing partially built directories from being mistaken for reusable scenes.

## Native sidecars

All six per-scene runs passed with 12 targets per scene. Their walls for `0030/0055/0453/0501/1046/1085` were
`77.05/71.65/51.94/50.90/47.34/50.17s`. Maximum worker peak allocated GPU memory was `4.1314GiB`.

Canonical aggregate:

`run://worldsim_v65/WS-V65-P3C-CALIBRATION-NATIVE-SIDECAR-01/20260827T150000Z__calibration-native-aggregate-s0-r1`

- targets: 72
- output bytes: 3,317,884,541
- inference repeated: false
- target/calibration/confirmation/test read: false

## Evidence

The first four scenes produced 48 partial units while later preprocessing and native GPU work ran. Canonical evidence:

`run://worldsim_v65/WS-V65-P3C-CALIBRATION-EVIDENCE-01/20260827T150000Z__calibration-evidence-s0-r1`

- units: 72
- reused units: 48
- disk bytes: 66,004,741
- wall: 33.8513s
- source-role overlap: 0

The first canonical CLI attempt omitted required `--processed-root`; Python `argparse` exited before run creation or input
read. This is recorded as `V65-F16`. Recovery supplied only the config-frozen standard processed root and did not change
the run id, cohort, units, reuse, seed, target, calibrator, or gates.

## Next read

Apply `sigmoid(1.7039771080 * logit(Qmean) - 0.4792216420)` once, without refit. This milestone supports input
capability only and makes no calibration, conformal, admission, planning, or safety claim.

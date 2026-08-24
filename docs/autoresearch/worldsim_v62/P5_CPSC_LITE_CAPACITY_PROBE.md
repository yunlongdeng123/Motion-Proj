# P5 CPSC-Lite capacity probe

Date: 2026-08-24  
Task: `WS-V62-P5-CPSC-LITE-TRAIN-01`  
Outcome: `done_capacity_probe_passed`

## Canonical run

```text
run://worldsim_v62/WS-V62-P5-CPSC-LITE-TRAIN-01/20260824T092410Z__cpsc-lite-capacity-s0-r1
```

Source: `1579780`. The probe used the first three scene-0071 train units, scene-0450/f017 as one selection unit, and stopped
after the frozen maximum of eight optimizer steps.

## Contract result

- Parameters: 608,366.
- Prior/query feature dimensions: 278/13.
- Precision: FP16; query batch 16,384; gradient accumulation2.
- Best/final selection objective: 2.136239; all reported losses finite.
- Peak GPU memory: 0.37242 GiB.
- Wall: 4.914 seconds.
- Hard constraints: 53,106 constrained rows, zero violations.
- BEST and FINAL checkpoints were produced.

The learned probe output was not all UNKNOWN: FREE/OCC/UNKNOWN fractions were 0.3652/0.4576/0.1773. Its target accuracy was
0.4233 versus 0.3713 for projection-only, and safe-OCC retention was 0.9569 versus 0.9502. Hidden-FREE false-OCC was 0.2680
versus 0.2616 for projection-only, so eight steps do not establish the main risk improvement. These figures are recorded only
as a non-degenerate capacity diagnostic and are not used to change the loss, split, seed, threshold, or model.

Target evidence and dropout evidence were supervision only, never method inputs. IR-WM was not resident. Legacy O_eval,
confirmation, and exact-once test remained unread, and no hash/checksum/fingerprint machinery was added.

Decision: run the frozen formal seed0 configuration over all 48 train units and 24 scene-disjoint selection units without an
additional smoke stage.

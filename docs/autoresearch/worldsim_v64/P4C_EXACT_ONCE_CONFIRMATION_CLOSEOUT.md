# P4C Exact-Once Conditional Confirmation Closeout

Date: 2026-08-26

## Outcome

The frozen conditional compiler mapping is supported on its fresh exact-once confirmation:

- canonical run: `run://worldsim_v64/WS-V64-P4C-CONDITIONAL-EXACT-ONCE-CONFIRMATION-01/20260826T173000Z__exact-once-confirmation-s0-r1`;
- verdict: `supported_exact_once_conditional_confirmation`;
- scenes/cases: 8/96;
- C0 global coverage: 0.39994442456469775;
- C0 failures: 0/96;
- M0 conditional coverage: 0.4749608367012231;
- M0 failures: 0/96;
- absolute coverage uplift: 0.07501641213652532;
- M0 construction/night/rain/vulnerable-transit failures: 0/24, 0/24, 0/24, 0/24.

The preregistered minimum uplift, maximum overall failures, and maximum per-stratum failures gates all passed.

## Frozen comparison

C0 used nominal global coverage 0.40. M0 used nominal coverage 0.40 for rain and 0.50 for construction, night, and vulnerable transit. Both arms used the same frozen full-native MLP, risk order, conflict threshold, fresh evidence, and 96-case denominator. There was no model refit, coverage sweep, run-time policy selection, second mapping, or second confirmation.

GPU scoring took 12.174508595839143 seconds and peak RSS was 0.7957801818847656 GiB.

## Claim boundary

This result supports the frozen conditional coverage map for the observed fresh case-risk contract. It is not a real-world safety claim, a calibrated population-risk bound, or a downstream simulation/compiler integration result.

No hash, checksum, fingerprint, smoke matrix, or regression matrix was added. Failure delta is none. V64-F18 remains closed as a pre-quality input recovery.

## Operational tail

Scientific scoring is complete. Remaining work is operational only: resume both paused prep controllers to EOF, remove their recoverable temporary raw directories through controller ownership, and semantic-union the isolated replacement member-to-shard catalog into the persistent superset catalog.

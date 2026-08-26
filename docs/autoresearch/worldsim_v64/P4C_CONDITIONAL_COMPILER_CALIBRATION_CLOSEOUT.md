# P4C conditional compiler calibration closeout

Date: 2026-08-26

Canonical run: `run://worldsim_v64/WS-V64-P4C-CONDITIONAL-COMPILER-01/20260826T160000Z__conditional-compiler-s0-r1`.

On the already-consumed eight-scene, 96-case calibration set, C0 global 0.40 achieved mean realized coverage 0.3999668 with zero failures. The frozen M0 map (rain 0.40; the other three strata 0.50) achieved mean realized coverage 0.4749773 with zero failures, an absolute uplift of 0.0750105. Every M0 stratum remained at 0/24 failures. All four preregistered candidate gates passed.

GPU scoring took 13.3357 seconds with peak RSS 0.7901 GiB. The model was not refit and the run did not select a mapping or coverage. Verdict: `supported_conditional_candidate`.

This is a calibration replay over previously read target quality, so it only freezes a candidate. It does not establish fresh conditional-compiler generalization, real-world safety, or downstream simulation performance. The separately frozen new confirmation cohort remains unread.

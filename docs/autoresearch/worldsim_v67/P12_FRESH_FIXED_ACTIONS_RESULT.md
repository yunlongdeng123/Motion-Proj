# V6.7 P12 fresh fixed actions result

Canonical=`run://worldsim_v67/WS-V67-P12-FRESH-FIXED-ACTIONS-01/20260828T114900Z__fresh-actions-s0-r1`;
verdict=`supported_v67_fixed_budget_action_set`; 6/6 gates pass.

At the frozen 469/938 budget, L0 handles 341/563 conflicts (`0.605684` reduction), q0 handles 335 (`0.595027`), and
the oracle handles 469 (`0.833037`). Actor retention, zero removal, zero hazard shift, emitted fraction and scene yield are
`1/0/0/0.5/1`. L0 is at least 0.5 in five scenes and 0.4324 in scene-0373; no post-hoc scene gate is added.

The first invocation serialized the PowerShell command substitution as a literal backslash run directory. The completed
directory was atomically renamed to the canonical locator above without repeating evaluation (`V67-F03 resolved`). P13 now
applies the exact P4R/P8 physical rule once.

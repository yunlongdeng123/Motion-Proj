# P42 Hybrid Conditioned Action Result

Canonical run: `run://worldsim_v67/WS-V67-P42-HYBRID-CONDITIONED-ACTION-01/20260828T214000Z__hybrid-conditioned-action-s0-r1`.

The case-centered action refinement trained on 2,412 conditioned cases and 27,087 action rows from nine domains. Final
soft selected cost was `0.057805`, residual RMS `0.037035`, and domain losses ranged from `0.064454` to `0.110721`.

On action-target-untouched P6R, the hybrid selected exactly `294/294` actions, covered 55/78 cases (`0.705128`), retained
minimum group coverage `0.50`, and improved or tied all seven scenes. Reduction was `0.800132`, versus `0.792220` for
frozen P31 and `0.550120` for fixed P20: deltas `+0.007912` and `+0.250012`. All gates passed; verdict
`supported_hybrid_conditioned_action`.

The supported mechanism is compositional: P31 preserves cross-case allocation while the learned centered residual refines
within-case action order. Evidence is globally consumed and does not establish fresh-population planning or safety performance.

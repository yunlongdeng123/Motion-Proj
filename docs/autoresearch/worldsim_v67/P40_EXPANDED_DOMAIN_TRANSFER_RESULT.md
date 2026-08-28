# P40 Expanded-Domain Transfer Result

Canonical run: `run://worldsim_v67/WS-V67-P40-EXPANDED-DOMAIN-TRANSFER-01/20260828T204500Z__expanded-domain-transfer-s0-r1`.

Frozen P39 selected exactly `363/363` P10R4 actions, covered 73/96 cases (`0.760417`), retained minimum group coverage
`0.583333`, and improved or tied all eight scenes. Frozen P31 covered 68/96 cases (`0.708333`).

P39 reduction was `0.654575`, below P31's `0.674930` by `-0.020355`; fixed P20 was `0.281451`. The exact, group,
and scene gates passed but decision improvement failed. Verdict `rejected_fourth_cohort_expanded_domain_transfer`.

This repeats the cross-cohort separation between broader case coverage and lower selected cost. No model or gate adjustment
is made from this read; the globally consumed cohort does not support a fresh-population or safety claim.

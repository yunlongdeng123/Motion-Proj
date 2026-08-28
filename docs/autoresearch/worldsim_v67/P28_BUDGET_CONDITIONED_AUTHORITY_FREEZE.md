# P28 Budget-Conditioned Authority Freeze

P28 adds the requested selected-action fraction as the eighth case feature. The same bounded offset model trains jointly at
fractions `0.25` and `0.50` across ten domains. P10R4 is excluded from P28 training; after model freeze, its existing cache is
read once at the held-out fraction `1/3`.

The selected action count must exactly equal the fixed one-third per-case total. Each of the four pre-existing P10R4 strata and
the cohort globally must cover at least 50% of evaluable cases; P20 within-case order is immutable and at most eight actions may
be selected per case.

Gates require exact total budget, global and per-group coverage at least `0.50`, cost reduction at least `0.40`, delta over fixed
P20 at least `0.10`, and at least six non-increasing scenes. No fraction, architecture, offset, coverage, group, loss, or gate
sweep is allowed. P10R4 is globally consumed, so the claim is held-out-budget transfer only. No hash/checksum/fingerprint.

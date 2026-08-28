# P39 Expanded-Domain Conditioned Action Result

Canonical run: `run://worldsim_v67/WS-V67-P39-EXPANDED-DOMAIN-CONDITIONED-ACTION-01/20260828T202000Z__expanded-domain-topk-s0-r1`.

Six development domains and three budgets produced 1,656 conditioned cases and 18,300 action rows. Final soft selected
cost was `0.061912`, residual RMS `0.038086`, and domain losses ranged from `0.084726` to `0.114948`.

On action-target-untouched P3C at `(budget=1/3,H=2s)`, P39 selected exactly `238/238` actions, covered 46/60 cases
(`0.766667`), retained minimum group coverage `0.708333`, and improved or tied all five scenes. Reduction was `0.724052`,
versus `0.710835` for P31 and `0.368126` for fixed P20: deltas `+0.013217` and `+0.355925`. All gates passed;
verdict `supported_expanded_domain_conditioned_action`.

This supports development-domain diversity as the successful recovery from P37/P38 cross-cohort failures. The cohort is
globally consumed and the result is not a fresh-population, planning, closed-loop, collision, or safety claim.

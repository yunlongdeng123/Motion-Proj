# P40 Expanded-Domain Transfer Freeze

P40 loads the exact P39 model and normalizer. P10R4 action targets were excluded from P20, P31, and P39 training; the
cache is globally consumed by an earlier tail-risk task, so this read supports method transfer only.

At `(budget=1/3,H=2s)`, P40 uses the same exact-total and four scene-pair group constraints as prior authority experiments.
The comparator is frozen P31. Gates: exact total; minimum group coverage `>=0.50`; reduction delta over P31 `>=0.005`;
at least six of eight non-increasing scenes. No training, refit, temperature, model, or gate sweep. No planning or safety claim.

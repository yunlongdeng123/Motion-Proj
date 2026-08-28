# V6.7 P25 coverage-constrained budget freeze

P25 freezes P20 within-case action order and trains a new nine-domain bounded case offset. At confirmation the total selected
action count exactly equals the fixed 0.25-per-case baseline. A case may receive zero to six actions, but at least half of evaluable
cases must receive authority.

The model is frozen before materializing V64 P4C scenes `0992/1101/0454/1102/0876/0895/0321/0813`. Gates require exact total
budget, case coverage >=0.50, cost reduction >=0.60, gain over fixed P20 >=0.10 and six non-increasing scenes.

One run only; no coverage, maximum-action, offset, architecture, fraction or gate sweep. This is selective task authority at a
fixed action budget, not collision avoidance, planning policy, closed loop or safety.

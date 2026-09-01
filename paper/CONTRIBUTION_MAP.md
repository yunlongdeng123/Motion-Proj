# HARP-3D paper contribution map

## Central claim

World validity is not world difficulty. The method must repair evidence-inconsistent physical geometry without deleting legitimate hazardous Actors.

| Contribution | Paper location | Code-facing interface | Primary evidence | Claim boundary |
|---|---|---|---|---|
| C1 Hazard-preserving physical compilation | Sec. 3.1, Fig. 1/3, Table 1 | `physical_compiler.py`, `sceneir_adapter.py` | ray/free-space, surface error, temporal jitter, Actor/ID/TTC retention | no claim that Gaussian opacity is occupancy; unknown is not free |
| C2 Validity--hazard factorization | Sec. 3.2, Fig. 4, Table 2 | `validity_hazard.py` | four-quadrant pairs, clean-hazard false artifact, cross-probe leakage | paired invariance, not global statistical independence |
| C3 Continuous task reliability | Sec. 3.3--3.4, Fig. 5, Table 3 | `actor_reliability.py`, `boundary_cost_density.py`, `runtime_surface.py` | NLL/Brier/calibration, monotonicity, nuScenes-to-AV2 degradation | empirical reliability, not a formal road-safety guarantee |

## Causal chain

`physical artifact repair -> hazard preservation -> task-conditioned reliability -> minimal downstream utility`

## Result ownership

- `paper/results/results_macros.tex` is the single numeric interface for the manuscript.
- P1--P3 own C1 physical and preservation metrics.
- P4 owns paired factorization metrics.
- P5--P6 own density, multi-horizon, and AV2 zero-shot metrics.
- P8 owns final exact-once numbers; earlier development values cannot overwrite them.

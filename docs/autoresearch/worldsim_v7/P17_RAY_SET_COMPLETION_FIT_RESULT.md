# P17 Ray-Set Completion Fit Result

Date: 2026-09-02

Canonical: `run://worldsim_v7/WS-V7-P17-RAY-SET-COMPLETION-FIT-01/20260903T084500Z__ray-set-fit-s71701-r1`

The joint first-return objective is directionally effective but fails the frozen Pareto decision. On 228 disjoint nuScenes test
Actors, completion coverage falls to 71.43%. Total new-early decreases from `.96618%` to `.91077%`; hazardous new-early decreases
from `1.43622%` to `1.37597%`, and clear new-early decreases from `.44250%` to `.39249%`. This supports the ray-set diagnosis of
`V7-F24`.

However, mean Chamfer worsens from `.1945868 m` to `.1994111 m`, composite gain falls from `.0567090 m` to `.0518848 m`, and
new hits fall from 39,255 to 34,242. Pure first-return depth supervision therefore learns to suppress risky completion but has no
direct responsibility to preserve the complete 3D surface.

Register `V7-F25`; do not consume fresh AV2 and do not sweep the hard threshold, ray count, epochs, or seed. The sole recovery is
the hybrid supervision used by differentiable occupancy work: retain the identical model/ray renderer and add a differentiable
expected bidirectional Chamfer term. Normalize ray and Chamfer terms by their frozen always-COMPLETE source values so no hand-tuned
mixture coefficient is introduced.

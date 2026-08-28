# P31 Joint Budget-Horizon Authority Result

Canonical: `run://worldsim_v67/WS-V67-P31-JOINT-BUDGET-HORIZON-AUTHORITY-01/20260828T174000Z__joint-budget-horizon-s0-r1`.

The model trained 788 joint-condition rows and evaluated the unseen pair `(budget=1/3,H=1.5s)` on 66 P10X cases. It selected
exactly 236 actions, covered 48 cases (`72.7273%`), and maintained minimum context coverage `58.3333%`.

Reduction was `0.690636`, versus fixed P20 `0.190718` (`+0.499918`). Five of six scenes were non-increasing and all six gates
pass. Wall/GPU/RSS=`32.587s/0.01705GiB/1.1803GiB`. Claim: consumed-cohort joint-condition mechanism only.

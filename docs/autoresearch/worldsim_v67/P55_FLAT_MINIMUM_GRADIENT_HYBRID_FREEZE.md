# P55 Flat-Minimum Gradient Hybrid Freeze

P55 keeps P53 data, architecture, double endpoint anchor, gradient-direction penalty, four budgets, optimizer, losses, seed,
and 6,000 epochs. The sole method change is an equal parameter average over the fixed last 20% of training (epochs
4,800--5,999; 1,200 checkpoints). There is no validation-based averaging window, learning-rate change, or sweep.

P10R4 H=`.8s` is materialized in parallel: `984/1152` eligible actions from 96 source cases. At budget `.375`, the same
formal read evaluates P55 and frozen non-averaged P53 with identical P31 offsets. Gates are exact total, minimum group `.50`,
delta over P31 `+.005`, delta over P53 `+.002`, and six non-increasing scenes.

This is SWAD/SWA-inspired fixed-tail averaging, not an equivalence claim. No averaging-window/schedule/model/loss/gate sweep;
no fresh-population, collision, planning, policy, closed-loop, or safety claim; no hash/checksum/fingerprint.

References: https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html ;
https://www.auai.org/uai2018/proceedings/papers/313.pdf .

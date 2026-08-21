#!/usr/bin/env python
"""运行 V6 PT3 冻结风险策略干预分布外实验。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.pt3_risk_policy_robustness import main


if __name__ == "__main__":
    raise SystemExit(main())

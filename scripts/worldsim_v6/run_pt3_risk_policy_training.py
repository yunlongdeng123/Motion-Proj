#!/usr/bin/env python
"""运行 V6 PT3 二维 typed-edit 风险策略训练实验。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.pt3_risk_policy_training import main


if __name__ == "__main__":
    raise SystemExit(main())

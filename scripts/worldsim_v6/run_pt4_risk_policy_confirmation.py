#!/usr/bin/env python
"""运行 V6 PT4 冻结风险策略一次性 confirmation。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.pt4_risk_policy_confirmation import main


if __name__ == "__main__":
    raise SystemExit(main())

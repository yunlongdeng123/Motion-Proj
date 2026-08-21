#!/usr/bin/env python
"""运行 V6 PT1 verified compiled episode 回归拦截效用实验。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.pt1_regression_utility import main


if __name__ == "__main__":
    raise SystemExit(main())

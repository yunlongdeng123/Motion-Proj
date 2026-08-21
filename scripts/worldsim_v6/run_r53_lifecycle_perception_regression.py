#!/usr/bin/env python3
"""运行 WorldSim V6 R53 lifecycle perception regression。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r53_lifecycle_perception_regression import main


if __name__ == "__main__":
    raise SystemExit(main())

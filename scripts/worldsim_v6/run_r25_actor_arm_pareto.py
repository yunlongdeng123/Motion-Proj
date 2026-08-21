#!/usr/bin/env python
"""运行 V6 R25 actor arm Pareto evaluation。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r25_actor_arm_pareto import main


if __name__ == "__main__":
    raise SystemExit(main())

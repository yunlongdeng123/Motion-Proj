#!/usr/bin/env python
"""运行 V6 R35 actor rigid trajectory compiler。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r35_actor_rigid_trajectory_compiler import main


if __name__ == "__main__":
    raise SystemExit(main())


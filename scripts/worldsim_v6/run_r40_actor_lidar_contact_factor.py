#!/usr/bin/env python
"""运行 V6 R40 actor logged-LiDAR contact factor。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r40_actor_lidar_contact_factor import main


if __name__ == "__main__":
    raise SystemExit(main())


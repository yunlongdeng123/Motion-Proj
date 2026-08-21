#!/usr/bin/env python3
"""WorldSim V6 R13 actor sensor/perception 正式入口。"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from motion_proj.worldsim_v6.r13_actor_sensor_perception import main


if __name__ == "__main__":
    raise SystemExit(main())

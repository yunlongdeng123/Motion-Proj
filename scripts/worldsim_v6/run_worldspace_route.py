#!/usr/bin/env python3
"""WorldSim V6 R13 world-space route 正式入口。"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from motion_proj.worldsim_v6.r13_worldspace_route import main


if __name__ == "__main__":
    raise SystemExit(main())

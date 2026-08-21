#!/usr/bin/env python3
"""运行 WorldSim V6 R41 factor fusion 正式实验。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r41_actor_edit_factor_fusion import main


if __name__ == "__main__":
    raise SystemExit(main())

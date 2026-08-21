#!/usr/bin/env python3
"""运行 WorldSim V6 R61 因子筛选后的第二 actor sensor 实验。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r61_selected_second_actor_sensor import main


if __name__ == "__main__":
    raise SystemExit(main())

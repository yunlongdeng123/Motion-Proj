#!/usr/bin/env python
"""运行 V6 PT8 纵向 closed-loop utility 实验。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.pt8_closed_loop_utility import main


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""运行 WorldSim V6 R58 second actor counterfactual sensor。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r58_second_actor_translation_sensor import main


if __name__ == "__main__":
    raise SystemExit(main())

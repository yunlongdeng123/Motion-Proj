#!/usr/bin/env python3
"""Run WorldSim V6 R81 three-actor joint interaction."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r81_three_actor_joint_interaction import main


if __name__ == "__main__":
    raise SystemExit(main())

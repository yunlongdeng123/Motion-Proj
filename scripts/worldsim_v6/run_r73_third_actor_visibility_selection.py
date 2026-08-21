#!/usr/bin/env python3
"""Run WorldSim V6 R73 third-actor visibility selection."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r73_third_actor_visibility_selection import main


if __name__ == "__main__":
    raise SystemExit(main())

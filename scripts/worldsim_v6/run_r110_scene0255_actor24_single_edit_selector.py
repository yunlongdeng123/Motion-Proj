#!/usr/bin/env python3
"""Run WorldSim V6 R110 actor24-only selector cell."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r110_scene0255_actor24_single_edit_selector import main


if __name__ == "__main__":
    raise SystemExit(main())

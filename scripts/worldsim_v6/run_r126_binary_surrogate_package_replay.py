#!/usr/bin/env python3
"""Run WorldSim V6 R126 binary surrogate package replay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r126_binary_surrogate_package_replay import main


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run WorldSim V6 R88 selective perception runtime gate."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r88_selective_perception_runtime_gate import main


if __name__ == "__main__":
    raise SystemExit(main())

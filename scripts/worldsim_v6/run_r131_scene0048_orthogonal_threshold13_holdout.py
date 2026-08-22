#!/usr/bin/env python3
"""Run WorldSim V6 R131 scene0048 orthogonal threshold13 holdout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r131_scene0048_orthogonal_threshold13_holdout import main


if __name__ == "__main__":
    raise SystemExit(main())

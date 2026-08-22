#!/usr/bin/env python3
"""Run WorldSim V6 R128 scene0230 orthogonal surrogate holdout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r128_scene0230_orthogonal_surrogate_holdout import main


if __name__ == "__main__":
    raise SystemExit(main())

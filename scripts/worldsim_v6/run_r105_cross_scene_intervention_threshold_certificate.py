#!/usr/bin/env python3
"""Run WorldSim V6 R105 cross-scene/intervention threshold certificate."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r105_cross_scene_intervention_threshold_certificate import main


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run WorldSim V6 R82 three-actor scene package bake."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v6.r82_three_actor_scene_package_bake import main


if __name__ == "__main__":
    raise SystemExit(main())

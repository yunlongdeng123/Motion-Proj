#!/usr/bin/env python3
"""WorldSim V6 R12 LogSim 正式入口。"""

import sys
from pathlib import Path


# 允许直接以脚本路径启动，避免依赖调用方预设 PYTHONPATH。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from motion_proj.worldsim_v6.r12_logsim import main


if __name__ == "__main__":
    raise SystemExit(main())

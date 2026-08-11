#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from motion_proj.worldsim_v4.p0_contract import run_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结并验证 WorldSim V4 P0 paper-first 合同")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(args.config, args.run_dir, project_root=args.project_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

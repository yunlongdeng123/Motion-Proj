#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from motion_proj.worldsim_v33.source_audit import run_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结并验证 WorldSim V3.3 P0 source 审计")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--verify-large-assets",
        action="store_true",
        help="重新计算 V3.2 大型 canonical 资产 SHA-256",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(
        args.config,
        args.run_dir,
        verify_large_assets=args.verify_large_assets,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

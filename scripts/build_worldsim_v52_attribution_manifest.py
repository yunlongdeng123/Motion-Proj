#!/usr/bin/env python3
"""生成 V5.2.1 人工归因与后续回测 manifest。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v521.attribution import build_attribution_package


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-cases", type=Path, required=True)
    parser.add_argument("--badcase-registry", type=Path, required=True)
    parser.add_argument("--matched-frame-registry", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_attribution_package(
        review_cases_path=args.review_cases,
        badcase_registry_path=args.badcase_registry,
        matched_frame_registry_path=args.matched_frame_registry,
        annotation_config_path=args.annotations,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

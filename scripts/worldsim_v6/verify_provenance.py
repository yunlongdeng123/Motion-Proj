#!/usr/bin/env python3
"""fresh-process provenance package verifier。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motion_proj.worldsim_v6.provenance import verify_provenance_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--sceneir-package", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_provenance_package(args.package, args.sceneir_package),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

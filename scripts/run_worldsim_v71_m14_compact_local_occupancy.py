"""Run the frozen M14 compact local occupancy field protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_worldsim_v71_m13_local_signed_field import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

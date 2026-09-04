"""Run the frozen M8 checkpoint through the generic scene-ready AV2 evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_worldsim_v71_m7_av2_zero_shot as generic_runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            generic_runner.run(
                args.config.resolve(), args.repo_root.resolve(), args.run_id
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


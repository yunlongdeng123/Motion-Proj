#!/usr/bin/env python3
"""Validate and aggregate completed model-case result contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_proj.cfbench.evaluate import aggregate_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_paths = sorted(args.results_root.glob("**/result.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    aggregate = aggregate_results(results)
    aggregate["source_files"] = [str(path) for path in result_paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result_count": len(results), "models": sorted(aggregate["models"])}, indent=2))


if __name__ == "__main__":
    main()

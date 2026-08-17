#!/usr/bin/env python3
"""Read-only audit of frozen Stage E KNN edge-length denominators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.protocol import ProtocolError, load_yaml, sha256_file


def audit(config_path: Path) -> dict[str, object]:
    config = load_yaml(config_path)
    scenes = []
    for scene in config["historical_inputs"]:
        scene_name = scene["scene"]
        b3_path = Path(scene["b3_unary"]["path"])
        if sha256_file(b3_path) != scene["b3_unary"]["sha256"]:
            raise ProtocolError(f"E0a edge audit B3 drift: {scene_name}")
        edge_path = Path(scene["v5_graph_run"]["path"]) / "artifacts/graph/edges.npz"
        expected_edge_sha = scene["v5_graph_run"]["files"]["artifacts/graph/edges.npz"]
        if sha256_file(edge_path) != expected_edge_sha:
            raise ProtocolError(f"E0a edge audit edge drift: {scene_name}")
        with np.load(b3_path, allow_pickle=False) as table:
            center = np.asarray(table["center"], dtype=np.float64)
        with np.load(edge_path, allow_pickle=False) as table:
            source = np.asarray(table["source_gaussian_id"], dtype=np.int64)
            target = np.asarray(table["target_gaussian_id"], dtype=np.int64)
        distance = np.linalg.norm(center[source] - center[target], axis=1)
        finite = np.isfinite(distance)
        positive = distance[finite & (distance > 0.0)]
        if positive.size == 0:
            raise ProtocolError(f"E0a edge audit has no positive edge: {scene_name}")
        scenes.append(
            {
                "scene": scene_name,
                "edge_count": int(distance.size),
                "finite_edge_count": int(np.count_nonzero(finite)),
                "zero_edge_count": int(np.count_nonzero(finite & (distance == 0.0))),
                "nonfinite_edge_count": int(np.count_nonzero(~finite)),
                "minimum_positive_edge_length_m": float(positive.min()),
                "positive_edge_quantiles_m": {
                    "q50": float(np.quantile(positive, 0.5, method="linear")),
                    "q75": float(np.quantile(positive, 0.75, method="linear")),
                    "q90": float(np.quantile(positive, 0.9, method="linear")),
                },
            }
        )
    return {
        "status": "passed",
        "conclusion": "frozen_knn_edge_lengths_audited_with_zero_length_edges_explicit",
        "scene_count": len(scenes),
        "scenes": scenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_d_progressive_preflight_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.config.resolve())
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

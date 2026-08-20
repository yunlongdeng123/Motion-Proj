#!/usr/bin/env python3
"""在隔离 model view 中重渲染 AD-GS Discovery adapter。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adgs-python", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--model-source", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("partition") != "discovery" for row in rows):
        raise RuntimeError("records 必须为非空 Discovery-only")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    isolated_model = output / "model"
    isolated_model.mkdir()
    (isolated_model / "point_cloud").symlink_to((args.model_source.resolve() / "point_cloud"), target_is_directory=True)
    bundle_paths = sorted((args.model_source.resolve() / "point_cloud" / "iteration_60000").glob("*"))
    before = {path.name: sha256_file(path) for path in bundle_paths if path.is_file()}
    started = time.monotonic()
    command = [
        str(args.adgs_python.resolve()),
        "render.py",
        "--config",
        str(args.config.resolve()),
        "--model_path",
        str(isolated_model),
        "--source_path",
        str(args.adapter.resolve()),
        "--iteration",
        "60000",
        "--skip_train",
        "--quiet",
    ]
    completed = subprocess.run(
        command,
        cwd=args.source_root.resolve(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output / "render.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"AD-GS render 失败，exit={completed.returncode}")
    after = {path.name: sha256_file(path) for path in bundle_paths if path.is_file()}
    if before != after:
        raise RuntimeError("AD-GS checkpoint bundle before/after SHA 漂移")
    render_dir = isolated_model / "test" / "ours_60000" / "renders"
    predictions = sorted(render_dir.glob("*.png"))
    if len(predictions) != len(rows):
        raise RuntimeError(f"AD-GS prediction count {len(predictions)} != {len(rows)}")
    render_rows = []
    for sequence, (row, prediction) in enumerate(zip(rows, predictions)):
        render_rows.append(
            {
                "sequence": sequence,
                "scene": row["scene"],
                "frame": int(row["frame"]),
                "camera": int(row["camera"]),
                "partition": "discovery",
                "prediction_path": str(prediction.resolve()),
                "prediction_sha256": sha256_file(prediction),
            }
        )
    (output / "RENDER_MAP.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in render_rows), encoding="utf-8"
    )
    audit = {
        "schema": "worldsim_v521_adgs_render_v1",
        "source_root": str(args.source_root.resolve()),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "model_source": str(args.model_source.resolve()),
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": after,
        "adapter": str(args.adapter.resolve()),
        "quality_partition": "discovery",
        "confirmation_original_pixels_decoded": 0,
        "views": len(render_rows),
        "render_seconds": time.monotonic() - started,
        "command": command,
    }
    (output / "RENDER_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

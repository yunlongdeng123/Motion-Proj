#!/usr/bin/env python3
"""用官方 GaussianDWM loader 审计公开样例与兼容包装。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gaussiandwm_cvpr.data.gauss_normalizer import GaussNormalizer


def _load(path: Path) -> dict[str, object]:
    try:
        tensor = GaussNormalizer().load_and_normalize(
            [str(path)], scene_idx=None, frame_idx=None
        )
    except Exception as error:  # 审计需要保留官方 loader 的实际拒绝类型。
        return {
            "accepted": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "accepted": True,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--compatible", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_version": "worldsim_v75_gaussiandwm_sample_contract_v1",
        "gpu_operations_run": False,
        "raw": {"path": str(args.raw.resolve()), **_load(args.raw.resolve())},
        "compatible": {
            "path": str(args.compatible.resolve()),
            **_load(args.compatible.resolve()),
        },
    }
    report["contract_mismatch_confirmed"] = (
        not report["raw"]["accepted"]
        and report["compatible"]["accepted"]
        and report["compatible"].get("shape") == [16000, 14]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

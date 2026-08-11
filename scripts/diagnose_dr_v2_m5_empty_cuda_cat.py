#!/usr/bin/env python
"""复现 DriveStudio 实例初始化中的 CUDA 空 tensor 拼接行为。"""
from __future__ import annotations

import json

import torch


def main() -> None:
    rows = []
    for count in (1, 16, 64, 128, 196, 256, 512, 1024):
        try:
            output = torch.cat(
                [torch.empty((0, 3), device="cuda") for _ in range(count)], dim=0
            )
            torch.cuda.synchronize()
            rows.append(
                {"count": count, "status": "done", "shape": list(output.shape)}
            )
        except RuntimeError as error:
            rows.append(
                {"count": count, "status": "failed", "error": str(error)}
            )
            torch.cuda.synchronize()
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()

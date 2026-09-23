#!/usr/bin/env python3
"""只读取 safetensors 头部，生成不触发 GPU/权重反序列化的结构证据。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--expected-tensor-count", type=int)
    parser.add_argument("--required-prefix", action="append", default=[])
    parser.add_argument("--sample-limit", type=int, default=64)
    args = parser.parse_args()

    path = args.path.expanduser().resolve(strict=True)
    marker = Path(f"{path}.aria2")
    if marker.exists():
        raise SystemExit(f"拒绝检查仍在下载的文件：{marker}")
    size_bytes = path.stat().st_size
    if args.expected_size is not None and size_bytes != args.expected_size:
        raise SystemExit(
            f"文件大小不符：{size_bytes} != {args.expected_size}"
        )
    if args.sample_limit < 0:
        raise SystemExit("--sample-limit 不能为负数")

    from safetensors import safe_open

    dtype_tensor_counts: Counter[str] = Counter()
    dtype_element_counts: Counter[str] = Counter()
    rank_counts: Counter[int] = Counter()
    top_level_prefix_counts: Counter[str] = Counter()
    samples: list[dict[str, object]] = []
    total_elements = 0
    with safe_open(path, framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        metadata = tensors.metadata()
        for index, key in enumerate(keys):
            view = tensors.get_slice(key)
            shape = list(view.get_shape())
            dtype = str(view.get_dtype())
            elements = math.prod(shape)
            total_elements += elements
            dtype_tensor_counts[dtype] += 1
            dtype_element_counts[dtype] += elements
            rank_counts[len(shape)] += 1
            top_level_prefix_counts[key.split(".", 1)[0]] += 1
            if index < args.sample_limit:
                samples.append(
                    {
                        "key": key,
                        "dtype": dtype,
                        "shape": shape,
                        "elements": elements,
                    }
                )

    if args.expected_tensor_count is not None and len(keys) != args.expected_tensor_count:
        raise SystemExit(
            f"tensor 数量不符：{len(keys)} != {args.expected_tensor_count}"
        )
    missing_prefixes = [
        prefix
        for prefix in args.required_prefix
        if not any(key == prefix or key.startswith(prefix + ".") for key in keys)
    ]
    if missing_prefixes:
        raise SystemExit(f"缺少必需 key prefix：{missing_prefixes}")

    report = {
        "schema_version": "safetensors_metadata_inspection_v1",
        "gpu_operations_run": False,
        "tensor_payloads_loaded": False,
        "path": str(path),
        "size_bytes": size_bytes,
        "tensor_count": len(keys),
        "total_elements": total_elements,
        "dtype_tensor_counts": dict(sorted(dtype_tensor_counts.items())),
        "dtype_element_counts": dict(sorted(dtype_element_counts.items())),
        "rank_counts": {str(key): value for key, value in sorted(rank_counts.items())},
        "top_level_prefix_counts": dict(sorted(top_level_prefix_counts.items())),
        "required_prefixes": args.required_prefix,
        "metadata": metadata,
        "sample_tensors": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "size_bytes": size_bytes,
                "tensor_count": len(keys),
                "total_elements": total_elements,
                "dtype_tensor_counts": report["dtype_tensor_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

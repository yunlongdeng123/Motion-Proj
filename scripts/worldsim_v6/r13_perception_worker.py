#!/usr/bin/env python3
"""隔离执行 R13 actor-edit 冻结语义感知。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_model(root: Path) -> torch.nn.Module:
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(
        weights=None, weights_backbone=None, num_classes=19, aux_loss=True
    )
    # 冻结归档主 head 为 19 类，auxiliary head 保留 torchvision 的 21 类结构。
    model.aux_classifier[4] = torch.nn.Conv2d(256, 21, kernel_size=1)
    checkpoint = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    return model.eval().to("cuda")


def _load_rgb(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        image = np.asarray(archive["rgb"])
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = image.astype(np.float32)
    # renderer 浮点输出允许轻微超出 1；只有明显 0--255 输入才不做放大。
    if float(np.nanmax(image)) <= 2.0:
        image = image * 255.0
    return np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)


def _predict(model: torch.nn.Module, image: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    tensor = ((tensor - mean) / std).to("cuda")
    with torch.inference_mode():
        logits = model(tensor)["out"]
    return logits.argmax(dim=1)[0].to(torch.int16).cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260821)
    torch.cuda.manual_seed_all(20260821)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = _load_model(args.model_root)
    rows: list[dict[str, Any]] = []
    for line in args.index.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        item = json.loads(line)
        labels = _predict(model, _load_rgb(Path(item["render_path"])))
        relative = f"{item['case_id']}__repeat{item['repeat_index']}.npy"
        output_path = args.output_dir / relative
        np.save(output_path, labels, allow_pickle=False)
        rows.append(
            {
                "case_id": item["case_id"],
                "repeat_index": item["repeat_index"],
                "label_path": relative,
                "label_array_sha256": _array_sha256(labels),
                "label_file_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "unique_label_count": int(np.unique(labels).size),
            }
        )
    _write_jsonl(args.output_dir / "PERCEPTION_OUTPUTS.jsonl", rows)
    _write_json(
        args.output_dir / "WORKER_RESULT.json",
        {
            "schema_version": "worldsim_v6.r13_perception_worker.v1",
            "row_count": len(rows),
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_memory_mib": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

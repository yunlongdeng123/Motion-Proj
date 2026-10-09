"""只读检查 P0 输入和依赖，输出缺口；不启动 GPU 或下载模型。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--svd-root", type=Path, required=True)
    parser.add_argument("--raft", type=Path, required=True)
    parser.add_argument("--flow-completion", type=Path, required=True)
    args = parser.parse_args()
    checks = {"rgb_directory": args.rgb_root.is_dir(), "svd_directory": args.svd_root.is_dir(),
              "raft_file": args.raft.is_file(), "flow_completion_file": args.flow_completion.is_file()}
    required_svd = ["model_index.json", "unet/config.json", "vae/config.json",
                    "scheduler/scheduler_config.json", "image_encoder/config.json",
                    "feature_extractor/preprocessor_config.json",
                    "unet/diffusion_pytorch_model.fp16.safetensors",
                    "vae/diffusion_pytorch_model.fp16.safetensors",
                    "image_encoder/model.fp16.safetensors"]
    for relative in required_svd:
        checks["svd/" + relative] = (args.svd_root / relative).is_file()
    dependencies = {name: importlib.util.find_spec(name) is not None for name in
                    ["torch", "torchvision", "diffusers", "transformers", "accelerate", "einops", "PIL", "numpy"]}
    gpu = {"available": False, "count": 0}
    if dependencies["torch"]:
        import torch
        gpu = {"available": torch.cuda.is_available(), "count": torch.cuda.device_count(), "torch": torch.__version__}
    missing = [name for name, valid in checks.items() if not valid] + ["dependency/" + name for name, valid in dependencies.items() if not valid]
    print(json.dumps({"inputs": checks, "dependencies": dependencies, "gpu": gpu,
                      "missing": missing, "gpu_training_ready": not missing and gpu["available"],
                      "model_execution_verified": False,
                      "note": "仅检查输入可用性；不证明真实梯度、参数更新或推理已通过。"}, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

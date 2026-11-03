"""Validate the exact local frozen image encoder and strict real RGB hint on CPU."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
root = Path(__file__).parent
sys.path.insert(0, str(root / "python_overlay"))
sys.path.insert(0, "/root/autodl-tmp/third_party/hunyuan3d-omni-v61-me2")
import numpy as np
import torch
from PIL import Image
from hy3dshape.preprocessors import ImageProcessorV2
from hy3dshape.models.conditioners.omni_encoder import DinoImageEncoder

torch.set_num_threads(8)
started = time.monotonic()
contract_path = root.parent.parent / "inputs/a6_prior_contract.json"
contract = json.loads(contract_path.read_text())
dino_root = Path("/root/autodl-tmp/models/c2_dinov2_large_47b73")
weight = dino_root / "model.safetensors"
h = hashlib.sha256()
with weight.open("rb") as stream:
    for block in iter(lambda: stream.read(16*1024*1024), b""):
        h.update(block)
if h.hexdigest() != "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23":
    raise RuntimeError("DINO weight SHA256 mismatch")
preprocessor = ImageProcessorV2()
pixels, mask = preprocessor(contract["conditioning_image"], return_mask=True)
encoder = DinoImageEncoder(version=str(dino_root), image_size=518, use_cls_token=True)
encoder.requires_grad_(False)
encoder.eval()
with torch.no_grad():
    output = encoder(pixels)["dino"]["last_hidden_state"]
if tuple(output.shape) != (1, 1370, 1024) or not torch.isfinite(output).all():
    raise RuntimeError(f"DINO output mismatch {tuple(output.shape)}")
out = root / "dino_cpu_probe"
out.mkdir(exist_ok=True)
image = ((pixels[0].permute(1,2,0).numpy()+1)/2*255).round().clip(0,255).astype(np.uint8)
Image.fromarray(image).save(out / "model_preprocessed_hint.png")
np.savez_compressed(out / "dino_features.npz", features=output.numpy())
report = dict(weight_sha256=h.hexdigest(), dino_root=str(dino_root),
              original_visible_pixels=contract["true_visible_pixel_count"],
              preprocessed_shape=list(pixels.shape), dino_feature_shape=list(output.shape),
              feature_finite=True, frozen=True, gpu_initialized=torch.cuda.is_initialized(),
              cpu_threads=8, wall_seconds=time.monotonic()-started,
              claim="Official preprocessing turns transparent unknown pixels into a white prior hint background; no such pixels become observed RGB constraints")
(out / "report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report),flush=True)

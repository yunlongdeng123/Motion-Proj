"""Run the already installed original SegFormer B5 Cityscapes preprocessing."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from mmseg.apis import init_segmentor, inference_segmentor

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=Path, required=True)
parser.add_argument('--camera', type=int, default=0)
args = parser.parse_args()
torch.set_num_threads(6)
config = '/root/autodl-tmp/external/dggt_segformer/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py'
checkpoint = '/root/autodl-tmp/dggt/pretrained/segformer.b5.1024x1024.city.160k.pth'
model = init_segmentor(config, checkpoint, device='cuda:0')
image_dir = args.scene / 'image' / f'image_{args.camera}'
sky_dir = args.scene / 'sky' / f'sky_{args.camera}'
sky_dir.mkdir(parents=True, exist_ok=True)
records = []
start = time.monotonic()
for path in sorted(image_dir.glob('*.png')):
    out = sky_dir / path.name
    if out.exists():
        mask = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
    else:
        with torch.inference_mode():
            semantic = inference_segmentor(model, str(path))[0]
        mask = (semantic == 10).astype(np.uint8) * 255
        temporary = out.with_name(out.stem + '.partial.png')
        assert cv2.imwrite(str(temporary), mask)
        temporary.replace(out)
    records.append({'frame': path.stem, 'shape': list(mask.shape), 'sky_fraction': float((mask > 0).mean())})
    print(f'{len(records)} {path.name} sky={records[-1]["sky_fraction"]:.4f}', flush=True)
assert records, image_dir
manifest = {'model': 'Original NVlabs SegFormer B5 Cityscapes', 'config': config,
            'checkpoint': checkpoint, 'sky_class': 10, 'frames': records,
            'seconds': time.monotonic() - start}
(args.scene / f'sky_manifest_camera_{args.camera}.json').write_text(json.dumps(manifest, indent=2) + '\n')

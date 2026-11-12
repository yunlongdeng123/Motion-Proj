"""依据已保存的原图路径与编码前allowed mask重建输入，不复制原始RGB资产。"""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image


def materialize(root):
    root=Path(root);manifest=json.loads((root/'dataset.json').read_text())
    sources=json.loads((root/manifest.get('sources','sources.json')).read_text())['sources'];built=0
    for source in sources:
        raw=Path(source['image_path'])
        if not raw.is_file():raise FileNotFoundError(f'Original source RGB must remain available: {raw}')
        allowed=np.asarray(Image.open(root/source['feature_allowed_mask']))>0
        rgb=np.asarray(Image.open(raw).convert('RGB')).copy()
        if rgb.shape[:2]!=allowed.shape:raise ValueError('Native source/mask shape mismatch')
        rgb[~allowed]=128;target=root/source['feature_input'];target.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(rgb).save(target);built+=1
    print(f'MATERIALIZED {built} source inputs; original RGB retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,type=Path);a=p.parse_args();materialize(a.dataset)

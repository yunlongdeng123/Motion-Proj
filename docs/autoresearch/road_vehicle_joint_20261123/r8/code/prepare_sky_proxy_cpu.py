"""Map existing real-photo SegFormer labels into the exact DGGT input grid.

Reuses stored frozen-model outputs; no new semantic inference or RGB rewrite.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--input-bundle',type=Path,required=True);a=p.parse_args()
folder=a.input_bundle.parent;contract=json.loads((folder/'contract.json').read_text());assert sha(a.input_bundle)==contract['input_bundle_sha256']
with np.load(a.input_bundle,allow_pickle=False) as z:frames=z['raw_frames'].copy();H,W=z['images_srgb'].shape[-2:]
original=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/semantics');rows=[];labels=[];valid=[]
for raw,row in zip(frames,contract['rows']):
    assert int(raw)==row['raw'];cache=Path(row['cache_path']);assert sha(cache)==row['cache_sha256']
    semantic=original/f'{int(raw):03d}.png';source=np.asarray(Image.open(semantic));assert source.ndim==2 and source.max()<=18
    with np.load(cache,allow_pickle=False) as z:uv=z['photo_source_brown_uv'];pv=z['photo_pinhole_valid']
    pinhole=map_coordinates(source.astype(np.float32),[uv[...,1],uv[...,0]],order=0,mode='nearest').astype(np.uint8)
    resized=np.asarray(Image.fromarray(pinhole).resize((W,H),Image.Resampling.NEAREST));v=np.asarray(Image.fromarray(pv.astype(np.uint8)).resize((W,H),Image.Resampling.NEAREST))>0
    labels.append(resized);valid.append(v);Image.fromarray((resized==10).astype(np.uint8)*255).save(folder/f'{int(raw):03d}_sky_proxy.png')
    rows.append(dict(raw=int(raw),original_Brown_semantic=str(semantic),semantic_sha256=sha(semantic),cache_sha256=sha(cache),sky_pixels=int(((resized==10)&v).sum()),valid_pixels=int(v.sum())))
labels=np.asarray(labels);valid=np.asarray(valid);sky=(labels==10)&valid;bg=(labels!=10)&valid
target=folder/'sky_proxy.npz'
with target.open('wb') as f:np.savez_compressed(f,raw_frames=frames,semantic_labels=labels,sky_mask=sky,background_geometry_mask=bg,pixel_valid=valid)
value=dict(schema='R8.real_photo_sky_proxy_CPU_mapping.v1',complete=True,input_bundle_sha256=sha(a.input_bundle),sky_proxy_sha256=sha(target),frames=frames.tolist(),input_hw=[H,W],
    model='existing frozen SegFormer B5 Cityscapes prediction cache',source_receipt=str(original/'receipt.json'),source_receipt_sha256=sha(original/'receipt.json'),
    source_labelmap='Cityscapes train IDs 0..18; sky=10, road=0, car=13; distinct from DGGT native 10-class head',
    mapping='nearest stored Brown label via exact native Brown->pinhole UV cache, then nearest to the fixed 350x518 image grid',
    semantic_ground_truth=False,predicted_DGGT_semantic_head_used=False,RGB_modified=False,rows=rows,
    new_semantic_forward_calls=0,new_diffusion_calls=0,new_Gaussian_calls=0,CPU_only=True)
(folder/'sky_proxy_receipt.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))

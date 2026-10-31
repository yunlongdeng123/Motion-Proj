"""合法二值mask解码：RGBA透明度不是前景语义通道。"""
from pathlib import Path
import numpy as np
from PIL import Image


def read_body_mask(path):
    image=Image.open(Path(path))
    raw=np.asarray(image)
    color=np.asarray(image.convert('RGB'))
    mask=(color>0).any(axis=2)
    naive=(raw>0).any(axis=2) if raw.ndim==3 else raw>0
    return mask,dict(path=str(path),mode=image.mode,width=image.width,height=image.height,
        foreground_pixels=int(mask.sum()),naive_any_channel_foreground_pixels=int(naive.sum()),
        alpha_ignored=True,RGB_RGBA_semantics='foreground from RGB channels only; no alpha-channel foreground')


def validate_RGBA_equivalence():
    import tempfile
    rgb=np.zeros((3,4,3),np.uint8);rgb[1,1:3]=255
    rgba=np.concatenate((rgb,np.full((3,4,1),255,np.uint8)),axis=2)
    gray=rgb[:,:,0];la=np.stack((gray,np.full_like(gray,255)),axis=2)
    masks=[]
    with tempfile.TemporaryDirectory(prefix='r4-mask-decode-') as folder:
        for name,array in [('RGB',rgb),('RGBA',rgba),('L',gray),('LA',la)]:
            path=Path(folder)/(name+'.png');Image.fromarray(array).save(path)
            mask,_=read_body_mask(path);masks.append(mask)
    if any(not np.array_equal(masks[0],m) for m in masks) or int(masks[0].sum())!=2:
        raise AssertionError('RGB/RGBA/L/LA alpha changed mask semantics')
    return dict(RGB_RGBA_gray_LA_foreground_equal=True,foreground_pixels=2,opaque_alpha_pixels=12,
        actual_PNG_roundtrip_and_shared_decoder=True)

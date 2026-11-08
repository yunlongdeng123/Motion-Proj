from pathlib import Path
import numpy as np
from PIL import Image
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val029_v44_20261108_r1/inputs/native')
folder=root/'dynamic_masks/dynamic_mask_0';folder.mkdir(parents=True,exist_ok=True)
for p in (root/'mask/mask_0/Annotations').glob('*.png'):
    labels=np.array(Image.open(p));Image.fromarray(np.uint8(labels>0)*255).save(folder/p.name)
assert len(list(folder.glob('*.png')))==158
print('158 explicit binary masks ready')

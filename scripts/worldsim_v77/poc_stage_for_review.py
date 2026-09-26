"""Copy fixed review frames for the two-scene local, lossless compositor."""
import json
import pathlib
import shutil
from PIL import Image

RUN=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
STAGE=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/review_input')
STAGE.mkdir(parents=True,exist_ok=True)
reg=json.loads((RUN/'registration.json').read_text())
for spec in reg['scenes']:
    name=spec['name'];source=pathlib.Path(spec['root'])
    for row in spec['frames']:
        f=row['frame'];dest=STAGE/name/f'{f:03d}';dest.mkdir(parents=True,exist_ok=True)
        for cam in range(6):
            Image.open(source/'images'/f'{f:03d}_{cam}.jpg').convert('RGB').resize((688,384),Image.Resampling.BICUBIC).save(dest/f'raw_cam{cam}.jpg',quality=93)
            shutil.copy2(RUN/name/'bg_input'/f'{f:03d}'/f'cam{cam}.png',dest/f'pixel_bg_cam{cam}.png')
            shutil.copy2(RUN/name/'background'/f'{f:03d}'/f'bg_cam{cam}.png',dest/f'omega_bg_cam{cam}.png')
print(STAGE)

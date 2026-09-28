from nine_common import *
from PIL import Image
reg=read(ROOT/'registration.json')
for s in reg['scenes']:
    base=ROOT/s['name'];asset=base/'asset';src=read(asset/'registration.json')['source'];c=src['camera'];f=src['frame']
    Image.open(base/f'cam{c}/rgb'/f'{f:05}.png').resize((688,384),Image.Resampling.BICUBIC).save(asset/'source_full.png')
    Image.open(base/f'cam{c}/core'/f'{f:05}.png').resize((688,384),Image.Resampling.NEAREST).save(asset/'source_core.png')
print('SOURCE_FULL_READY')

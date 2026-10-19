"""Arrange complete 480x320 native images; never crop or replace model outputs."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
W=Path(__file__).resolve().parent
O=W.parent.parent/'outputs/gn11_stage1_v2_20261019'
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
arms=['full','real_only','prior_only','ownership_only']
dest=W/'native_review_sheets';dest.mkdir(exist_ok=True)
manifest=[]
for batch,start in enumerate(range(89,138,7)):
    frames=range(start,min(start+7,138));sheet=Image.new('RGB',(1920,len(frames)*350+40),'#122232');draw=ImageDraw.Draw(sheet)
    for col,arm in enumerate(arms):draw.text((col*480+12,8),arm,fill='white',font=font)
    for row,frame in enumerate(frames):
        for col,arm in enumerate(arms):
            root=W/'native30k_pair' if arm in ['full','real_only'] else W/'cpu_exports/native'
            image=root/arm/'static'/f'0008{frame:03d}.png'
            with Image.open(image) as im:
                assert im.size==(480,320);sheet.paste(im.convert('RGB'),(col*480,40+row*350))
            draw.text((col*480+8,40+row*350+322),f'raw{frame}',fill='white',font=font)
    path=dest/f'static_{start:03d}_{min(start+6,137):03d}.jpg';sheet.save(path,quality=95)
    manifest.append(str(path))
print(json.dumps(manifest))

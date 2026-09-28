from nine_common import *
from PIL import Image,ImageDraw
import tarfile
reg=read(ROOT/'registration.json');sheet=Image.new('RGB',(1200,900),(23,33,46));d=ImageDraw.Draw(sheet)
for i,s in enumerate(reg['scenes']):
    base=ROOT/s['name'];r=read(base/'asset/registration.json');im=Image.open(base/'asset/source_rgba.png').convert('RGBA');im.thumbnail((390,250));x=(i%3)*400;y=(i//3)*300
    d.rectangle((x,y+28,x+397,y+297),fill=(130,130,130));sheet.paste(im,(x+(400-im.width)//2,y+35),im)
    src=r['source'];d.text((x+5,y+5),f"{s['name']} actor{s['actor']} f{src['frame']} CAM{src['camera']} core={src['core_pixels']}",fill='white')
sheet.save(ROOT/'input_assets.jpg',quality=95)
with tarfile.open(ROOT/'input_review.tar','w') as tar:
    tar.add(ROOT/'input_assets.jpg',arcname='input_assets.jpg')
    for s in reg['scenes']:
        tar.add(ROOT/s['name']/'asset/registration.json',arcname=s['name']+'/asset_registration.json')
        for v in s['streams']:
            if v['active']:tar.add(ROOT/s['name']/f"cam{v['camera']}/mask_review.jpg",arcname=s['name']+f"/mask_cam{v['camera']}.jpg")
print('INPUT_REVIEW_DONE')

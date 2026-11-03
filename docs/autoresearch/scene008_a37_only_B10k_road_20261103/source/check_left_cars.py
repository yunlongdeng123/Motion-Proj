from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_erosion
R=Path(__file__).resolve().parent.parent;rows=[]
for raw in range(89,138):
    d=np.load(R/'inputs/views'/f'{raw:03d}.npz');labels=d['actor_labels'];mask=(labels>=1)&(labels<=3)
    a=np.asarray(Image.open(R/'review/frames'/f'{raw:03d}/baseline.png'),int);b=np.asarray(Image.open(R/'review/frames'/f'{raw:03d}/repaired.png'),int);err=np.abs(a-b).max(2)
    y,x=np.unravel_index(np.argmax(np.where(mask,err,-1)),mask.shape)
    rows.append(dict(raw=raw,MAE_8bit=float(np.abs(a-b)[mask].mean()) if mask.any() else 0,max_8bit=int(err[y,x]),x=int(x),y=int(y),pixels_over16=int(((err>16)&mask).sum())))
worst=sorted(rows,key=lambda r:r['max_8bit'],reverse=True)[0];raw=worst['raw'];x=worst['x'];y=worst['y'];box=(max(0,x-35),max(0,y-25),min(480,x+36),min(320,y+26));images=[]
for name in ['baseline','repaired']:
    image=Image.open(R/'review/frames'/f'{raw:03d}'/(name+'.png')).crop(box);images.append(image.resize((image.width*4,image.height*4),Image.Resampling.NEAREST))
strip=Image.new('RGB',(images[0].width*2,images[0].height));strip.paste(images[0],(0,0));strip.paste(images[1],(images[0].width,0));strip.save(R/'review/left_worst_crop.png')
(R/'review/left_vehicle_check.json').write_text(json.dumps(dict(rows=rows,worst=worst,comparison='B10k baseline on left, final repair on right; nearest-neighbour4x crop'),indent=2)+'\n')
print(json.dumps(dict(worst=worst,pixels_over16=sum(r['pixels_over16'] for r in rows))))

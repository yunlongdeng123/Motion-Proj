from pathlib import Path
import numpy as np,json
from PIL import Image,ImageDraw
from scipy.ndimage import binary_fill_holes,binary_dilation
ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927/r28');rows=[];sheet=Image.new('RGB',(1200,4*260),(12,20,30))
for i,f in enumerate([0,3,8,9]):
 m=np.array(Image.open(ROOT/'write_mask'/f'{f:05}.png'))>0;im=np.array(Image.open(ROOT/'input'/f'{f:05}.png'));holes=binary_fill_holes(m)&~m;edge=binary_dilation(m,iterations=3)&~m;v=im.copy();v[m]=(.3*v[m]+.7*np.array([0,180,150])).astype('uint8');v[holes]=[255,0,255];v[edge]=(.5*v[edge]+.5*np.array([255,200,0])).astype('uint8');y,x=np.where(m);box=(x.min()-15,y.min()-10,x.max()+16,y.max()+16)
 for j,(name,a) in enumerate([('original',Image.fromarray(im)),('mask: cyan; outer3px:yellow; holes:magenta',Image.fromarray(v)),('r28 native',Image.open(ROOT/'native'/f'{f:05}.png'))]):
  tile=Image.new('RGB',(400,260),(12,20,30));tile.paste(a.crop(box).resize((400,235)),(0,25));ImageDraw.Draw(tile).text((5,5),f'f{f} {name}',fill='white');sheet.paste(tile,(j*400,i*260))
 rows.append(dict(frame=f,internal_holes=int(holes.sum()),outer3px=int(edge.sum())))
sheet.save(ROOT/'mask_boundary_audit.jpg',quality=97);(ROOT/'mask_boundary_audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)

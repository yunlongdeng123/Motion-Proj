from PIL import ImageDraw
from hybrid_common import *
for s in read(ROOT/'registration.json')['scenes']:
 out=ROOT/s['name']
 if not (out/'final/00029.png').exists():continue
 sheet=Image.new('RGB',(480*4,280*4))
 zoom=Image.new('RGB',(420*4,245*4))
 for j,i in enumerate([0,7,15,29]):
  f=s['source_frames'][i];core=mask(out/'delete'/f'{i:05}.png');y,x=np.where(core);cx=(x.min()+x.max())/2;cy=(y.min()+y.max())/2;cw=max(240,(x.max()-x.min())+90);ch=cw*9/16;left=max(0,min(W-cw,cx-cw/2));top=max(0,min(H-ch,cy-ch/2));rect=(int(left),int(top),int(left+cw),int(top+ch))
  paths=[out/'rgb'/f'{i:05}.png',FULL/s['name']/f'cam{s["camera"]}/background/{f:05}.png',out/'prior_png'/f'{i:05}.png',out/'final'/f'{i:05}.png']
  for k,(path,label) in enumerate(zip(paths,['Original','Old DriveEditor','Internal prior','DiffuEraser final'])):
   im=Image.fromarray(rgb(path));small=im.resize((480,268));draw=ImageDraw.Draw(small);draw.text((8,6),f'{label} | {s["name"]} f{f}',fill='yellow');sheet.paste(small,(k*480,j*280));crop=im.crop(rect).resize((420,236));ImageDraw.Draw(crop).text((6,6),f'{label} f{f}',fill='yellow');zoom.paste(crop,(k*420,j*245))
 sheet.save(out/'quicklook.jpg',quality=95);zoom.save(out/'quicklook_zoom.jpg',quality=95)

from pathlib import Path
import sys,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');R=T/'r5';R.mkdir(exist_ok=False)
S=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-REPAIR-20260927/r1/official_000');rows=[];tiles=[]
for f in [0,3,6,9]:
    rgb=np.array(Image.open(S/'rgb'/f'{f:05}.png').convert('RGB'));m=np.array(Image.open(S/'sam'/f'core_{f:05}.png'))>0;y,x=np.where(m);box=[max(0,int(x.min())-12),max(0,int(y.min())-12),min(1024,int(x.max())+13),min(576,int(y.max())+13)];rgba=np.dstack([rgb,m.astype('uint8')*255]);crop=Image.fromarray(rgba).crop(box);crop.save(R/f'source_{f:03}.png');context=Image.fromarray(rgb).crop(box);context.save(R/f'context_{f:03}.png');tile=Image.new('RGB',(512,344),(35,35,35));context.thumbnail((512,320));tile.paste(context,(0,24));ImageDraw.Draw(tile).text((5,5),f'official000 / target12 f{f} source crop',fill='white');tiles.append(tile);rows.append(dict(frame=f,box=box,mask_pixels=int(m.sum()),source_rgb=str(S/'rgb'/f'{f:05}.png'),source_mask=str(S/'sam'/f'core_{f:05}.png'),mask_bbox=[int(x.min()),int(y.min()),int(x.max()),int(y.max())]))
sheet=Image.new('RGB',(1024,688))
for i,im in enumerate(tiles):sheet.paste(im,((i%2)*512,(i//2)*344))
sheet.save(R/'source_selection.jpg',quality=95);dump(R/'source_selection.json',rows)
state=read(T/'r1/drive_state.json');print('DRIVE',state['state'],[(v['scene'],v['start']) for v in state['completed']])

"""Actual two-frame I direction boards; no result/mask fabrication or scoring."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

R=Path(__file__).resolve().parent.parent
WORKSPACE=R.parent.parent
C=WORKSPACE/'outputs/scene008_v44_a5_c49_rgb_20261107'
WORLD=Path('C:/Users/dengyunlong/Documents/Codex/2026-11-05/j-ao-j/outputs/scene008_road_fusion_bake/delivery/repro/red_review/world_masks49_v1')
CASES=[('Delivered21K',R/'evidence/baseline4'),('H2 + C material2000',C/'evidence/H2_material2000_retry1'),('I world handover .4m',R/'evidence/I_world_v1')]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    with np.load(WORLD/'137.npz',allow_pickle=False) as z:
        m=z['lateral_y_minus']|z['lateral_y_plus'];y,x=np.where(m)
    lateral=(max(0,int(x.min())-6),max(0,int(y.min())-6),min(480,int(x.max())+7),min(320,int(y.max())+7))
    rows=[(128,None,'full native480x320'),(128,(165,188,208,234),'original primary128 bbox; display only'),
          (137,None,'full native480x320'),(137,lateral,'fixed world lateral-band bbox+6; display only')]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16);sources=[];out=[]
    for state in ['keep','delete']:
        board=Image.new('RGB',(1510,1260),'#f4f7f8');draw=ImageDraw.Draw(board)
        for col,(label,case) in enumerate(CASES):draw.text((30+490*col,12),label,font=font,fill='#173b4a')
        top=45
        for raw,box,description in rows:
            for col,(label,case) in enumerate(CASES):
                p=case/f'frames/{raw:03d}/{state}.png'
                with Image.open(p) as im:assert im.size==(480,320);native=im.convert('RGB').copy()
                v=native if box is None else native.crop(box)
                if box is not None:
                    scale=min(5,480/v.width,200/v.height);v=v.resize((round(v.width*scale),round(v.height*scale)),Image.Resampling.NEAREST)
                draw.text((30+490*col,top),f'raw{raw} {state.upper()} / '+description,font=font,fill='#173b4a')
                board.paste(v,(30+490*col+(480-v.width)//2,top+26))
                sources.append(dict(raw=raw,state=state,label=label,path=str(p),sha256=sha(p),native_size=[480,320],display_crop=box))
            top+=350 if box is None else 250
        p=R/f'figures/I_v1_partial_{state}_direction.png';board.save(p);out.append(dict(path=str(p),sha256=sha(p),size=board.size))
    report={'status':'ACTUAL_TWO_RAW_FOUR_STATE_DIRECTION_BOARDS_NOT_FULL_PILOT','raws':[128,137],'states':['keep','delete'],
       'early94_105_rendered':False,'all49_rendered':False,'frames_original_unchanged':True,
       'zoom':'NEAREST only for display; no black masked/interpolated algorithm image; metrics use original fixed domains',
       'world137_lateral_display_bbox':lateral,'sources':sources,'outputs':out,'quality_accepted':False,'human_verdict':None}
    (R/'evidence/I_v1_direction_board_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'outputs':out,'partial_only':True}))


if __name__=='__main__':main()

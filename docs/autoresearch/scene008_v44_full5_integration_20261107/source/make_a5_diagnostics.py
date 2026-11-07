"""CPU display boards for an actual saved integration case; never render."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    root=Path(__file__).absolute().parent.parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',default='full49_actor5')
    parser.add_argument('--raw',type=int,default=94)
    args=parser.parse_args()
    assert '/' not in args.case and '\\' not in args.case
    rows=json.loads((root/'evidence/a5_contact_crop_contract.json').read_text(encoding='utf-8'))['frames']
    row=next(r for r in rows if r['raw']==args.raw)
    box=tuple(row['crop_xyxy'])
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    folder=root/'figures';folder.mkdir(exist_ok=True)
    for state in ('keep','delete'):
        paths=[root/'evidence/v44_all49_reference/frames'/f'{args.raw:03d}'/'observed.png',
               root/'evidence/v44_all49_reference/frames'/f'{args.raw:03d}'/(state+'.png'),
               root/'evidence'/args.case/'frames'/f'{args.raw:03d}'/(state+'.png')]
        if not all(p.is_file() for p in paths):
            print('PENDING',state);continue
        full=Image.new('RGB',(1480,370),'#f3f7f9');zoom=Image.new('RGB',(1000,310),'#f3f7f9')
        for col,(label,path) in enumerate(zip(('Observed (a37 present)','v44 '+state.upper(),'Integrated '+state.upper()),paths)):
            with Image.open(path) as im:
                assert im.size==(480,320)
                im=im.convert('RGB')
                full.paste(im,(5+490*col,45))
                crop=im.crop(box)
                scale=min(320/crop.width,240/crop.height)
                display=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
                zoom.paste(display,(5+330*col+(320-display.width)//2,48+(240-display.height)//2))
            ImageDraw.Draw(full).text((8+490*col,12),label,font=font,fill='#193b4a')
            ImageDraw.Draw(zoom).text((8+330*col,12),label,font=font,fill='#193b4a')
        stem=args.case+'_raw'+f'{args.raw:03d}'+'_'+state
        full.save(folder/(stem+'_full.png'));zoom.save(folder/(stem+'_GTmask_crop.png'))
        print(stem,'native full1480x370; same GTmask bbox+6',box,'display zoom only1000x310')


if __name__=='__main__':
    main()

"""Assemble actual native RGB; crop/zoom is display-only and never a score mask."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',action='append',required=True)
    p.add_argument('--world-mask-root',type=Path,required=True)
    p.add_argument('--wheel-registry',type=Path)
    p.add_argument('--wheel-variant',action='append',help='Separate diagnostic130 case override, preserving pilot receipt')
    p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--case-id',default='initial_pilot4')
    args=p.parse_args();variants={k:Path(v) for k,v in (s.split('=',1) for s in args.variant)}
    assert len(variants)==3 and 'baseline' in variants and set(variants)<={'baseline','AA','C','D','E','F','H'}
    columns=['baseline']+[name for name in ['AA','C','D','E','F','H'] if name in variants]
    wheel_variants={k:Path(v) for k,v in (s.split('=',1) for s in args.wheel_variant)} if args.wheel_variant else variants
    folder=args.output_root/'figures';folder.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    sources=[];boxes={}
    for raw in [94,105,128,137]:
        if raw in [94,105]:box=(0,0,480,320)
        elif raw==128:box=(165,188,208,234)
        else:
            with np.load(args.world_mask_root/'137.npz',allow_pickle=False) as data:y,x=np.nonzero(data['world_patch_clean_road'])
            box=(max(0,int(x.min())-6),max(0,int(y.min())-6),min(480,int(x.max())+7),min(320,int(y.max())+7))
        boxes[str(raw)]=box
    outputs=[]
    for state in ['delete','keep']:
        board=Image.new('RGB',(1510,1240),'#f3f7f9');draw=ImageDraw.Draw(board)
        for col,name in enumerate(columns):draw.text((35+490*col,12),{'baseline':'Latest v44+a5 (frozen)','AA':'+ AA road support','C':'+ C material / same support','D':'+ D fixed support (posthoc)','E':'E original-surface appearance','F':'F bounded-surface appearance','H':'H2 model-foreground culling'}[name],font=font,fill='#193b4a')
        top=48
        for raw in [94,105,128,137]:
            box=boxes[str(raw)];full=raw in [94,105];height=350 if full else 250
            for col,name in enumerate(columns):
                source=variants[name]/f'frames/{raw:03d}/{state}.png'
                with Image.open(source) as im:assert im.size==(480,320);native=im.convert('RGB').copy()
                sources.append(dict(raw=raw,state=state,variant=name,path=str(source),sha256=sha(source),native_size=[480,320],display_crop=box))
                view=native.crop(box)
                if not full:
                    scale=min(4,480/view.width,195/view.height)
                    view=view.resize((max(1,round(view.width*scale)),max(1,round(view.height*scale))),Image.Resampling.NEAREST)
                draw.text((35+490*col,top),f'raw{raw} {state.upper()} / '+('full native frame' if full else 'fixed road crop; display zoom'),font=font,fill='#193b4a')
                board.paste(view,(30+490*col+(480-view.width)//2,top+27))
            top+=height
        path=folder/(args.case_id+'_'+state+'_four_frames.png');board.save(path);outputs.append(str(path))
        close=Image.new('RGB',(1200,300),'#f3f7f9');close_draw=ImageDraw.Draw(close)
        for col,name in enumerate(columns):
            with Image.open(variants[name]/f'frames/128/{state}.png') as im:view=im.convert('RGB').crop((165,188,208,234))
            close_draw.text((10+400*col,10),'raw128 '+name+' '+state.upper(),font=font,fill='#193b4a')
            close_draw.text((10+400*col,36),'Original blue bbox / x5 display',font=font,fill='#193b4a')
            view=view.resize((215,230),Image.Resampling.NEAREST);close.paste(view,(10+400*col,65))
        path=folder/(args.case_id+'_raw128_'+state+'_original_blue_box.png');close.save(path);outputs.append(str(path))
    wheel_status='NOT_REQUESTED' if not args.wheel_registry else 'NOT_YET_AVAILABLE'
    if args.wheel_registry and args.wheel_registry.is_file() and set(wheel_variants)==set(columns) and all((directory/'frames/130/delete.png').is_file() for directory in wheel_variants.values()):
        registry=json.loads(args.wheel_registry.read_text(encoding='utf-8-sig'))
        rays=registry['rays']['130'];pixels=np.asarray([r['pixel_xy'] for r in rays]);x,y=pixels.T
        box=(max(0,int(x.min())-8),max(0,int(y.min())-8),min(480,int(x.max())+9),min(320,int(y.max())+9))
        board=Image.new('RGB',(1510,650),'#f3f7f9');draw=ImageDraw.Draw(board)
        for col,name in enumerate(columns):
            source=wheel_variants[name]/'frames/130/delete.png'
            with Image.open(source) as im:assert im.size==(480,320);native=im.convert('RGB').copy()
            draw.text((35+490*col,12),'raw130 '+name+' DELETE / separate diagnostic',font=font,fill='#193b4a');board.paste(native,(30+490*col,40))
            view=native.crop(box);scale=min(4,480/view.width,220/view.height);view=view.resize((round(view.width*scale),round(view.height*scale)),Image.Resampling.NEAREST)
            draw.text((35+490*col,375),'Original35-ray bbox; display zoom only',font=font,fill='#193b4a');board.paste(view,(30+490*col+(480-view.width)//2,405))
            sources.append(dict(raw=130,state='delete',variant=name,path=str(source),sha256=sha(source),native_size=[480,320],display_crop=box,diagnostic_only=True))
        path=folder/(args.case_id+'_wheel130_separate.png');board.save(path);outputs.append(str(path));wheel_status='BUILT_FROM_ACTUAL_NATIVE_PNG'
    receipt=dict(case_id=args.case_id,status='ACTUAL_NATIVE_PNG_BOARDS_NOT_QUALITY_ACCEPTANCE',outputs=outputs,
        pilot_raws=[94,105,128,137],variants=columns,separate_wheel130=wheel_status,crop_boxes=boxes,sources=sources,
        original_pixels_preserved_in_sources=True,display_zoom='NEAREST only; never changes metric denominator',
        no_black_masked_pixels_or_inpainting=True)
    (args.output_root/'evidence'/f'{args.case_id}_board_manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(outputs=outputs,wheel=wheel_status)))


if __name__=='__main__':main()

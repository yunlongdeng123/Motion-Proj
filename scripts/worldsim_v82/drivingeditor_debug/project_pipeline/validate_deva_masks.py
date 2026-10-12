#!/usr/bin/env python3
"""Validate real DEVA instance PNGs/pred.json and produce an actor-selection inventory."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scene',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
images=sorted((args.scene/'image/image_0').glob('*.png'))
masks=sorted((args.output/'Annotations').glob('*.png'))
assert [p.stem for p in masks]==[p.stem for p in images], 'Mask filenames/count differ from image inputs'
annotations=json.loads((args.output/'pred.json').read_text())['annotations']
by_name={Path(row['file_name']).stem:row for row in annotations}
assert set(by_name)=={p.stem for p in images} and len(annotations)==len(images)
frames=[];tracks=collections.defaultdict(list);samples=[]
sample_indices=set(np.linspace(0,len(images)-1,min(5,len(images)),dtype=int).tolist())
rng=np.random.RandomState(21)
palette=rng.randint(32,256,(256,3),dtype=np.uint8);palette[0]=0
id_font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
id_font=ImageFont.truetype(str(id_font_path),25) if id_font_path.exists() else ImageFont.load_default()
for index,(image_path,mask_path) in enumerate(zip(images,masks)):
    rgb=np.array(Image.open(image_path).convert('RGB'))
    mask=np.array(Image.open(mask_path))
    assert mask.ndim==2 and mask.shape==rgb.shape[:2], f'Wrong mask dimensions: {mask_path.name}'
    assert mask.dtype==np.uint8, 'Author short-ID output must be uint8'
    ids,areas=np.unique(mask,return_counts=True)
    counts={int(k):int(v) for k,v in zip(ids,areas) if k!=0}
    announced={int(item['id']):item for item in by_name[image_path.stem]['segments_info']}
    assert set(counts)==set(announced), f'PNG IDs differ from pred.json for {image_path.name}'
    assert all(counts[k]==int(announced[k]['area']) for k in counts), 'pred.json pixel area mismatch'
    assert len(counts)<=200, 'More than the author default200 simultaneous objects'
    # ObjectManager may randomly reassign collided IDs in1..255; the200-object
    # cap limits concurrent instances, not the numerical ID range.
    for instance,area in counts.items():
        yy,xx=np.where(mask==instance)
        tracks[instance].append({'frame':image_path.stem,'area':area,
            'bbox_xyxy':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]})
    frames.append({'frame':image_path.stem,'instance_count':len(counts),
        'nonbackground_fraction':float(np.mean(mask!=0)),'ids':sorted(counts)})
    if index in sample_indices:
        colored=palette[mask]
        overlay=rgb.copy();selected=mask!=0
        overlay[selected]=(rgb[selected]*.55+colored[selected]*.45).astype(np.uint8)
        labeled=Image.fromarray(overlay);label_draw=ImageDraw.Draw(labeled)
        for instance,area in counts.items():
            if area<500: continue
            box=tracks[instance][-1]['bbox_xyxy'];x=(box[0]+box[2])//2;y=(box[1]+box[3])//2
            label=str(instance);width=label_draw.textlength(label,font=id_font)
            label_draw.rectangle((x-3,y-3,x+width+4,y+29),fill=(0,0,0))
            label_draw.text((x,y),label,font=id_font,fill=tuple(int(v) for v in palette[instance]))
        labeled.save(args.output/f'instance_ids_{image_path.stem}.jpg',quality=92)
        samples.append((image_path.stem,Image.fromarray(rgb),Image.fromarray(overlay),counts))
assert tracks, 'All masks are empty; this is not a successful instance segmentation'
report={'status':'passed','frames':len(images),'distinct_instance_ids':len(tracks),
    'min_instances_per_frame':min(f['instance_count'] for f in frames),
    'max_instances_per_frame':max(f['instance_count'] for f in frames),
    'mask_dtype':'uint8','image_size':[rgb.shape[1],rgb.shape[0]],'frames_detail':frames,
    'semantic_limitation':'Instance IDs include static scene regions. Dynamic actor selection remains pending.'}
(args.output/'mask_validation.json').write_text(json.dumps(report,indent=2)+'\n')
inventory={'status':'actor_selection_pending','instances':[
    {'id':instance,'observed_frames':len(entries),'first_frame':entries[0]['frame'],
     'last_frame':entries[-1]['frame'],'area_min':min(x['area'] for x in entries),
     'area_max':max(x['area'] for x in entries),'observations':entries}
    for instance,entries in sorted(tracks.items())]}
(args.output/'track_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
fontfile=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
font=ImageFont.truetype(str(fontfile),16) if fontfile.exists() else ImageFont.load_default()
sheet=Image.new('RGB',(972,len(samples)*352),(235,238,242));draw=ImageDraw.Draw(sheet)
for index,(frame,rgb_im,overlay,counts) in enumerate(samples):
    y=index*352
    sheet.paste(rgb_im.resize((480,320),Image.Resampling.LANCZOS),(0,y))
    sheet.paste(overlay.resize((480,320),Image.Resampling.LANCZOS),(492,y))
    draw.text((5,y+325),f'{frame}: {len(counts)} instances | source / DEVA+SAM instance overlay',font=font,fill=(0,0,0))
sheet.save(args.output/'mask_contact_sheet.jpg',quality=91)
print(json.dumps({k:v for k,v in report.items() if k!='frames_detail'},indent=2))

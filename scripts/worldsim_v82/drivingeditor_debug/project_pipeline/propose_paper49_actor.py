#!/usr/bin/env python3
"""Identify candidate49-frame track by exact21-frame anchor-mask overlap.

Creates review evidence only, never authorizes an identity without visual review.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont

base=Path('/root/autodl-tmp/drivingeditor_repro')
scene=base/'data/waymo_paper_scenes/scene0017_000_048'
anchor=base/'data/waymo_scenes/0017085_demo_000_020'
maskroot=scene/'mask/mask_0'
assert json.loads((maskroot/'inference_manifest.json').read_text())['status']=='validated'
assert json.loads((maskroot/'mask_validation.json').read_text())['status']=='passed'
prep=json.loads((scene/'preparation_manifest.json').read_text())
overlap=[]
for i in range(21):
    name=f'{17000+i:07d}'
    a=np.array(Image.open(anchor/'dynamic_masks/dynamic_mask_0'/f'{name}.png'))>0
    b=np.array(Image.open(maskroot/'Annotations'/f'{name}.png'))
    ids,count=np.unique(b[a],return_counts=True)
    eligible=[(int(k),int(v)) for k,v in zip(ids,count) if k]
    best_id=max(eligible,key=lambda x:x[1])[0]
    iou=float(np.sum(a&(b==best_id))/np.sum(a|(b==best_id)))
    overlap.append({'frame':name,'candidate_id':best_id,'anchor_iou':iou})
ids=set(x['candidate_id'] for x in overlap)
assert len(ids)==1 and min(x['anchor_iou'] for x in overlap)>.75,'Ambiguous anchor match; manual investigation needed'
candidate_id=next(iter(ids))
frames=[]
sheet=Image.new('RGB',(7*288,7*226),'white');draw=ImageDraw.Draw(sheet)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
for i,row in enumerate(prep['frames']):
    name=row['id'];rgb=np.array(Image.open(scene/'image/image_0'/f'{name}.png').convert('RGB'))
    mask=np.array(Image.open(maskroot/'Annotations'/f'{name}.png'))==candidate_id
    if mask.any():
        yy,xx=np.where(mask);box=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
        x0=max(0,box[0]-40);y0=max(0,box[1]-40);x1=min(rgb.shape[1],box[2]+41);y1=min(rgb.shape[0],box[3]+41)
        overlay=rgb.copy();overlay[mask]=(overlay[mask]*.6+np.array([255,90,20])*.4).astype(np.uint8)
        im=Image.fromarray(overlay).crop((x0,y0,x1,y1));im.thumbnail((288,192))
    else:
        box=None;im=Image.fromarray(rgb);im.thumbnail((288,192))
    rowrec={'frame':name,'candidate_id':candidate_id,'pixels':int(mask.sum()),'bbox_xyxy':box,'left_image_edge':bool(box and box[0]==0),'status':'candidate_visible' if mask.any() else 'candidate_missing_requires_visibility_review'}
    frames.append(rowrec)
    x=(i%7)*288;y=(i//7)*226;sheet.paste(im,(x+(288-im.width)//2,y+(192-im.height)//2));draw.text((x+4,y+194),f'{name} ID{candidate_id} / {mask.sum()} px',font=font,fill='black')
result={'status':'visual_review_pending','scene':scene.name,'candidate_instance_id':candidate_id,
    'anchor_scene':str(anchor),'anchor_selected_id':154,'anchor_matches':overlap,'frames':frames,
    'instruction':'Review all49 crops, especially leftboundary. Missingtrack pixels are not proof that targetisphysicallyabsent.'}
(scene/'paper49_actor_candidate.json').write_text(json.dumps(result,indent=2))
sheet.save(scene/'paper49_actor_candidate_contact.jpg',quality=95)
print(json.dumps({'status':result['status'],'candidate_id':candidate_id,'anchor_min_iou':min(x['anchor_iou'] for x in overlap),'mask_present_frames':sum(x['pixels']>0 for x in frames),'contact':str(scene/'paper49_actor_candidate_contact.jpg')}),flush=True)

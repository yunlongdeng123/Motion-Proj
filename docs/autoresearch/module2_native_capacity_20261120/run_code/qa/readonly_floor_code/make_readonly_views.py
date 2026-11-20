"""Full fixed-mask views and a light readonly-audit image manifest; CPU only."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image
from readonly_floor_audit import B,HIGH,load,category_overlay,sha,write

def rgb8(x):return np.rint(np.clip(x,0,1)*255).astype(np.uint8)

def prepare(root):
    root=Path(root);d=json.loads((root/'results.json').read_text());public=json.loads((B/'public1920/export_r1/packets_manifest.json').read_text())
    packets={(x['arm'],x['raw']):x for x in public['rows']};images=[];mask_checks=[]
    if (root/'images_manifest.json').exists():raise FileExistsError('preserve image-manifest version')
    for raw in [86,110]:
        context=load(root/'contexts'/f'{raw:03d}.npz');src=load(HIGH/f'targets/{raw:03d}.npz',['C_pre_alpha','S_pre_alpha'])
        P=load(packets['G_DETAIL',raw]['path'],['road_rgb'])['road_rgb'];shape=P.shape[1:]
        if shape!=(1280,1920):raise ValueError('actual native packet required')
        for role in ['C','S']:
            domain=src[role+'_pre_alpha'];seen=context[role+'_actual_seen_context'];held=context[role+'_actual_held_context'];other=context[role+'_unqualified_pre_alpha'];hw=context[role+'_world_held']
            if not np.array_equal(domain,seen|held|other):raise ValueError('full source mask lost in context partition')
            entry=next(x for x in d['crops']if x['raw']==raw and x['source']==role);x0,y0,x1,y1=entry['bbox_xyxy']
            if int(domain[y0:y1,x0:x1].sum())!=int(domain.sum()):raise ValueError('frozen crop omits source pixels; usefull view, do not call cropfull')
            mask_checks.append(dict(raw=raw,source=role,full_pre_alpha_pixels=int(domain.sum()),inside_frozen_source_crop=int(domain[y0:y1,x0:x1].sum()),all_pixels_in_crop=True,seen_context=int(seen.sum()),held_context=int(held.sum()),unqualified=int(other.sum())))
            category=np.zeros((*shape,3),np.float32)
            for m,c in [(seen,[.10,.85,.35]),(held,[.75,.20,.90]),(other,[1.,.50,.05])]:category[m]=c
            overlay,_=category_overlay(P,domain,seen,held);world=P.transpose(1,2,0).copy();world[hw]=.62*world[hw]+.38*np.asarray([.75,.2,.9])
            views=dict(prealpha_mask=np.repeat(domain[...,None],3,axis=2).astype(float),training_status=category,training_context_overlay=overlay,world_held_overlay=world)
            for name,a in views.items():
                path=root/'contexts'/f'{raw:03d}_{role}_{name}.png';Image.fromarray(rgb8(a)).save(path)
                images.append(dict(path=str(path.relative_to(root)),raw=raw,source=role,role=name,scope='full_native_frame_full_fixed_source_mask',width=1920,height=1280,native1to1=True,SHA256=sha(path),bytes=path.stat().st_size))
            for name in entry['files']:
                path=root/name
                with Image.open(path)as im:size=im.size
                images.append(dict(path=name,raw=raw,source=role,role=Path(name).stem.split(f'{raw:03d}_{role}_',1)[1],scope='frozen_source_bbox_plus32_all_source_pixels_included',bbox_xyxy=entry['bbox_xyxy'],width=size[0],height=size[1],native1to1=True,SHA256=sha(path),bytes=path.stat().st_size))
    for path in sorted(root.glob('*_readonly_context_contact.png')):
        parts=path.name.split('_');raw=int(parts[0]);role=parts[1]
        with Image.open(path)as im:size=im.size
        images.append(dict(path=path.name,raw=raw,source=role,role='four_panel_labeled_contact',scope='same frozen source crop in four panels; all image pixels remainnative1to1 within panels',width=size[0],height=size[1],native1to1_panels=True,SHA256=sha(path),bytes=path.stat().st_size))
    write(root/'images_manifest.json',dict(schema='readonly-floor-fixed-source-views.v1',images=images,full_mask_checks=mask_checks,
        legend=dict(green='actual seen training context',purple='held stable context',orange='unqualified within complete prealpha source mask',world_held_overlay='all canonical held world blocks including lowalpha'),
        ROI_policy='all source C/S pixels, fixed bbox+32 from the preexisting source mask manifest; no strip/worst-error ROI selection',
        signals=dict(readonly_P='encoded-sRGB native premultreadonly contribution; not divided byalpha or brightness-enhanced',eligible_We_over_alpha='native ratio, displayclip[0,1]only',overlay='annotation only; numerical data unchanged'),
        source_results_SHA256=sha(root/'results.json'),code_SHA256=sha(__file__),GPU_executed=False,native_calls=0,optimizer_updates=0))
    print(json.dumps(dict(images=len(images),full_masks_verified=len(mask_checks),output=str(root/'images_manifest.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=B/'readonly_floor_audit_r1');a=p.parse_args();prepare(a.root)

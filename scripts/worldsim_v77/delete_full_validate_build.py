"""冻结产物验证和同时间窗口接缝；不再次调用模型。"""
import json,pathlib,subprocess
import numpy as np
from PIL import Image,ImageDraw
from delete_full_common import ROOT,dump
reg=json.loads((ROOT/'registration.json').read_text());assert json.loads((ROOT/'drive_state.json').read_text())['state']=='complete'
rows=[];seams=[];source_counts=[]
for s in reg['scenes']:
    panels=[]
    for stream in s['streams']:
        c=stream['camera'];base=ROOT/s['name']/f'cam{c}'
        for f in range(s['count']):
            rgb=np.array(Image.open(base/f'rgb/{f:05}.png'));mask=np.array(Image.open(base/f'mask/{f:05}.png'))>0;bg=np.array(Image.open(base/f'background/{f:05}.png'));sam=np.array(Image.open(base/f'target_sam/{f:05}.png'))>0
            assert bg.shape==rgb.shape==(576,1024,3)
            assert not (sam&~mask).any(),(s['name'],c,f,'SAM未覆盖')
            assert np.array_equal(bg[~mask],rgb[~mask]),(s['name'],c,f,'mask外改动')
            rows.append({'scene':s['name'],'camera':c,'frame':f,'mask_area':int(mask.sum()),'sam_area':int(sam.sum())})
        for w in stream['windows']:
            wd=base/'windows'/f'{w["start"]:05}';r=json.loads((wd/'result.json').read_text())
            if not r['previous_condition']:continue
            f=w['start'];kept=np.array(Image.open(base/f'background/{f:05}.png'));other=np.array(Image.open(wd/'native_00.png'));mask=np.array(Image.open(base/f'mask/{f:05}.png'))>0
            diff=np.abs(kept.astype('float32')-other.astype('float32'))
            row={'scene':s['name'],'camera':c,'overlap_frame':f,'mask_area':int(mask.sum()),'same_time_overlap_mask_rgb_mae':float(diff[mask].mean()) if mask.any() else None};seams.append(row)
            panel=Image.new('RGB',(768,160),'white')
            for i,(im,label) in enumerate([(Image.open(base/f'rgb/{f:05}.png'),'source'),(Image.fromarray(kept),'previous kept'),(Image.fromarray(np.where(mask[...,None],other,kept)),'next overlap candidate')]):
                panel.paste(im.resize((256,144)),(i*256,16));ImageDraw.Draw(panel).text((i*256+3,2),f'CAM{c} f{f} {label}',fill='black')
            panels.append(panel)
    (ROOT/'review'/s['name']).mkdir(parents=True,exist_ok=True)
    if panels:
        im=Image.new('RGB',(768,len(panels)*160),'white')
        for i,p in enumerate(panels):im.paste(p,(0,160*i))
        im.save(ROOT/'review'/s['name']/'seams.jpg',quality=93)
dump(ROOT/'build_validation.json',{'decoded_rgb_views':len(rows),'sam_covered_all':True,'outside_mask_unchanged_all':True,'seams':seams,'rows':rows,'human_verdict':None})
print('BUILD_VALIDATED',len(rows),'RGB views;',len(seams),'same-time overlap comparisons')

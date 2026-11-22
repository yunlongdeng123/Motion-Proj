"""Bind actual original PNGs and crop without resizing pixels for manual review."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from verify_joint_asset_cpu import read,write,pin

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'joint_persistent_asset_audit_v1/formal_visual_review';OUT.mkdir(parents=True,exist_ok=True)
baseline=ROOT/'report_inputs/baseline_A0_five_views';formal=ROOT/'formal400_preview_pngs'
refs=read(ROOT/'joint_persistent_asset_audit_v1/baseline_A0_png_reference.json');manifest=read(formal/'manifest.json')
mapping={'shared':'rgb','road':'road_only_rgb','actor':'actor_only_rgb'};rows=[];contact=[]
for raw in [86,110,134,179,198]:
    files={};pairs={}
    for component,name in mapping.items():
        old=baseline/f'{raw:03d}'/(name+'.png');new=formal/f'{raw:03d}_{component}.png'
        ref=next(x for x in refs['rows'] if x['raw']==raw and x['name']==name)
        expected=next(x for x in manifest['files'] if x['raw']==raw and x['component']==component)
        assert pin(old)['sha256']==ref['sha256'] and pin(new)['sha256']==expected['sha256']
        left=Image.open(old).convert('RGB');right=Image.open(new).convert('RGB');assert left.size==right.size==(1152,768)
        pairs[component]=(left,right);files[component]={'A0':pin(old),'formal400':pin(new)}
        panel=Image.new('RGB',(2304,792),'#111827');draw=ImageDraw.Draw(panel)
        draw.text((8,5),f'{raw:03d} A0 {component} | original pixels',fill='white');draw.text((1160,5),f'{raw:03d} FORMAL400 {component} | original pixels',fill='white')
        panel.paste(left,(0,24));panel.paste(right,(1152,24));panel.save(OUT/f'{raw:03d}_{component}_A0_formal_native_pair.png')
    alpha=np.asarray(Image.open(baseline/f'{raw:03d}/actor_only_alpha.png'))
    yy,xx=np.nonzero(alpha>12);bbox=(max(int(xx.min())-24,0),max(int(yy.min())-24,0),min(int(xx.max())+25,1152),min(int(yy.max())+25,768))
    lower=(max(bbox[0]-60,0),max(bbox[3]-50,0),min(bbox[2]+60,1152),min(bbox[3]+110,768))
    for component in ['shared','actor']:
        for label,crop in [('actor_focus',bbox),('underbody_focus',lower)]:
            a,b=pairs[component];a=a.crop(crop);b=b.crop(crop);panel=Image.new('RGB',(a.width*2,a.height+24),'#111827')
            ImageDraw.Draw(panel).text((6,5),f'{raw:03d} A0 | FORMAL400; native crop {crop}',fill='white');panel.paste(a,(0,24));panel.paste(b,(a.width,24))
            panel.save(OUT/f'{raw:03d}_{component}_{label}_native_pair.png')
    thumb=Image.open(OUT/f'{raw:03d}_shared_A0_formal_native_pair.png').resize((1152,396));contact.append(thumb)
    rows.append({'raw':raw,'original_wh':[1152,768],'files':files,'actor_native_crop_xyxy':bbox,'underbody_native_crop_xyxy':lower,
       'original_colour_pixels_unchanged_by_preparation':True,'baseline_remote_PNG_SHA_exact':True,'formal_preview_manifest_SHA_exact':True})
sheet=Image.new('RGB',(1152,sum(x.height for x in contact)));top=0
for item in contact:sheet.paste(item,(0,top));top+=item.height
sheet.save(OUT/'five_view_shared_A0_formal_contactsheet.png')
write(OUT/'input_bindings.json',{'schema':'R9.formal_native_visual_input_bindings.v1','complete':True,'frames':[86,110,134,179,198],
  'baseline_reference':pin(ROOT/'joint_persistent_asset_audit_v1/baseline_A0_png_reference.json'),'formal_preview_manifest':pin(formal/'manifest.json'),
  'rows':rows,'contactsheet':pin(OUT/'five_view_shared_A0_formal_contactsheet.png'),'native_pair_and_focus_crops_not_resized':True,
  'contactsheet_is_display_only':True,'GPU_forward':0,'diffusion_calls':0,'updates':0})
print('Original PNG source pins and all15 original-size component pairs ready; no RGB edited.')

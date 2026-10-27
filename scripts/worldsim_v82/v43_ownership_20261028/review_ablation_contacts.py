"""Native RGB and target-region crops; do not invent counterfactual RGB truth."""
from pathlib import Path
import json,zlib,base64
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';S=B/'outputs/GS_Scene_Diagnostics_20261027/14705303724557273004_3105_000_3125_000';D=O/'first_review_ablation_delivery_v1'
v=json.loads((S/'v43_groups_data.json').read_text());groups={g['id']:g for g in v['groups']};s=(S/'semantic_observations.js').read_text();packed=json.loads(s[s.index('=')+1:].rstrip(';'));records=[]
def unpack(d):
    data=zlib.decompress(base64.b64decode(d['data']));return {k:np.frombuffer(data[a['offset']:a['offset']+a['bytes']],np.dtype(a['dtype'])).reshape(a['shape']) for k,a in d['arrays'].items()}
for raw in [98,101,109,113,114]:
    body=unpack(packed[str(raw)])['mask'].astype(bool).reshape(320,480);ys,xs=np.where(body);box=(max(0,int(xs.min())-20),max(0,int(ys.min())-15),min(480,int(xs.max())+21),min(320,int(ys.max())+16))
    paths=[('Actor10K original',S/groups['v43_motion_no_edit']['frames'][str(raw)]['channels']['rgb']),('Delete actor only',S/groups['v43_motion_delete_target']['frames'][str(raw)]['channels']['rgb']),('Remove unknown / 0px',D/'0px/hide_unknown'/('%03d.png'%raw)),('Remove unknown / 5px',D/'5px/hide_unknown'/('%03d.png'%raw)),('Remove car+U / 0px',D/'0px/hide_both'/('%03d.png'%raw)),('Remove car+U / 5px',D/'5px/hide_both'/('%03d.png'%raw))]
    canvas=Image.new('RGB',(1440,2*540),'#0c1727');draw=ImageDraw.Draw(canvas)
    for j,(name,p) in enumerate(paths):
        im=Image.open(p).convert('RGB');x=j%3*480;y=j//3*540;draw.text((x+10,y+7),name+' / raw'+str(raw),fill='white');canvas.paste(im,(x,y+28));crop=im.crop(box);crop.thumbnail((450,170));canvas.paste(crop.resize((int(crop.width*min(450/crop.width,170/crop.height)),int(crop.height*min(450/crop.width,170/crop.height)))),(x+10,y+357))
    canvas.save(O/('ablation_raw%d.png'%raw))
    for radius in [0,5]:
        for mode in ['hide_unknown','hide_vehicle','hide_both']:
            a=np.asarray(Image.open(D/(str(radius)+'px')/mode/('%03d_alpha.png'%raw)),dtype=np.float32)/255
            records.append(dict(raw=raw,radius=radius,mode=mode,target_mask_composite_alpha_mean=float(a[body].mean()),whole_image_composite_alpha_mean=float(a.mean()),not_counterfactual_GT=True))
(O/'ablation_visual_review.json').write_text(json.dumps(dict(records=records,RGB_scores_used=False,manual_verdict=None,ghost_fully_removed=None),indent=2))
print(json.dumps(dict(contact_sheets=5,actual_native_RGB=True,diagnostic_region='original target DEVA mask, not edited truth')))

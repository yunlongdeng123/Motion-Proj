from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
B=Path(__file__).resolve().parents[2];S=B/'outputs/GS_Scene_Diagnostics_20261027/14705303724557273004_3105_000_3125_000';O=B/'outputs/V43_Ownership_20261027/gpu_v43'
s=(S/'scene.js').read_text(encoding='utf8');D=json.loads(s[s.index('=')+1:].rstrip(';'));D['groups']+=json.loads((S/'v43_groups_data.json').read_text())['groups'];groups={g['id']:g for g in D['groups']}
ids=['motion_no_edit','v43_motion_no_edit','motion_turn_left15','v43_motion_turn_left15','motion_translate_lateral_1m','v43_motion_translate_lateral_1m','v43_motion_delete_target'];raws=[91,109,114]
im=Image.new('RGB',(480*3,350*len(ids)),(12,22,36));draw=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
for i,id in enumerate(ids):
    g=groups[id]
    for j,raw in enumerate(raws):
        p=S/g['frames'][str(raw)]['channels']['rgb'];frame=Image.open(p).convert('RGB');frame.load();im.paste(frame,(j*480,i*350+30));draw.text((j*480+8,i*350+5),g['title']+' / raw'+str(raw),font=font,fill='white')
im.save(O/'first_structural_contact.jpg',quality=94)
print(O/'first_structural_contact.jpg')

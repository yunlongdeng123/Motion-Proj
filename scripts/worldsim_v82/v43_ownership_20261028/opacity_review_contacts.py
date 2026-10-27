"""Same raw frame and edit, plus fixed crop. No RGB score or case-based corrections."""
from pathlib import Path
import json
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];S=B/'outputs/GS_Scene_Diagnostics_20261027/14705303724557273004_3105_000_3125_000';O=B/'outputs/V43_Ownership_20261027/gpu_v43';v=json.loads((S/'v43_groups_data.json').read_text());groups={g['id']:g for g in v['groups']}
for raw in [98,101,109,113,114]:
    ids=[('Frozen Actor10K','v43_motion_turn_left15'),('Fixed opacity boost','v43_density2_turn'),('Legacy +10K','v43_opacity_legacy_turn_left15'),('Balanced +10K','v43_opacity_balanced_turn_left15'),('Shared field +10K','v43_opacity_field_turn_left15')]
    if 'v43_opacity_boundary_turn_left15' in groups:ids.append(('Boundary balanced +10K','v43_opacity_boundary_turn_left15'))
    c=Image.new('RGB',(1440,1080),'#0c1727');d=ImageDraw.Draw(c)
    for j,(title,gid) in enumerate(ids):
        p=S/groups[gid]['frames'][str(raw)]['channels']['rgb'];im=Image.open(p).convert('RGB');x=j%3*480;y=j//3*540;d.text((x+8,y+8),title+' / raw'+str(raw),fill='white');c.paste(im,(x,y+28));crop=im.crop((190,140,390,265)).resize((280,175));c.paste(crop,(x+8,y+357))
    c.save(O/('opacity_turn_raw%d.png'%raw))
print('Opacity native turn contact sheets written')

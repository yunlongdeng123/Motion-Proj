from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import json,hashlib,datetime
B=Path(__file__).resolve().parents[2];O=B/'outputs/GS_Scene_Diagnostics_20261027';scene=O/'14705303724557273004_3105_000_3125_000';broken=[];checked=0
class Parser(HTMLParser):
    def handle_starttag(self,tag,attrs):
        global checked
        for k,v in attrs:
            if k not in ['src','href'] or not v:continue
            u=urlsplit(v)
            if u.scheme or not u.path:continue
            p=(self.root/unquote(u.path)).resolve();checked+=1
            if not p.exists():broken.append(str(p))
for p in [O/'index.html',scene/'index.html']:
    parser=Parser();parser.root=p.parent;parser.feed(p.read_text(encoding='utf8'))
s=(scene/'index.html').read_text(encoding='utf8');assert s.count('src="v43_groups.js')==1
g=json.loads((scene/'v43_groups_data.json').read_text());frames=0;images=0
from PIL import Image
for row in g['groups']:
    assert len(row['frames'])==(49 if row['kind'] in ['v43_structure','v43_bypass'] or (row['kind']=='v43_opacity_calibration' and row['id'].endswith('_no_edit')) else 31)
    for f in row['frames'].values():
        frames+=1
        for p in f['channels'].values():
            file=scene/p
            with Image.open(file) as im:assert im.size==(480,320);im.verify()
            assert file.with_name(file.name+'.js').exists();images+=1
assert not broken,broken
r=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),html_links_checked=checked,broken=broken,v43_groups=len(g['groups']),frame_records=frames,decoded_images=images,single_group_script=True,geometry_validation='first_motion_viewer_validation.json',browserQA=False,browser_limitation='Browser UI policy disallows file protocol; validated local files and actual Canvas runtime instead.',quality_accepted=False,human_verdict=None)
out=B/'outputs/V43_Ownership_20261027/gpu_v43/first_delivery_validation.json';out.write_text(json.dumps(r,indent=2));print(json.dumps(r))

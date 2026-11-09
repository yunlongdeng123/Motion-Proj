from pathlib import Path
from html.parser import HTMLParser
import json,hashlib,gzip,base64,re,subprocess
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];OUT=BASE/'outputs/waymo_val065_experiment'
PLATFORM=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a/outputs/GS_Scene_Diagnostics_20261027');CONTEXT='15021599536622641101_556_150_576_150'
class Page(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.paths=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        for k in ['src','href']:
            v=a.get(k)
            if v and not v.startswith(('#','http:','https:','file:','data:')):self.paths.append(v)
def main():
    for p in [OUT/'index.html',OUT/'report.html',PLATFORM/CONTEXT/'index.html']:
        text=p.read_text(encoding='utf8');assert '@@' not in text;v=Page();v.feed(text);assert len(v.ids)==len(set(v.ids));assert all((p.parent/x).exists() for x in v.paths),p
    root=Page();root.feed((PLATFORM/'index.html').read_text(encoding='utf8'));assert root.links.count(f'{CONTEXT}/index.html')==1
    backup=Path(json.loads((HERE/'build_receipt.json').read_text(encoding='utf8'))['backup']);old=Page();old.feed((backup/'index.html').read_text(encoding='utf8'));assert set(old.links)<=set(root.links)
    manifest=json.loads((HERE/'retrieval_manifest.json').read_text(encoding='utf8'))
    for row in manifest['files']:
        p=OUT/row['member'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        if row['member'].startswith('viewer/'):
            assert hashlib.sha256((PLATFORM/CONTEXT/row['member']).read_bytes()).hexdigest()==row['sha256']
        if p.suffix=='.png':
            with Image.open(p) as im:im.verify()
    for raw in [30,60,150,198]:
        text=(OUT/'viewer/layers'/f'{raw}.js').read_text(encoding='utf8');z=json.loads(text.split(f'[{raw}]=',1)[1].removesuffix(';'))
        for state in ['keep','delete']:
            arr=np.frombuffer(gzip.decompress(base64.b64decode(z[state])),dtype='<f4').reshape(9,320,480);source=np.load(OUT/'evidence/key_float'/str(raw)/f'{state}.npz')
            assert np.array_equal(arr,np.concatenate([source['group_alpha'],source['alpha'],source['depth_sum'],source['rgb']]))
    for raw in [30,60,150,198]:
        text=(OUT/'viewer/layers'/f'{raw}.js').read_text(encoding='utf8');z=json.loads(text.split(f'[{raw}]=',1)[1].removesuffix(';'))
        for state in ['keep','delete']:
            arr=np.frombuffer(gzip.decompress(base64.b64decode(z['native_'+state])),dtype='<f4').reshape(9,320,480);source=np.load(OUT/'evidence/native_float'/str(raw)/f'{state}.npz')
            assert np.array_equal(arr,np.concatenate([source['group_alpha'],source['alpha'],source['depth_sum'],source['rgb']]))
    subprocess.run(['node','--check',str(HERE/'viewer.js')],check=True);subprocess.run(['node',str(HERE/'test_viewer.cjs')],check=True)
    receipt=dict(status='PASS_RETRIEVAL_RESOURCES_FLOATS_INTERACTION_LOGIC',frames=199,original_image_pngs=995,float_keys=[30,60,150,198],old_links_preserved=True,scene_entry_unique=True,raws_heldout=[79,118],browser_visual_acceptance='未执行；此前file协议被拒绝，未尝试绕过',human_verdict=None,automatic_shutdown=False)
    for p in [HERE/'verification.json',OUT/'viewer_validation.json',PLATFORM/CONTEXT/'viewer_validation.json']:p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()

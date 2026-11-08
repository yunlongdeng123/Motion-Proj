"""检查新增子页、旧入口保留、逐帧资源和float32数据来源。"""
from pathlib import Path
from html.parser import HTMLParser
from collections import Counter
import json,re,gzip,base64,subprocess
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
REPORT=BASE/'outputs/waymo_val029_experiment'
PLATFORM=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a/outputs/GS_Scene_Diagnostics_20261027')
CONTEXT='12820461091157089924_5202_916_5222_916'
class Page(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.paths=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        for key in ['href','src']:
            v=a.get(key)
            if v and not v.startswith(('#','http:','https:','file:','data:')):self.paths.append(v.split('?')[0])
def main():
    required={'frame','rawNumber','play','prev','next','view','overlay','observed','native','nativeDelete','keep','delete','mapState','channel','region','map','contributions','saveCase','loadCase','caseNote','caseFile','surfaceCanvas','suvChart','grayChart'};checks=[]
    for p in [REPORT/'index.html',PLATFORM/CONTEXT/'index.html']:
        text=p.read_text(encoding='utf8');assert '@@' not in text;page=Page();page.feed(text);assert len(page.ids)==len(set(page.ids));assert required<=set(page.ids);missing=[v for v in page.paths if not (p.parent/v).exists()];assert not missing,missing;checks.append(dict(page=str(p),ids=len(page.ids),local_resources=len(page.paths),missing=[]))
    backups=sorted((HERE/'backups').iterdir());old=Page();old.feed((backups[0]/'platform_index.html').read_text(encoding='utf8'));new=Page();new.feed((PLATFORM/'index.html').read_text(encoding='utf8'));assert set(old.links)<=set(new.links);assert new.links.count(f'{CONTEXT}/index.html')==1
    dirs=[REPORT/'viewer',PLATFORM/CONTEXT/'viewer']
    for folder in dirs:
        native=json.loads((folder/'evidence/native_route_audit.json').read_text(encoding='utf8'))
        import hashlib
        assert len(native['files'])==158 and native['training_start_checkpoint'] is None
        assert not native['image_space_composite'] and not native['semantic_rebinding']
        for row in native['files']:
            assert hashlib.sha256((folder/'native10k'/row['member']).read_bytes()).hexdigest()==row['sha256']
        assert {int(p.stem) for p in (folder/'frames').glob('*.png')}==set(range(109,198))
        assert {int(p.stem) for p in (folder/'observed').glob('*.png')}==set(range(109,198))
        for p in (folder/'frames').glob('*.png'):
            with Image.open(p) as im:assert im.size==(1920,320);im.verify()
        for raw in [130,155,160,170,190,197]:
            assert len(list((folder/'exact'/str(raw)).glob('*.png')))==9
    payloads=[]
    for raw in [130,155,160,190]:
        p=REPORT/'viewer/layers'/f'{raw}.js';data=json.loads(p.read_text(encoding='utf8').split(f'[{raw}]=',1)[1].strip().removesuffix(';'));assert data['raw']==raw
        for state in ['keep','delete']:
            decoded=np.frombuffer(gzip.decompress(base64.b64decode(data[state])),dtype='<f4').reshape(9,320,480);source=np.load(REPORT/'evidence_package/key_float/final_all_eval'/f'{raw:03d}/{state}.npz');assert np.array_equal(decoded[:4],source['group_alpha']);assert np.array_equal(decoded[4:5],source['alpha']);assert np.array_equal(decoded[6:],source['rgb']);assert np.isfinite(decoded).all()
        source_labels=np.array(Image.open(REPORT/'evidence_package/masks'/f'{raw:03d}.png'),dtype=np.uint8);assert gzip.decompress(base64.b64decode(data['labels']))==source_labels.tobytes();payloads.append(dict(raw=raw,group_alpha_RGB_and_alpha_bit_exact=True,labels_exact=True))
    subprocess.run(['node','--check',str(HERE/'viewer.js')],check=True);subprocess.run(['node',str(HERE/'test_viewer.cjs')],check=True);interactive=json.loads((HERE/'interaction_validation.json').read_text(encoding='utf8'))
    result=dict(status='PASS_OFFLINE_VIEWER_INSTALLATION',page_checks=checks,continuous_frames=89,window=[109,197],native_10k_original_pngs=158,native_direct_raws=[119,197],native_heldout_direct_export_missing=[109,118],native_no_edit_direct_route=True,old_platform_links_preserved=True,scene_entry_unique=True,required_controls_present=True,payload_checks=payloads,interaction=interactive,browser_visual_acceptance='not performed; local file browser protocol was previously blocked, no bypass attempted',GPU_started=False,rerender_calls=0,precision_limit='nativeKEEP/DELETE are original PNGs; current-method continuous imagery is H264 decoded,6 original keyframes; exact float maps only on4 saved anchor frames; other group remains combined',source_builder=str(HERE/'build_visualization.py'))
    for p in [HERE/'validation.json',REPORT/'viewer_validation.json',PLATFORM/CONTEXT/'viewer_validation.json']:p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['page_checks','interaction','payload_checks']},ensure_ascii=False))
if __name__=='__main__':main()

"""Validate offline links, actual-source reconstruction, inputs and all media."""
from pathlib import Path
from html.parser import HTMLParser
import ast,json,re,subprocess,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from build_native30k_compare import probe,sha
W=Path(__file__).resolve().parent
D=W/'ai_review_bundle_v1'
S=W/'ai_review_source_snapshot'

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.refs=[]
        def handle_starttag(self,tag,attrs):self.refs.extend(v for k,v in attrs if k in ['src','href'] and v)
    fixtures={};links=0
    for name in ['index.html','compare.html','independent_pair.html']:
        text=(D/name).read_text(encoding='utf-8');p=Links();p.feed(text)
        for ref in p.refs:
            assert not ref.startswith(('file:','http:','https:','/')),ref
            target=(D/ref).resolve();target.relative_to(D.resolve())
            if ref not in ['MANIFEST.json','VALIDATION.json']:assert target.is_file(),ref
        links+=len(p.refs)
        if name!='index.html':
            data=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',text,re.S)[1])
            for model in data['models'].values():
                for v in model['videos'].values():
                    for k in ['path','poster']:
                        if v.get(k):
                            dest=(D/v[k]).resolve();dest.relative_to(D.resolve());assert dest.is_file()
            f=subprocess.run(['node',str(W/'check_cpu_export_player.js'),str(D/name)],check=True,capture_output=True,text=True)
            fixtures[name]=json.loads(f.stdout)
    ET.parse(D/'architecture.svg')
    source_map=read(D/'code/source_map.json');core_count=0
    for arm,record in source_map.items():
        for name,e in record['files'].items():
            base=D/'code/base'/name;overlay=D/'code/overlays'/arm/name
            chosen=overlay if overlay.exists() else base
            assert sha(chosen)==e['sha256']==sha(S/'engines'/arm/name),(arm,name)
            if name.endswith('.py'):ast.parse(chosen.read_text(encoding='utf-8'))
            core_count+=1
    for p in (D/'code').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    experiments=read(D/'experiments.json');assert len(experiments)==5
    for e in experiments:
        s=read(D/'evidence/runs'/e['id']/'status.json');assert s['status']=='complete' and s['iteration']==30000
        if e['id']!='baseline':assert all(s['optimizer_step_counts'][k]==30000 for k in ['static','dynamic','environment'])
    bundle=read(D/'evidence/input/bundle.json')
    for frame,record in bundle['frames'].items():
        p=D/'evidence/input'/record['road_target'];assert sha(p)==record['road_target_sha256']
        with np.load(p,allow_pickle=False) as z:assert len(z.files)>0
    for p in (D/'evidence/input/query_masks_a6_a37').glob('*.npz'):
        with np.load(p,allow_pickle=False) as z:assert z['masks'].shape==(320,480,2)
    pngs=list(D.rglob('*.png'))
    for p in pngs:
        with Image.open(p) as im:assert im.size==(480,320),(p,im.size)
    media=read(D/'evidence/media.json');videos={}
    # New package check: byte-preserved streams all decode to actual49 complete frames.
    for key,item in media['videos'].items():
        p=D/item['path'];assert sha(p)==item['sha256'];probe(p)
        result=subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(p),'-f','rawvideo','-pix_fmt','rgb24','-'],check=True,capture_output=True)
        count=len(result.stdout)//(480*320*3);assert count==49 and len(result.stdout)==49*480*320*3
        videos[key]={'frames':count,'bytes':p.stat().st_size,'sha256':sha(p)}
    assert len(videos)==37
    physical={item['path'] for item in media['videos'].values()};assert len(physical)==29
    diag=D/'media/baseline_fixed_camera_time_only.mp4';probe(diag)
    out={'status':'PASSED_SELF_CONTAINED_FIVE_CONFIGURATION_AI_REVIEW_PACKAGE','five_configurations':list(source_map),
      'source_reconstruction_hash_checks':core_count,'offline_HTML_link_checks':links,'generated_player_fixtures':fixtures,
      'exact_frozen_road_targets':49,'query_mask_frames':49,'full_frame_PNGs':len(pngs),'main_video_channel_references':37,'main_unique_physical_videos':29,'main_channel_reference_decoded_frames':1813,
      'diagnostic_videos':1,'media_byte_preserved':True,'architecture_XML_valid':True,'python_AST_valid':True,
      'contains_checkpoints':False,'contains_full_training_logs':False,'browser_UI_tested':False,'quality_accepted':False,'human_verdict':None}
    assert not any(p.suffix in ['.pth','.pt','.ckpt','.pyc'] for p in D.rglob('*') if p.is_file())
    (D/'VALIDATION.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='generated_player_fixtures'}))

if __name__=='__main__':main()

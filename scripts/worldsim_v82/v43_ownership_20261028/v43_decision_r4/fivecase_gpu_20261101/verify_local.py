from pathlib import Path
import concurrent.futures, hashlib, json, shutil, subprocess
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import numpy as np

W=Path(__file__).resolve().parent
OUT=W.parent.parent/'outputs/V43_R4_FiveCase_GPU_20261101'
def load(p): return json.loads(p.read_text(encoding='utf-8'))
class Parser(HTMLParser):
    def __init__(self): super().__init__();self.refs=[];self.ids=[];self.scripts=[];self.inscript=False
    def handle_starttag(self,t,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        for k in ['href','src']:
            if k in a:self.refs.append(a[k])
        if t=='script':self.inscript='src' not in a
    def handle_endtag(self,t):
        if t=='script':self.inscript=False
    def handle_data(self,s):
        if self.inscript:self.scripts.append(s)
def media(p):
    args=[shutil.which('ffprobe'),'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate,width,height','-of','json',str(p)]
    j=json.loads(subprocess.check_output(args))['streams'][0]
    assert int(j['nb_read_frames'])==49 and j['r_frame_rate']=='10/1',(p,j)
    assert j['width']==960 and j['height']==320,(p,j)
    return dict(file=str(p.relative_to(OUT)),**j,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
    cases=load(OUT/'summary.json')['cases'];assert len(cases)==5
    ledger=load(OUT/'mechanisms/mechanism_ledger.json');assert ledger['record_count']==40 and not ledger['all_mechanisms_isolated']
    assert all(not x['ownership_training_label_permission'] for x in ledger['rows'])
    masks=list((OUT/'mechanisms').rglob('regions/*.npz'));assert len(masks)==245
    for p in masks:
        z=np.load(p);assert not z['ownership_training_label_allowed'].any()
    arrays=list((OUT/'fp32_controls').rglob('*.npz'));assert len(arrays)==20
    closure=0.0
    for p in arrays:
        z=np.load(p)
        assert all(np.isfinite(z[k]).all() for k in z)
        closure=max(closure,float(np.abs(z['final']-z['C_G']-z['fill']).max()),float(np.abs(z['fill']-(1-z['alpha'])*z['E']).max()))
    assert closure<2e-6,closure
    data=json.loads((OUT/'review/data.js').read_text(encoding='utf-8')[len('window.FIVE_DATA='):-1])
    refs=0
    for c in data:
        assert len(c['frames'])==49 and c['raw_context'] and c['raw_track']
        for d in c['domains'].values():
            assert len(d['images'])==d['routes']*4*3
            for im in d['images']:
                assert (OUT/'review'/im['path']).is_file(),im;refs+=1
        for v in c['video_pairs']:
            for k in ['path','boxed_path']:assert (OUT/'review'/v[k]).is_file()
    html_checked=[]
    for p in [OUT/'report.html',OUT/'mechanisms/index.html',OUT/'review/index.html']:
        parser=Parser();parser.feed(p.read_text(encoding='utf-8'))
        assert len(parser.ids)==len(set(parser.ids)),p
        for ref in parser.refs:
            r=urlsplit(ref)
            if r.scheme or not r.path:continue
            assert (p.parent/unquote(r.path)).exists(),(p,ref)
        for i,script in enumerate(parser.scripts):
            f=W/(p.parent.name+'_'+p.stem+f'_{i}.js');f.write_text(script,encoding='utf-8')
            subprocess.run(['node','--check',str(f)],check=True)
        html_checked.append(str(p.relative_to(OUT)))
    subprocess.run(['node','--check',str(OUT/'source/edit_comparison_entry.js')],check=True)
    subprocess.run(['node',str(W/'verify_viewer.js'),str(OUT)],check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        videos=list(pool.map(media,sorted((OUT/'review/videos').glob('*.mp4'))))
    assert len(videos)==40
    publish=load(OUT/'publishing_receipt.json');assert publish['shared_root_unchanged']
    for row in publish['entries']:
        target=Path(row['path']);index=(target/'index.html').read_text(encoding='utf-8')
        assert index.count('src="edit_comparison_entry.js"')==1
        dest=target/'edit_comparisons'
        assert not (dest/'report.html').exists() and not (dest/'mechanisms').exists()
        assert '../index.html' in (dest/'index.html').read_text(encoding='utf-8')
        for p in (OUT/'review').rglob('*'):
            if p.is_file() and p.name!='index.html':assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((dest/p.relative_to(OUT/'review')).read_bytes()).digest()
    result=dict(result='PASS',cases=5,mechanism_records=40,mechanism_isolation_complete=False,all_245_label_masks_disabled=True,fp32_files=20,compositing_closure_max_abs=closure,anchor_image_references=refs,videos=videos,html_checked=html_checked,viewer_navigation='Node DOM-harness; all cases/domains/routes/operations/anchors',browser_interaction_tested=False,shared_root_unchanged=True,notes='JPEG/H264 for browsing only; complete native FP32 and restore states remain remote.')
    (OUT/'delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('VERIFIED',len(videos),'videos;',refs,'anchor images; 245 disabled masks; closure',closure)
if __name__=='__main__':main()

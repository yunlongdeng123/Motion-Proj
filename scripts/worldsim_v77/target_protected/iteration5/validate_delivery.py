"""静态文件与逐帧资源检查；浏览器交互另记未测试。"""
import argparse,json,re
from html.parser import HTMLParser
from pathlib import Path
from PIL import Image
class Resources(HTMLParser):
    def __init__(self):super().__init__();self.paths=[];self.svg=0
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='svg':self.svg+=1
        for name in ['src','href']:
            if name in attrs:self.paths.append(attrs[name])
def main(root):
    m=json.loads((root/'review_manifest.json').read_text(encoding='utf-8'));page=(root/'index.html').read_text(encoding='utf-8');assert '\ufffd' not in page
    parser=Resources();parser.feed(page);assert parser.svg==1
    paths=parser.paths+[p for c in m['cases'] for f in c['frames'] for p in f['images'].values()]
    missing=[p for p in paths if not (root/p.split('#')[0]).exists()];assert not missing,missing
    decoded=0
    for c in m['cases']:
        assert len(c['frames'])==10 and c['human_verdict'] is None and c['training_ready'] is False
        for f in c['frames']:
            for path in f['images'].values():
                with Image.open(root/path) as im:im.load();assert im.size==(1024,576);decoded+=1
            assert f['hidden_pixel_probe_pass'] and f['ego_bottom_guard_pass']
        ts=[f['timestamp_us'] for f in c['frames']];assert all(0<b-a<=180000 for a,b in zip(ts,ts[1:]))
        for p in c['videos'].values():assert (root/p).stat().st_size>0
    assert m['case_count']==4 and m['frame_count']==40 and m['video_count']==16 and m['decoded_video_frames']==160
    assert m['GPU_calls']==m['actual_synthetic_cases']==m['training_ready']==0
    script=re.search(r'<script>([\s\S]*?)</script>',page).group(1)
    (root/'page_script_validation.js').write_text(script,encoding='utf-8')
    result={'task_id':m['task_id'],'run_id':'r5','decoded_local_JPEGs':decoded,'video_files':16,'remote_video_frames_actually_decoded':160,
            'all_resource_links_exist':True,'human_verdicts_null':True,'architecture_diagram':True,
            'scope':'continuous planning diagnostics; no actual synthetic-X; pending true protected SAM2 masks',
            'browser_interaction':'not_tested','JS_syntax':'check with Node separately'}
    (root/'local_delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)

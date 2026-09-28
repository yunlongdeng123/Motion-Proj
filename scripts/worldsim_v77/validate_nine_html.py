"""实解码全部视频，并检查三个视图/两种背景模式的动态路径。"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import argparse,json,subprocess,xml.etree.ElementTree as ET
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();R=a.output
class Parser(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=[];self.videos=0
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d:self.ids.append(d['id'])
        if tag=='video':self.videos+=1
        for k in ['src','href','poster']:
            if k in d:self.links.append(d[k])
t=Parser();t.feed((R/'index.html').read_text(encoding='utf-8'));assert t.videos==27 and len(set(t.ids))==9
for link in t.links:
    u=urlparse(link)
    if u.scheme or u.netloc:continue
    if u.path:assert (R/unquote(u.path)).exists(),link
    elif u.fragment:assert u.fragment in t.ids,link
reg=json.loads((R/'evidence/effective_registration.json').read_text(encoding='utf-8'));dynamic=[]
display={r['scene']:r['camera'] for r in json.loads((R/'evidence/display_registration.json').read_text(encoding='utf-8'))}
for s in reg['scenes']:
    extras=[f"cam{v['camera']}_" for v in s['streams'] if v['active'] and v['camera']!=display[s['name']]]
    for view in ['','zoom_','six_']+extras:
        for k in ['original','factual','delete','hybrid_factual','hybrid_delete']:
            path=f"media/{s['name']}/{view}{k}.mp4";assert (R/path).exists(),path;dynamic.append(path)
ET.parse(R/'architecture.svg');pictures=[]
for f in R.rglob('*'):
    if f.suffix.lower() in ['.jpg','.png']:
        with Image.open(f) as im:im.verify()
        pictures.append(str(f.relative_to(R)))
videos=[]
for f in sorted(R.rglob('*.mp4')):
    subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(f),'-f','null','-'],check=True,capture_output=True)
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(f)]));v=data['streams'][0];assert int(v['nb_read_frames'])==30
    videos.append(dict(path=str(f.relative_to(R)),**v))
assert len(videos)==193
subprocess.run(['node','--check',str(R/'review.js')],check=True)
out=dict(task_id=reg['task_id'],run_id='r1',default_columns=27,scenes=9,local_refs=len(t.links),dynamic_paths_verified=len(dynamic),images_verified=len(pictures),videos=videos,total_decoded_frames=sum(int(v['nb_read_frames']) for v in videos),js_syntax='pass',browser_interaction_tested=False,human_verdict=None)
(R/'validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print('NINE_HTML_VALIDATED',len(videos),out['total_decoded_frames'],len(pictures),flush=True)

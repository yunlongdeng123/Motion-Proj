"""验证离线引用、JS语法和每个视频的实际帧数，不冒称浏览器交互通过。"""
import argparse,json,subprocess,tempfile
from pathlib import Path
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory).resolve()
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.refs=[];self.ids=[];self.js=[];self.script=False
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if 'id' in d:self.ids.append(d['id'])
  self.refs.extend(d[k] for k in ['src','href','poster'] if k in d)
  if tag=='script':self.script=True
 def handle_endtag(self,tag):
  if tag=='script':self.script=False
 def handle_data(self,data):
  if self.script:self.js.append(data)
x=Parser();x.feed((root/'index.html').read_text(encoding='utf-8'));assert len(x.ids)==len(set(x.ids))
for ref in x.refs:
 if ref.startswith(('http://','https://','mailto:')):continue
 if ref.startswith('#'):assert ref[1:] in x.ids;continue
 assert (root/ref.split('#')[0]).is_file(),ref
with tempfile.TemporaryDirectory(prefix='v77-temporal-js-') as d:
 f=Path(d)/'inline.js';f.write_text('\n'.join(x.js),encoding='utf-8');subprocess.run(['node','--check',str(f)],check=True)
expected={v['file']:v for v in json.loads((root/'video_validation.json').read_text())['videos']};files=sorted(root.rglob('*.mp4'));assert {f.relative_to(root).as_posix() for f in files}==set(expected)
def check(f):
 key=f.relative_to(root).as_posix();s=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(f)],text=True))['streams'][0]
 assert (s['width'],s['height'],s['r_frame_rate'],int(s['nb_read_frames']))==(1024,576,'10/1',expected[key]['decoded_frames']);return dict(file=key,**s)
with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(check,files))
for f in root.rglob('*.json'):json.loads(f.read_text(encoding='utf-8-sig'))
result=dict(local_references_checked=len(x.refs),inline_js_syntax='passed',local_videos_decoded=len(rows),decoded_frames=sum(int(r['nb_read_frames']) for r in rows),videos=rows,browser_interaction='not tested; no browser workaround used',human_verdict=None)
(root/'review_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print({k:v for k,v in result.items() if k!='videos'})

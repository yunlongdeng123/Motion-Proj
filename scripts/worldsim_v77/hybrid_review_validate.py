"""审核包静态引用/JS语法和传输后的实际视频解码检查；不冒称浏览器QA。"""
import argparse,json,subprocess,tempfile
from html.parser import HTMLParser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args();root=Path(args.directory).resolve()
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.refs=[];self.ids=[];self.in_script=False;self.js=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  for k in ['src','href','poster']:
   if k in a:self.refs.append(a[k])
  if tag=='script':self.in_script=True
 def handle_endtag(self,tag):
  if tag=='script':self.in_script=False
 def handle_data(self,data):
  if self.in_script:self.js.append(data)
x=Parser();x.feed((root/'index.html').read_text(encoding='utf-8'));checked=[]
assert len(x.ids)==len(set(x.ids))
for ref in x.refs:
 if ref.startswith(('https://','http://','mailto:')):continue
 if ref.startswith('#'):assert ref[1:] in x.ids;continue
 f=(root/ref.split('#')[0]).resolve();assert f.is_file(),f;checked.append(ref)
with tempfile.TemporaryDirectory(prefix='v77-review-') as td:
 js=Path(td)/'review.js';js.write_text('\n'.join(x.js),encoding='utf-8');subprocess.run(['node','--check',str(js)],check=True)
files=sorted(root.rglob('*.mp4'));assert len(files)==24
def verify(f):
 cmd=['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(f)]
 stream=json.loads(subprocess.check_output(cmd,text=True))['streams'][0]
 assert (stream['width'],stream['height'])==(960,536)
 assert stream['nb_read_frames']=='30' and stream['r_frame_rate']=='10/1'
 return dict(file=f.relative_to(root).as_posix(),**stream)
with ThreadPoolExecutor(max_workers=3) as pool:videos=list(pool.map(verify,files))
for f in root.rglob('*.json'):json.loads(f.read_text(encoding='utf-8'))
result=dict(local_references_checked=len(checked),inline_js_syntax='passed',local_videos_decoded=len(videos),decoded_frames=720,videos=videos,browser_interaction='not tested: CUA file URL security policy blocked preview; no workaround attempted',human_verdict=None)
(root/'review_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:v for k,v in result.items() if k!='videos'},ensure_ascii=False))

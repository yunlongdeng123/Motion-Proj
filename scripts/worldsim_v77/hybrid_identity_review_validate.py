"""审核HTML静态引用/JS与传输后视频实际解码；不冒称浏览器QA。"""
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
 def handle_data(self,x):
  if self.script:self.js.append(x)
x=Parser();x.feed((root/'index.html').read_text(encoding='utf-8'));assert len(x.ids)==len(set(x.ids))
for ref in x.refs:
 if ref.startswith(('http://','https://','mailto:')):continue
 if ref.startswith('#'):assert ref[1:] in x.ids;continue
 if ref=='review_validation.json':continue # 本命令完成后写入的检查结果。
 assert (root/ref.split('#')[0]).is_file(),ref
with tempfile.TemporaryDirectory(prefix='v77-stage-js-') as d:
 f=Path(d)/'inline.js';f.write_text('\n'.join(x.js),encoding='utf-8');subprocess.run(['node','--check',str(f)],check=True)
manifest=json.loads((root/'video_validation.json').read_text(encoding='utf-8'));files=sorted(root.rglob('*.mp4'));assert len(files)==len(manifest['videos'])==8
def check(f):
 stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(f)],text=True))['streams'][0]
 assert (stream['width'],stream['height'],stream['r_frame_rate'],stream['nb_read_frames'])==(960,536,'10/1','30'),(f,stream)
 return dict(file=f.relative_to(root).as_posix(),**stream)
with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(check,files))
for f in root.rglob('*.json'):json.loads(f.read_text(encoding='utf-8'))
result=dict(local_references_checked=len(x.refs),inline_js_syntax='passed',local_videos_decoded=len(rows),decoded_frames=30*len(rows),videos=rows,browser_interaction='not tested; earlier CUA URL policy denied file preview, no workaround',human_verdict=None)
(root/'review_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');print({k:v for k,v in result.items() if k!='videos'})

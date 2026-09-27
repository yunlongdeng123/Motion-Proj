from pathlib import Path
import argparse,json,subprocess,xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
from PIL import Image
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'outputs/v77-expansion');args=ap.parse_args();ROOT=args.output
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d:self.ids.add(d['id'])
        for k in ['href','src']:
            if k in d:self.links.append(d[k])
parser=Links();parser.feed((ROOT/'index.html').read_text(encoding='utf-8'));missing=[]
for link in parser.links:
    url=urlparse(link)
    if url.scheme or url.netloc:continue
    if url.path and not (ROOT/unquote(url.path)).exists():missing.append(link)
    elif not url.path and url.fragment not in parser.ids:missing.append(link)
assert not missing,missing
ET.parse(ROOT/'architecture.svg');images=[]
for p in ROOT.rglob('*'):
    if p.suffix.lower() in ['.png','.jpg']:
        with Image.open(p) as im:im.verify()
        images.append(str(p.relative_to(ROOT)))
videos=[]
for p in sorted(ROOT.rglob('*.mp4')):
    subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(p),'-f','null','-'],check=True,capture_output=True)
    r=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(p)]));stream=r['streams'][0];videos.append(dict(path=str(p.relative_to(ROOT)),**stream))
subprocess.run(['node','--check',str(ROOT/'review.js')],check=True)
doc=dict(local_refs=len(parser.links),missing=missing,images_verified=len(images),videos=videos,total_decoded_frames=sum(int(v['nb_read_frames']) for v in videos),js_syntax='pass',browser_interaction_tested=False,human_verdict=None)
(ROOT/'validation.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8');print('HTML_VALIDATED',len(videos),doc['total_decoded_frames'],len(images))

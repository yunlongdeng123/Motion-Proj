"""Read-only file/media checks and JS fixture; no browser or GPU operations."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
from urllib.request import url2pathname
import json,subprocess,re,hashlib,datetime,shutil
from PIL import Image
BASE=Path(__file__).resolve().parents[2];WORK=BASE/'work/shadow_sequence_v2';OUT=BASE/'outputs';E=OUT/'shadow_sequence_v2';PAGE=OUT/'DrivingEditor_shadow_sequence_comparison.html'
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.refs=[];self.ids=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  for k in ['src','href']:
   if a.get(k):self.refs.append(a[k])
html=PAGE.read_text(encoding='utf-8');parser=Parser();parser.feed(html);assert len(parser.ids)==len(set(parser.ids))
refs=[]
for value in parser.refs:
 if value.startswith('file:'):p=Path(url2pathname(unquote(urlparse(value).path)))
 else:p=(PAGE.parent/value).resolve()
 assert p.exists(),p
 refs.append(p)
data=json.loads(re.search(r'<script id="comparison-data" type="application/json">(.*?)</script>',html,re.S).group(1));images=set()
for row in data['frames']:
 for k in ['comparison','gt','old_delete']:
  p=(OUT/row[k]).resolve();assert p.exists();images.add(p)
for p in images:
 with Image.open(p) as im:im.load();assert im.size==((960,688) if p.parent.name=='comparison_frames' else (480,320)),p
node=subprocess.run(['node',str(WORK/'check_comparison_interactions.cjs'),str(PAGE)],capture_output=True,text=True,encoding='utf-8',check=True)
interaction=json.loads(node.stdout);assert interaction['interaction_fixture']=='PASS'
videos=[]
for p in interaction['video_files']:
 r=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=codec_name,nb_read_frames,width,height','-of','json',p],capture_output=True,text=True,check=True)
 s=json.loads(r.stdout)['streams'][0];assert s['codec_name']=='h264' and int(s['nb_read_frames'])==49;videos.append({'file':p,**s})
entry=json.loads((E/'evidence/unified_comparison_entry.json').read_text(encoding='utf-8'));unified=Path(entry['unified']);s=unified.read_text(encoding='utf-8-sig');pattern=r'<!-- SHADOW_SEQUENCE_V2_ENTRY_START -->.*?<!-- SHADOW_SEQUENCE_V2_ENTRY_END -->';block=re.search(pattern,s,re.S).group(0)
old=Path(entry['backup']).read_text(encoding='utf-8-sig');assert s.replace(block,'',1)==old # first insertion only
assert s.count('id="shadow-sequence-v2-entry"')==1 and PAGE.as_uri() in block
record={'status':'PASS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'images_decoded':len(images),'static_links_checked':len(refs),'videos_checked':videos,'interaction':interaction,'legacy_content_preserved_exactly':True,'unified_entry_unique':True,'browser_visual_QA':'not performed: browser policy disallows file:// URLs; no workaround attempted','GPU_operations':0}
(E/'evidence/comparison_page_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
for p in [WORK/'build_comparison_entry.py',WORK/'validate_comparison_entry.py',WORK/'check_comparison_interactions.cjs',WORK/'comparison_template.html']:shutil.copy2(p,E/'scripts'/p.name)
m_path=E/'manifest.json';m=json.loads(m_path.read_text(encoding='utf-8'));m['updated_utc']=record['created_utc']
for p in E.rglob('*'):
 if p.is_file() and p.name!='manifest.json':m['files'][p.relative_to(E).as_posix()]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
m_path.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':record['status'],'images_decoded':len(images),'videos_checked':len(videos),'interaction_states':interaction['frame_mode_states_checked'],'legacy_preserved':True,'browser_visual_QA':record['browser_visual_QA']},ensure_ascii=False))

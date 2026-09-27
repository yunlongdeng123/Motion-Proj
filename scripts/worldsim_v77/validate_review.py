import argparse,json,pathlib,re,subprocess,tempfile
from html.parser import HTMLParser
BASE=pathlib.Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=pathlib.Path,default=BASE/'outputs/v77-delete-full');OUT=parser.parse_args().output
class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['src','href'] and v and not v.startswith(('#','http')):self.refs.append(v)
p=Links();p.feed((OUT/'index.html').read_text(encoding='utf-8'));missing=[v for v in p.refs if not (OUT/v).exists()];assert not missing,missing
summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'));expect={s['scene']:s['frame_count'] for s in summary['scenes']};videos=[]
for path in sorted(OUT.glob('scene_*/*.mp4')):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(path)]))['streams'][0]
    assert int(info['nb_read_frames'])==expect[path.parent.name],(path,info)
    assert info['r_frame_rate']=='10/1';assert abs(float(info['duration'])-expect[path.parent.name]/10)<.001
    subprocess.run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(path),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
    videos.append({'path':str(path.relative_to(OUT)),**info,'decoded':True})
assert len(videos)==22,len(videos)
js=re.search(r'<script>(.*?)</script>',(OUT/'index.html').read_text(encoding='utf-8'),re.S).group(1)
with tempfile.TemporaryDirectory(prefix='v77-review-') as td:
    tmp=pathlib.Path(td)/'review_script.js';tmp.write_text(js,encoding='utf-8');subprocess.run(['node','--check',str(tmp)],check=True)
result={'local_links_checked':len(p.refs),'missing_links':missing,'videos':videos,'javascript_syntax':'passed','browser_interaction':'not exercised; static links, JS syntax and actual media decoding checked','query_unit_tests':4,'human_verdict':None}
(OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('REVIEW_VALIDATED',len(videos),'videos;',sum(int(v['nb_read_frames']) for v in videos),'decoded frames')

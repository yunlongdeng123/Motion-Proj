"""本地实际文件、视频解码和JS检查，不冒充浏览器播放验证。"""
from pathlib import Path
from html.parser import HTMLParser
import json,subprocess,shutil,re,argparse


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links += [v for k,v in attrs if k in ['src','href'] and v and not v.startswith(('https:','http:','#','data:'))]


def main(root,cases,videos):
    page=(root/'index.html').read_text(encoding='utf-8');m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    links=Links();links.feed(page);missing=[v for v in links.links if not (root/v.split('#')[0]).exists()];assert not missing,missing
    assert len(m['cases'])==cases and all(c['human_verdict'] is None for c in m['cases'])
    vs=[v for c in m['cases'] for v in c['videos']];assert len(vs)==len(set(vs))==videos
    out=[]
    for v in vs:
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(root/v)]))['streams'][0]
        assert int(info['nb_read_frames'])==30 and info['width']==1024 and info['height']==576
        out.append({'file':v,**info})
    node=shutil.which('node') or 'C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
    for i,script in enumerate(re.findall(r'<script>(.*?)</script>',page,re.S)):
        js=root/f'inline_{i}.js';js.write_text(script,encoding='utf-8');subprocess.run([node,'--check',str(js)],check=True)
    result={'html_cases':cases,'video_count':videos,'decoded_frames':sum(int(v['nb_read_frames']) for v in out),'missing_links':missing,
            'inline_JS_syntax':'passed','human_scores_unset':True,'browser_playback_verified':False,'videos':out}
    (root/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print({k:v for k,v in result.items() if k!='videos'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--cases',type=int,required=True);p.add_argument('--videos',type=int,required=True)
    a=p.parse_args();main(a.root,a.cases,a.videos)

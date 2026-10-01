from pathlib import Path
from html.parser import HTMLParser
import json,subprocess,re
BASE=Path('C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia')
ROOT=BASE/'outputs/v77-target-protected-r7'
class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.articles=[]
    def handle_starttag(self,t,attrs):
        a=dict(attrs)
        if t=='article':self.articles.append(a['id'])
        for k in ('src','href'):
            if k in a and not a[k].startswith(('https:','http:','#')):self.refs.append(a[k])
def main():
    manifest=json.loads((ROOT/'delivery_validation.json').read_text(encoding='utf-8'))
    text=(ROOT/'index.html').read_text(encoding='utf-8');p=Links();p.feed(text)
    assert len(p.articles)==6
    for ref in p.refs:assert (ROOT/ref).is_file(),ref
    for v in manifest['videos']:
        f=ROOT/v['path'];assert f.stat().st_size==v['bytes'],f
    assert len(list(ROOT.rglob('*.mp4')))==manifest['actual_decoded_videos']
    js='\n'.join(re.findall(r'<script>(.*?)</script>',text,re.S))
    out=BASE/'work/v77_target_protected/iteration7/index_script.js';out.write_text(js,encoding='utf-8')
    node=Path('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
    subprocess.run([str(node),'--check',str(out)],check=True)
    result={'cases':len(p.articles),'links':len(p.refs),'all_local_links_exist':True,'all_video_sizes_match_remote_decoded_manifest':True,
            'videos':manifest['actual_decoded_videos'],'remote_decoded_frames':manifest['actual_decoded_frames'],'javascript_syntax':True,'browser_playback_verified':False}
    (ROOT/'local_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

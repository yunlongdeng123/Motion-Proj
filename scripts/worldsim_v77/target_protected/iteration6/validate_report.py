"""交付HTML链接及所有H264完整解码，空视频/帧数不一致失败。"""
from pathlib import Path
from html.parser import HTMLParser
import argparse,subprocess,json
import imageio_ffmpeg

class Links(HTMLParser):
    def __init__(self):super().__init__();self.items=[];self.videos=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        for key in ('src','href'):
            if key in a and not a[key].startswith(('http:','https:','#')):self.items.append(a[key])
        if tag=='video':self.videos.append(a['src'])

def main(root):
    delivery=root/'delivery';allvideos=set();pages={}
    for page in ('index.html','data_review.html'):
        l=Links();l.feed((delivery/page).read_text())
        assert all((delivery/p).is_file() for p in l.items),[p for p in l.items if not (delivery/p).is_file()]
        allvideos.update(l.videos);pages[page]={'links':len(l.items),'videos':len(l.videos)}
    allvideos.update(str(p.relative_to(delivery)) for p in (delivery/'eval_assets').rglob('*.mp4'))
    catalog=json.loads((delivery/'dataset_catalog.json').read_text());expected={}
    for c in catalog['cases']:
        cid=c['dataset_id'].replace('/','_')
        for role in ('Y','X','condition_preview'):expected[f'data_assets/{cid}/{role}.mp4']=c['frame_count']
    for p in allvideos:
        if p.startswith('eval_assets/'):expected[p]=10
    assert allvideos==set(expected)
    counts=[]
    plan=json.loads((root/'evaluation_plan.json').read_text())
    for path in sorted(allvideos):
        r=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(delivery/path),'-progress','pipe:1','-f','null','-'],capture_output=True,text=True)
        if r.returncode or r.stderr.strip():raise ValueError((path,r.stderr))
        frame=max(int(s.split('=')[1]) for s in r.stdout.splitlines() if s.startswith('frame='))
        assert frame==expected[path],(path,frame,expected[path]);counts.append({'path':path,'decoded_frames':frame,'bytes':(delivery/path).stat().st_size})
    result={'pages':pages,'case_count':len(catalog['cases']),'evaluation_case_count':8,'all_links_exist':True,'actual_decoded_videos':len(counts),'actual_decoded_frames':sum(r['decoded_frames'] for r in counts),'videos':counts,'browser_playback_verified':False,'classification':'static links + complete ffmpeg decode; no browser access claimed'}
    result.pop('evaluation_case_count');result.update(evaluation_task_count=8,evaluation_window_count=len(plan['cases']),empty_window_retained=sum(c.get('input_not_exercised',False) for c in plan['cases']))
    (root/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='videos'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)

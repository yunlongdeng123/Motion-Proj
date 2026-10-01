from pathlib import Path
from html.parser import HTMLParser
import json,subprocess
import imageio_ffmpeg
ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r7')
class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.articles=[]
    def handle_starttag(self,t,attrs):
        a=dict(attrs)
        if t=='article':self.articles.append(a['id'])
        for k in ('src','href'):
            if k in a and not a[k].startswith(('http:','https:','#')):self.refs.append(a[k])
def main():
    d=ROOT/'delivery';p=Links();p.feed((d/'index.html').read_text())
    external={'../v77-target-protected-r6/index.html'}
    for v in p.refs:
        if v in external:continue
        assert (d/v).is_file(),v
    rows=[]
    for f in sorted(d.rglob('*.mp4')):
        result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(f),'-progress','pipe:1','-f','null','-'],capture_output=True,text=True)
        assert result.returncode==0 and not result.stderr.strip(),(f,result.stderr)
        count=max(int(s.split('=')[1]) for s in result.stdout.splitlines() if s.startswith('frame='));assert count==10
        rows.append({'path':str(f.relative_to(d)),'decoded_frames':count,'bytes':f.stat().st_size})
    plan=json.loads((ROOT/'evaluation_plan.json').read_text());assert set(p.articles)=={c['eval_id'] for c in plan['cases']}
    r6=ROOT.parent/'r6/training/steps.jsonl';fixed=ROOT/'encoder_fixed_lowres/training/steps.jsonl'
    parse=lambda file:[(s['last_case'],s['last_window']) for s in map(json.loads,file.read_text().splitlines())]
    assert parse(r6)==parse(fixed) and len(parse(fixed))==160
    result={'videos':rows,'actual_decoded_videos':len(rows),'actual_decoded_frames':sum(r['decoded_frames'] for r in rows),'links':len(p.refs),
            'cases':len(p.articles),'training_case_and_window_order_identical_to_r6':True,'all_local_links_exist':True,
            'external_sibling_r6_report':True,'browser_playback_verified':False}
    (ROOT/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='videos'},ensure_ascii=False))
if __name__=='__main__':main()

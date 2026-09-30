"""静态核验本地CPU报告；不宣称浏览器播放或交互已验证。"""
import argparse,json
from html.parser import HTMLParser
from pathlib import Path
from PIL import Image

class Resources(HTMLParser):
    def __init__(self):
        super().__init__();self.paths=[];self.svg=0
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='svg':self.svg+=1
        for name in ['src','href']:
            if name in attrs:self.paths.append(attrs[name])

def main(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    page=(root/'index.html').read_text(encoding='utf-8');assert '\ufffd' not in page
    p=Resources();p.feed(page);assert p.svg==1
    missing=[s for s in p.paths if not s.startswith(('#','http:','https:')) and not (root/s.split('#')[0]).exists()]
    assert not missing,missing
    manifest=read('preflight_manifest.json');qa=read('independent_preflight_reviews.json');queue=read('gpu_queue.json');summary=read('summary.json')
    expected={'C010','C012','C014','C018'}
    assert {x['case_id'] for x in manifest['cases']}=={x['case_id'] for x in qa['cases']}==expected
    decoded=[]
    for c in manifest['cases']:
        assert c['frame']==5 and c['human_verdict'] is None and c['training_ready'] is False
        assert c['status'].startswith('source_preflight_')
        with Image.open(root/c['image']) as im:
            im.load();assert im.size==(1600,944);decoded.append(c['image'])
    for c in qa['cases']:
        assert c['reviewed_frame']==5 and c['human_verdict'] is None and c['training_ready'] is False
        assert c['temporal_visual_scope']=='not_reviewed'
    assert len(queue['jobs'])==4 and queue['enabled'] is False and queue['executed']==0
    assert summary['complete'] and summary['receiver_windows']==29 and summary['planned_cases']==4
    assert summary['GPU_model_calls']==summary['actual_synthetic_cases']==summary['training_ready']==0
    result={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r4','local_report':str(root/'index.html'),
            'report_kind':'CPU_source_geometry_preflight_not_synthetic_or_model_output',
            'case_count':4,'decoded_images':decoded,'local_links_checked':len(p.paths),'missing_links':missing,
            'architecture_svg':True,'human_verdicts_all_null':True,'GPU_queue_disabled':True,
            'browser_interaction':'not_tested','video_playback':'not_applicable_no_new_videos'}
    (root/'local_delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)

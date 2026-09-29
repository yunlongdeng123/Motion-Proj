"""本地交付完整性、JPEG实解码、HTML脚本提取；不冒称浏览器播放。"""
import argparse,json,re
from pathlib import Path
from PIL import Image

def main(out):
    data=json.loads((out/'review_manifest.json').read_text(encoding='utf-8'));ids=set();videos=images=0;human_frames=0
    for c in data['clips']:
        assert c['case_id'] not in ids;ids.add(c['case_id'])
        n=len(c['preview_frames']);assert n in [10,30] and c['frame_count']==n
        assert c['human_verdict'] is None and c['training_ready'] is False
        if c['review']['synthetic_status']=='pass':human_frames+=n
        assert set(c['review']['reviewed_frames'])==set(c['review_frames'])
        for path in c['contacts']:
            with Image.open(out/path) as im:im.load()
        for path in c['videos'].values():
            p=out/path;assert p.is_file() and p.stat().st_size>1000
            with p.open('rb') as f:assert b'ftyp' in f.read(32)
            videos+=1
        for i,f in enumerate(c['preview_frames']):
            assert f['frame']==i
            if i:assert f['timestamp_us']>c['preview_frames'][i-1]['timestamp_us']
            for key in ['gt','input','labels','condition']:
                with Image.open(out/f[key]) as im:im.load();assert im.size==(1024,576)
                images+=1
    assert videos==196 and images==3240 and len(ids)==49
    html=(out/'index.html').read_text(encoding='utf-8');scripts=re.findall(r'<script>(.*?)</script>',html,re.S);assert len(scripts)==1
    (out/'review_script.js').write_text(scripts[0],encoding='utf-8',newline='\n')
    result={'cases':len(ids),'videos_present_with_MP4_header':videos,'preview_JPEGs_actually_decoded':images,'human_full_review_frames':human_frames,'all_human_verdict_null':True,'actual_video_decode':'remote delivery_validation.json; local videos checked presence/container header only','browser_interaction_verified':False}
    (out/'local_delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)

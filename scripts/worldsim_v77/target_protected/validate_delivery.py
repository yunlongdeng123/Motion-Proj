"""核验来源页真实图片、逐帧链接与JS语法；不宣称浏览器播放通过。"""
import argparse
import json
from pathlib import Path
import re
import subprocess
from PIL import Image


def main(out,node):
    data=json.loads((out/'source_review_manifest.json').read_text(encoding='utf-8'))
    assert data['stage']=='source_only_cpu'
    ids=set();assets=set();frames=0
    for c in data['clips']:
        assert c['source_id'] not in ids;ids.add(c['source_id'])
        assert c['subagent_review'] is not None
        assert c['human_verdict'] is None and c['synthetic_quality']=='not_created'
        assert [f['frame'] for f in c['frames']]==list(range(30))
        timestamps=[f['timestamp_us'] for f in c['frames']]
        assert all(0<b-a<=180000 for a,b in zip(timestamps,timestamps[1:]))
        for f in c['frames']:
            frames+=1
            for k in data['columns']:assets.add(f[k])
        assets.add(c['contact'])
    for rel in sorted(assets):
        p=(out/rel).resolve();assert p.is_relative_to(out.resolve())
        with Image.open(p) as im:im.load();assert im.width>0 and im.height>0
    html=(out/'index.html').read_text(encoding='utf-8')
    scripts=re.findall(r'<script>(.*?)</script>',html,re.S)
    assert len(scripts)==1 and '__DATA__' not in scripts[0]
    js=out/'source-review.syntax-check.js';js.write_text(scripts[0],encoding='utf-8')
    subprocess.run([str(node),'--check',str(js)],check=True)
    result={'task_id':data['task_id'],'run_id':data['run_id'],'stage':'source_only_cpu',
            'clips':len(ids),'source_frames':frames,'actual_decoded_preview_images':len(assets),
            'columns':data['columns'],'all_subagent_reviews_present':True,'human_verdict_initially_empty':True,
            'missing_source_ids':data['missing_sources'],'js_syntax':'pass','browser_playback':'not_tested',
            'qualified_synthetic_pairs':0,'model_calls':0}
    (out/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--node',type=Path,required=True)
    a=p.parse_args();main(a.out,a.node)

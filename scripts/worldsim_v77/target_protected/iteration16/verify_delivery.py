"""检查CPU交付：链接、动态图像与JS语法；不加载研究模型。"""
from pathlib import Path
import json,re,subprocess
from PIL import Image

root=Path('outputs/v77-priors-r49').resolve()
page=(root/'index.html').read_text(encoding='utf-8')
links=re.findall(r'(?:src|href)="([^"#]+)"',page)
assert all((root/p).is_file() for p in links),[p for p in links if not (root/p).is_file()]
plan=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
assert len(plan['cases'])==9 and page.count('<article id=')==9
dynamic=0
for c in plan['cases']:
    cid=c['case_id']
    for frame in (0,5,9):
        for kind in ('original','owner','grid','projection'):
            p=root/'assets'/cid/f'{kind}_{frame:02}.jpg'
            with Image.open(p) as im:assert im.size==(1024,576);im.verify()
            dynamic+=1
    for slot in range(6):
        for suffix in ('','_patches'):
            with Image.open(root/'assets'/cid/f'reference_{slot:02}{suffix}.png') as im:
                assert im.size==(256,256);im.verify()
pre=json.loads((root/'preflight.json').read_text(encoding='utf-8'))
assert pre['CPU_ready'] and not pre['GPU_started']
state=json.loads((root/'controller_state.json').read_text(encoding='utf-8'))
assert state['GPU_jobs']==0 and state['training_steps']==0 and state['new_inference_windows']==0
js=re.findall(r'<script>(.*?)</script>',page,re.S)[0]
node='C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
result=subprocess.run([node,'--check','-'],input=js,text=True,capture_output=True)
assert result.returncode==0,result.stderr
out=Path('work/v77_target_protected/iteration16/local_delivery_check.json')
data={'case_cards':9,'links_exist':len(links),'dynamic_full_frame_images_verified':dynamic,
    'reference_images_verified':9*6*2,'inline_JS_syntax':True,'new_GPU_results':0,
    'historical_videos':len({p for p in links if p.endswith('.mp4')}),
    'scope':'CPU交付链接/图片/JS；历史视频沿用r48已完成解码，不作为r49输出',
    'human_verdict':None}
out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(data,ensure_ascii=False))

"""检查审核交付：链接、图片、JS和GPU结果视频实际解码；不加载模型。"""
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
assert state['GPU_jobs']==0 and state['training_steps'] in (0,64)
windows=state['new_inference_windows'];assert windows in (0,9,20)
js=re.findall(r'<script>(.*?)</script>',page,re.S)[0]
node='C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
result=subprocess.run([node,'--check','-'],input=js,text=True,capture_output=True)
assert result.returncode==0,result.stderr
out=Path('work/v77_target_protected/iteration16/local_delivery_check.json')
videos=sorted({p for p in links if p.endswith('.mp4')})
new_videos=[p for p in videos if Path(p).name.startswith(('zero_','step64_'))]
assert len(new_videos)==2*windows
decoded=[]
if windows:
    for p in videos:
        path=root/p
        probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=width,height,nb_frames,avg_frame_rate,duration','-of','json',str(path)],capture_output=True,text=True)
        assert probe.returncode==0,probe.stderr
        stream=json.loads(probe.stdout)['streams'][0]
        assert (stream['width'],stream['height'])==(1024,576),(p,stream)
        assert int(stream['nb_frames'])==10 and abs(float(stream['duration'])-1)<.01,(p,stream)
        decode=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(path),'-map','0:v:0','-f','null','-'],capture_output=True,text=True)
        assert decode.returncode==0,(p,decode.stderr)
        decoded.append(p)
data={'case_cards':9,'links_exist':len(links),'dynamic_full_frame_images_verified':dynamic,
    'reference_images_verified':9*6*2,'inline_JS_syntax':True,'new_GPU_windows':windows,
    'new_GPU_videos':len(new_videos),'historical_videos':len(videos)-len(new_videos),
    'actual_videos_decoded':len(decoded),'video_dimensions':[1024,576] if decoded else None,
    'scope':'审核链接/图片/JS；GPU结果存在时实际解码全部视频，不以历史视频冒充本轮输出',
    'human_verdict':None}
out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(data,ensure_ascii=False))

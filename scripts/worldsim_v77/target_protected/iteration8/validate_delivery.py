"""本地资源与真正视频解码检查；不冒称已测浏览器同步播放。"""
from pathlib import Path
import json,re,subprocess,argparse,shutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def main(root,node,data_only=False):
    root=Path(root);checks={'manifests':{},'errors':[],'browser_playback_tested':False};videos=set();images=set()
    for filename in (['data_manifest.json'] if data_only else ['data_manifest.json','effect_manifest.json']):
        data=json.loads((root/filename).read_text(encoding='utf-8'));checks['manifests'][filename]={'cases':len(data['cases']),'roles':len(data['roles'])}
        for c in data['cases']:
            videos.update(root/p for p in c['videos'].values());videos.update(root/p for p in (c.get('native_links') or {}).values())
            for i in range(10):
                for r in data['roles']:images.add(root/c['frame_pattern'].replace('{i}',f'{i:03}').replace('{role}',r['key']))
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    def decode(path):
        if not path.is_file():return {'path':str(path),'error':'missing'}
        p=subprocess.run([ffmpeg,'-v','error','-threads','1','-i',str(path),'-an','-f','null','-','-progress','pipe:1','-nostats'],capture_output=True,text=True)
        frames=re.findall(r'^frame=(\d+)$',p.stdout,re.M)
        if p.returncode or not frames or int(frames[-1])!=10:return {'path':str(path),'error':p.stderr[-1000:],'decoded_frames':int(frames[-1]) if frames else None,'returncode':p.returncode}
        return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        checks['errors'] += [r for r in pool.map(decode,sorted(videos)) if r]
    for path in sorted(images):
        try:
            with Image.open(path) as im:im.load();assert im.size==(1024,576)
        except Exception as e:checks['errors'].append({'path':str(path),'error':repr(e)})
    for page in (['data_review.html'] if data_only else ['index.html','data_review.html']):
        html=(root/page).read_text(encoding='utf-8');assert '__DATA__' not in html and '__INTRO__' not in html
        script=re.search(r'<script>(.*?)</script>',html,re.S).group(1);target=root/(page+'.check.js');target.write_text(script,encoding='utf-8')
        p=subprocess.run([node,'--check',str(target)],capture_output=True,text=True)
        if p.returncode:checks['errors'].append({'page':page,'error':p.stderr})
    checks.update(videos_checked=len(videos),decoded_video_frames=len(videos)*10,len_images_checked=len(images),static_HTML_and_JS_checked=True,success=not checks['errors'])
    (root/('data_delivery_validation.json' if data_only else 'delivery_validation.json')).write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(checks)
    assert checks['success']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--node',required=True);p.add_argument('--data-only',action='store_true');a=p.parse_args();main(a.root,a.node,a.data_only)

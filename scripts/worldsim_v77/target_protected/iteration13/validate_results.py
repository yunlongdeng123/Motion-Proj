"""实际解码所有交付视频；检验链接、帧数与人工字段，不用清单冒充解码。"""
from common import *
import re
import cv2
from PIL import Image


def main():
    root=O/'review';page=(root/'index.html').read_text()
    links=re.findall(r'(?:src|href)="([^"]+)"',page)
    paths=[root/p for p in links if not p.startswith(('#','http'))]
    assert all(p.is_file() for p in paths)
    videos=sorted(set(p for p in paths if p.suffix=='.mp4'))
    assert len(videos)==96, len(videos)
    rows=[]
    for p in videos:
        cap=cv2.VideoCapture(str(p));n=0;shapes=set()
        while True:
            ok,frame=cap.read()
            if not ok:break
            n+=1;shapes.add(frame.shape)
        fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        assert n==10 and shapes=={(576,1024,3)} and abs(fps-10)<1e-4,(p,n,shapes,fps)
        rows.append({'video':str(p.relative_to(root)),'decoded_frames':n,'fps':fps})
    for p in set(paths):
        if p.suffix=='.jpg':
            with Image.open(p) as im:im.load()
    metrics=read(O/'results_summary.json')
    assert len(metrics['cases'])==12 and all(c['human_verdict'] is None for c in metrics['cases'])
    for c in metrics['cases']:
        assert all(r['human_verdict'] is None and r['temporal_verdict'] is None for r in c['metrics'].values())
    result={'stage':'delivery_decoded','cases':12,'windows':36,'video_links':len(videos),
            'decoded_frames':sum(r['decoded_frames'] for r in rows),'all_links_exist':True,
            'human_verdict_unfilled':True,'browser_playback_not_claimed':True,'videos':rows}
    dump(O/'gpu_delivery_validation.json',result);dump(root/'gpu_delivery_validation.json',result)
    print({k:v for k,v in result.items() if k!='videos'})


if __name__=='__main__':main()

"""将真实视频元数据嵌入离线HTML，避免file://下fetch限制。"""
import argparse,json,pathlib
from PIL import Image,ImageDraw

p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--template',required=True);a=p.parse_args()
root=pathlib.Path(a.run_dir);out=root/'review';data=json.loads((out/'review_data.json').read_text());small=dict(data,scenes=[])
for s in data['scenes']:
 compact=dict(s,frames=[{k:r[k] for k in ['frame','gt_present','target_point_count','source_boxes','target_boxes','depth_scale']} for r in s['frames']]);small['scenes'].append(compact)
 im=Image.open(root/s['name']/'frames/020/original.png');c=s['default_camera'];x=(c%3)*688;y=(c//3)*384;im=im.crop((x,y,x+688,y+384))
 b=s['frames'][20]['source_boxes'][c]
 if b:ImageDraw.Draw(im).rectangle(b,outline='#ffbc57',width=4)
 im.save(out/s['name']/'target.jpg',quality=94)
raw=pathlib.Path(a.template).read_text();assert raw.count('__REVIEW_DATA__')==1
(out/'index.html').write_text(raw.replace('__REVIEW_DATA__',json.dumps(small,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')))
print(out/'index.html')

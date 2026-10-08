"""只打包网页实际使用的轻量资产；不重复搬RGB和中间帧目录。"""
from common import *
import argparse, tarfile

p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--keyframes-only',action='store_true');a=p.parse_args()
root=O/'review'
files=(sorted(root.glob('R*/structure_keyframe.jpg')) if a.keyframes_only else
       sorted(x for x in root.rglob('*') if x.is_file() and len(x.relative_to(root).parts)<=2 and
              x.suffix.lower() in {'.mp4','.jpg','.html','.json'}))
assert files
with tarfile.open(O/a.name,'w:gz') as archive:
    for x in files:archive.add(x,arcname=str(x.relative_to(root)))
print('PACKAGED',a.name,len(files),sum(x.stat().st_size for x in files),flush=True)

"""只链接已完整落盘的真实成员，允许CPU几何检查与长archive解包衔接。"""
import argparse,json
from pathlib import Path
from PIL import Image

def main(root):
    f=root/'native10_factory';names=json.loads((f/'all_required_files.json').read_text());linked=0
    for n in names:
        dst=f/'rgb'/n
        if dst.is_file():continue
        src=next((f/'by_shard'/s/n for s in ['03','07'] if (f/'by_shard'/s/n).is_file()),None)
        if src is None:continue
        try:
            if n.endswith('.jpg'):
                with Image.open(src) as im:im.load();assert im.size==(1600,900)
            else:assert src.stat().st_size>0 and src.stat().st_size%20==0
        except (OSError,AssertionError):continue
        dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src.resolve());linked+=1
    if not (f/'maps').exists():(f/'maps').symlink_to(root.parent/'r1/maps',target_is_directory=True)
    print('MATERIALIZED',linked,'total',sum((f/'rgb'/n).is_file() for n in names),'required',len(names),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)

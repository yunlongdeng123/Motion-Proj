"""只提取有限已选多视角RGB；复用已有文件，不写公共盘。"""
import argparse,json,sys,tarfile,time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_source_pool import sharding
from PIL import Image,ImageDraw

def read(p):return json.loads(Path(p).read_text())
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main(root,factory,pub):
    dest=root/'multiview';data=read(dest/'full_track_catalog.json');selected=data['selected'];rgb=dest/'rgb';rgb.mkdir(exist_ok=True)
    for r in selected:
        p=rgb/r['filename'];src=factory/'rgb'/r['filename']
        if src.is_file() and not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(src.resolve())
    missing={r['filename'] for r in selected if not (rgb/r['filename']).is_file()};groups=defaultdict(set);mapping=sharding()
    for n in missing:
        for shard in mapping[Path(n).name.split('__')[0]]:groups[shard].add(n)
    state={'state':'extracting','selected':len(selected),'shards':[]};dump(dest/'extraction_state.json',state)
    for shard,names in sorted(groups.items()):
        pending={n for n in names if not (rgb/n).is_file()};start=time.monotonic()
        if pending:
            archive=pub/f'v1.0-trainval{int(shard):02}_blobs.tgz'
            with tarfile.open(archive,'r|gz') as tf:
                for member in tf:
                    n=member.name.removeprefix('./')
                    if n not in pending:continue
                    assert member.isfile() and not Path(n).is_absolute() and '..' not in Path(n).parts
                    p=rgb/n;p.parent.mkdir(parents=True,exist_ok=True)
                    with tf.extractfile(member) as src,p.open('wb') as out:out.write(src.read())
                    with Image.open(p) as im:im.verify()
                    pending.remove(n)
                    if not pending:break
            state['shards'].append({'shard':shard,'requested':len(names),'missing':sorted(pending),'seconds':time.monotonic()-start});dump(dest/'extraction_state.json',state)
    sheet=Image.new('RGB',(1200,((len(selected)+2)//3)*300),(18,25,35));draw=ImageDraw.Draw(sheet);frames=[]
    for i,r in enumerate(selected):
        path=rgb/r['filename'];r['materialized']=path.is_file();r['view_id']=f'MV{i+1:02}'
        if not path.is_file():continue
        im=Image.open(path).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS)
        x0,y0,x1,y1=r['box_xyxy'];b=[max(0,int(x0)-15),max(0,int(y0)-15),min(1024,int(x1)+16),min(576,int(y1)+16)]
        crop=im.crop(b);crop.thumbnail((390,250));x=(i%3)*400;y=(i//3)*300;sheet.paste(crop,(x,y+43));draw.text((x+5,y+5),f"{r['view_id']} {','.join(r['source_ids'])} {r['camera']}",fill='white');draw.text((x+5,y+23),f"yaw {r['view_angles_deg'][0]:.1f} / {r['timestamp_us']}",fill='white')
        cp=dest/'crops'/f"{r['view_id']}.jpg";im.crop(b).save(cp,quality=96);r['crop']=f"crops/{r['view_id']}.jpg";frames.append(r)
    sheet.save(dest/'selected_contacts.jpg',quality=96)
    state.update(state='complete' if len(frames)==len(selected) else 'missing_rgb',materialized=len(frames));dump(dest/'extraction_state.json',state);dump(dest/'selected_views.json',{'views':selected,'state':state})
    print(json.dumps(state,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--factory',type=Path,required=True);p.add_argument('--pub',type=Path,required=True);a=p.parse_args();main(a.root,a.factory,a.pub)

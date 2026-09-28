"""补充非主视角GT邻车包络审计，读现成数据，不改生成。"""
from nine_common import *
from geometry import transform
from PIL import Image
import argparse,numpy as np
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--camera',type=int,required=True);a=p.parse_args();s=next(s for s in read(ROOT/'registration.json')['scenes'] if s['name']==a.scene);base=ROOT/a.scene;rows=[]
for f,fr in enumerate(read(base/'camera_frames.json')):
    c,k=camera(fr,a.camera);m=np.array(Image.open(base/f'cam{a.camera}/write_mask'/f'{f:05}.png'))>0;t=next(b for b in fr['all_boxes'] if b['actor_id']==s['actor']);tz=transform(np.array(t['pose'])[None,:3,3],np.linalg.inv(c))[0,2];ne=[]
    for b in fr['all_boxes']:
        if b['actor_id']==s['actor'] or not b.get('category','').startswith('vehicle.'):continue
        h=hull_mask(b,c,k,ground_extend=.4);n=int((m&h).sum())
        if n:
            z=transform(np.array(b['pose'])[None,:3,3],np.linalg.inv(c))[0,2];ne.append(dict(actor=b['actor_id'],pixels=n,write_fraction=n/max(1,int(m.sum())),center_behind=bool(z>tz)))
    rows.append(dict(frame=f,write_pixels=int(m.sum()),neighbors=ne))
path=ROOT/f'neighbor_probe_{a.scene}_cam{a.camera}.json';dump(path,dict(scene=a.scene,camera=a.camera,rows=rows,scope='GT volume overlap, not pixel visibility; absence is limited by GT recall',human_verdict=None));print(path,flush=True)
for r in rows:
    if r['frame'] in [0,9,14,18,27,29]:print(r,flush=True)

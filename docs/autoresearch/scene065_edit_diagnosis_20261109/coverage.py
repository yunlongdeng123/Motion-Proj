from pathlib import Path
import json,csv
import numpy as np
from PIL import Image,ImageDraw
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUN=CAM/'waymo_val065_delete30k_20261109_r1'
OUT=CAM/'waymo_val065_delete30k_diagnosis_20261109_r1'
OUT.mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text())
m=load(RUN/'final_manifest.json'); ref=np.load(RUN/'world_road/reference.npz')
rows=[]
for f in m['frames']:
    raw=f['key'];v=np.load(RUN/'world_road/views'/f'{raw:03d}.npz')
    mask=np.asarray(Image.open(f['instance_mask']).resize((480,320),Image.Resampling.NEAREST))==10
    idx=v['index']; inside=v['inside'];fit=v['fit']>.05
    r=dict(raw=raw,target_pixels=int(mask.sum()),inside=int((mask&inside).sum()),
           fit=int((mask&fit).sum()),keep=int((mask&(v['keep']>0)).sum()),
           supported=int((mask&v['real_domain']).sum()),inferred=int((mask&v['inferred_domain']).sum()),
           outside=int((mask&~inside).sum()),inside_unknown=int((mask&inside&~ref['active'][idx]&~ref['marked'][idx]).sum()),
           inside_marking=int((mask&inside&ref['marked'][idx]).sum()),
           active_but_not_fit=int((mask&inside&ref['active'][idx]&~fit).sum()),
           keep_and_fit=int((mask&fit&(v['keep']>0)).sum()))
    rows.append(r)
    if raw in [60,119,150,198]:
        rgb=np.asarray(Image.open(RUN/'delete_white'/f'{raw:03d}.png')) if (RUN/'delete_white'/f'{raw:03d}.png').exists() else None
        prov=load(RUN/'delete_white/provenance.json');entry=next(x for x in prov['frames'] if x['frame_key']==raw)
        image=Image.open(RUN/'delete_white'/entry['files']['rgb_png']).convert('RGB') if 'rgb_png' in entry['files'] else None
        a=np.zeros((320,480,3),np.uint8)
        a[mask&~inside]=[239,101,71]
        a[mask&inside&~ref['active'][idx]]=[236,185,61]
        a[mask&v['real_domain']]=[68,204,153]
        a[mask&v['inferred_domain']]=[80,170,240]
        Image.fromarray(a).save(OUT/f'coverage{raw}.png')
        np.savez_compressed(OUT/f'coverage{raw}.npz',target=mask,inside=inside,fit=fit,keep=v['keep'],real=v['real_domain'],inferred=v['inferred_domain'])
total={k:sum(r[k] for r in rows) for k in rows[0] if k!='raw'}
out=dict(task='WAYMO-065-V43-V44-DIAGNOSIS-20261109',rows=rows,total=total,road_receipt=load(RUN/'world_road/receipt.json'),training_updates=0,mask_role='existing training input, not independent pixel ownership ground truth')
(OUT/'coverage.json').write_text(json.dumps(out,indent=2)+'\n')
with (OUT/'coverage.csv').open('w') as s:
    w=csv.DictWriter(s,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(json.dumps(dict(total=total,keys=[r for r in rows if r['raw'] in [60,119,150,198]]),indent=2))

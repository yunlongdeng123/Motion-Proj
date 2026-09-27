"""r29：冻结r28生成图，单独检验梯度域写回是否缓解亮边。"""
from pathlib import Path
import json,time,datetime
import cv2,numpy as np
from PIL import Image,ImageDraw
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');ROOT=BASE/'r29';SRC=BASE/'r28'
dump=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');cv2.setNumThreads(4)
assert not ROOT.exists();ROOT.mkdir();beg=time.time()
dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r29',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',actor='12',frames=list(range(10)),camera=0,
 question='Once r28 removes the gray target silhouette, does gradient-domain writeback improve its color seam without changing the generator or edit extent?',
 source='Exact r28 native generation and original RGB; same precise write mask. No original hidden target pixels are copied inside the mask as a factual background.',
 intervention='One OpenCV NORMAL_CLONE Poisson solve at the exact source-mask bounding center, then force all outside-write pixels back to original. Preserve native geometry/gradient as guidance; destination boundary gives observed colors.',
 relation_to_r9='Prior0255 Poisson only softened color and could not repair wrong geometry. Here test only the observed third-scene r28 brightness boundary, not a remedy for generated identity/structure.',
 stop_rule='One fixed solve per10frames; reject if it leaves dark car-shaped stain, destroys lane markings, or alters retained truck. No mixed-mode/parameter grid.',resources=dict(gpu_forwards=0,cpu_threads=4),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
(ROOT/'final').mkdir();contacts=[Image.new('RGB',(1200,1250),(12,20,30)) for _ in range(2)];rows=[]
for f in range(10):
 original=np.array(Image.open(SRC/'input'/f'{f:05}.png'));native=np.array(Image.open(SRC/'native'/f'{f:05}.png'));old=np.array(Image.open(SRC/'final'/f'{f:05}.png'));m=np.array(Image.open(SRC/'write_mask'/f'{f:05}.png'))>0;y,x=np.where(m);xx,yy,w,h=cv2.boundingRect(m.astype('uint8'));center=(xx+w//2,yy+h//2)
 clone=cv2.seamlessClone(cv2.cvtColor(native,cv2.COLOR_RGB2BGR),cv2.cvtColor(original,cv2.COLOR_RGB2BGR),np.uint8(m)*255,center,cv2.NORMAL_CLONE);clone=cv2.cvtColor(clone,cv2.COLOR_BGR2RGB);clone[~m]=original[~m];assert np.array_equal(clone[~m],original[~m]);Image.fromarray(clone).save(ROOT/'final'/f'{f:05}.png')
 left=int(np.clip((x.min()+x.max())/2-160,0,704));top=int(np.clip((y.min()+y.max())/2-90,0,396));box=(left,top,left+320,top+180)
 for j,(name,im) in enumerate([('original',original),('r28 hard write',old),('r29 Poisson',clone)]):
  tile=Image.new('RGB',(400,250),(12,20,30));tile.paste(Image.fromarray(im).crop(box).resize((400,225)),(0,25));ImageDraw.Draw(tile).text((5,5),f'f{f} {name}',fill='white');contacts[f//5].paste(tile,(j*400,(f%5)*250))
 rows.append(dict(frame=f,changed_pixels_inside=int(np.any(clone!=old,-1)[m].sum()),write_pixels=int(m.sum()),outside_changed=0,center=list(center)))
for i,im in enumerate(contacts):im.save(ROOT/f'all10_part{i+1}.jpg',quality=96)
dump(ROOT/'state.json',dict(state='complete_pending_visual_review',seconds=time.time()-beg,frames=10,checks=rows,human_verdict=None,background_input_dir=None));print('R29_COMPLETE',time.time()-beg,flush=True)

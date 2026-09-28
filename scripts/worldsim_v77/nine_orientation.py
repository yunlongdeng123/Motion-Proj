"""共同的实际参考0/180方向判别；不是按编辑结果手调。"""
from pathlib import Path
import argparse,subprocess,json,copy,shutil,numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scene',required=True);a=p.parse_args();S=Path(__file__).resolve().parent;R=a.root;D=R/a.scene;B='D:/software/Blender/blender.exe'
if (D/'orientation.json').exists():raise RuntimeError('已有方向判别，不能不留证覆盖')
base=json.loads((R/'registration.json').read_text());spec=next(s for s in base['scenes'] if s['name']==a.scene);src=json.loads((D/'asset/registration.json').read_text())['source'];f=src['frame'];c=src['camera']
reference=np.array(Image.open(D/'asset/source_full.png').convert('RGB')).astype('float32')/255;truth=np.array(Image.open(D/'asset/source_core.png'))>0
rows=[];pics=[Image.fromarray(np.uint8(reference*255))]
for yaw in [0,180]:
    reg=copy.deepcopy(base);one=copy.deepcopy(spec);one['yaw']=yaw;one['streams']=[dict(camera=c,active_frames=[f])];reg['scenes']=[one]
    rn=f'orientation_{a.scene}_{yaw}.json';(R/rn).write_text(json.dumps(reg,indent=2));out=f'orientation_{yaw}'
    with (D/f'{out}.log').open('w') as log:subprocess.run([B,'--background','--python',str(S/'nine_blender.py'),'--','--root',str(R),'--scene',a.scene,'--registration',rn,'--output-folder',out],stdout=log,stderr=subprocess.STDOUT,check=True)
    rgba=np.array(Image.open(D/out/f'f{f:03}_cam{c}.png').convert('RGBA'));mask=rgba[...,3]>25;both=mask&truth;union=mask|truth;iou=float(both.sum()/max(1,union.sum()));mae=float(np.abs(rgba[...,:3].astype('float32')/255-reference)[both].mean()) if both.any() else 1.
    rows.append(dict(yaw_deg=yaw,mask_iou=iou,foreground_rgb_mae=mae,score=iou-.25*mae));alpha=rgba[...,3:]/255;comp=np.rint(reference*255*(1-alpha)+rgba[...,:3]*alpha).astype('uint8');pics.append(Image.fromarray(comp))
diff=abs(rows[0]['score']-rows[1]['score']);selected=0 if diff<.01 else max(rows,key=lambda r:r['score'])['yaw_deg']
out=dict(scene=a.scene,source=src,rule='horizontal PCA then two front/rear candidates on same real source; score=maskIoU-0.25*RGB_MAE[0,1]; scoregap<.01 chooses0 and flags ambiguity',selected_yaw_deg=selected,score_gap=diff,ambiguous=diff<.01,rows=rows,scope='BUILD reference alignment, not heldout appearance validation; occlusion/texture quality may mislead score',human_verdict=None)
(D/'orientation.json').write_text(json.dumps(out,indent=2));shutil.copy2(D/'actor.glb',D/'actor_axis_aligned.glb')
with (D/'bake_yaw.log').open('w') as log:subprocess.run([B,'--background','--python',str(S/'nine_bake_yaw.py'),'--',str(D)],stdout=log,stderr=subprocess.STDOUT,check=True)
sheet=Image.new('RGB',(1376,818),(20,30,40));draw=ImageDraw.Draw(sheet)
for i,pic in enumerate(pics):
    x=(i%2)*688;y=(i//2)*409;sheet.paste(pic,(x,y+25));draw.text((x+4,y+4),['real source','yaw0','yaw180'][i],fill='white')
draw.text((692,439),f'selected yaw{selected} | ambiguous={diff<.01}',fill='white');sheet.save(D/'orientation_contact.jpg',quality=94)
print('ORIENTATION',a.scene,selected,rows,flush=True)

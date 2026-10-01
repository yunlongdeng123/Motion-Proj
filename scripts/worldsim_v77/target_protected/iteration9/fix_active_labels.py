"""只修审核标注：活跃受保护实例先标B/C，未参与遮挡来源不冒充B。"""
from pathlib import Path
import sys,shutil
sys.path.insert(0,str(Path(__file__).parent));from temporal_factory import O,read,dump
from render_pairs import label,encode,font
import numpy as np
from PIL import Image,ImageDraw
def main():
 root=O/'data_review';m=read(root/'synthetic_manifest.json');backup=O/'labels_before_active_identity';backup.mkdir(exist_ok=True)
 for c in m['clips']:
  cid=c['case_id'];dest=Path(c['folder']);preview=root/'assets'/cid;panels=[];ids=c['protected_instances']
  for i in range(10):
   load=lambda r:np.asarray(Image.open(dest/r/f'{i:03}.png'))
   x=load('X');y=load('Y');cp=load('condition_preview');aa=load('alpha')>0
   pm={t:np.asarray(Image.open(dest/'protected'/f'{i:03}_{t}.png'))>0 for t in ids}
   orig=preview/f'{i:03}_labels.jpg';saved=backup/cid/orig.name;saved.parent.mkdir(exist_ok=True)
   if not saved.exists():shutil.copy2(orig,saved)
   ann=label(x,aa,pm);Image.fromarray(ann).save(orig,quality=94);panels.append({'gt':y,'input':x,'labels':ann,'condition':cp})
  old=root/'contacts'/f'{cid}_process.jpg';saved=backup/old.name
  if not saved.exists():shutil.copy2(old,saved)
  video=backup/cid/'labels.mp4'
  if not video.exists():shutil.copy2(preview/'labels.mp4',video)
  encode(preview,'labels');sheet=Image.new('RGB',(1600,1150),(14,21,31));d=ImageDraw.Draw(sheet)
  d.text((12,8),f'{cid} | {c["scene"]} | {c["process_family"]} | active protected = B/C only',font=font(22),fill='white')
  for n,i in enumerate([0,5,9]):
   d.text((12,45+n*350),f'f{i} | time {(c["frames"][i]["timestamp"]-c["frames"][0]["timestamp"])/1e6:.2f}s',font=font(18),fill='white')
   for j,(r,a) in enumerate(panels[i].items()):d.text((j*400+10,70+n*350),r,font=font(18),fill='white');sheet.paste(Image.fromarray(a).resize((400,225)),(j*400,100+n*350))
  sheet.save(old,quality=95);c['display_identity_contract']='B/C colors refer only to active protected_instances; first arbitrary source is not required to be occluded'
 dump(root/'synthetic_manifest.json',m);dump(O/'label_fix.json',{'changed':'QA label JPG/MP4/contact only','training_Y_X_H_protected_geometry_changed':False,'all_candidates':len(m['clips']),'old_labels_backup':str(backup),'human_verdict':None})
if __name__=='__main__':main()

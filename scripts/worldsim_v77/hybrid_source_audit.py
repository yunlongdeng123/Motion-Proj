"""r10先审计写回输入身份，固定原RGB/GT与mask，不做生成。"""
import sys,datetime,shutil
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from repair_common import camera,hull_mask
from PIL import ImageDraw
R2=ROOT;ROOT=ROOT.parent/'r10';ROOT.mkdir(exist_ok=True)
assert not (ROOT/'registration.json').exists()
dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r10',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),role='source identity and matte-boundary audit, zero generation',scenes=['scene_0230','scene_0255'],inputs=['source RGB','existing masks','GT boxes and calibration'],failure_ledger_refs=['V77-F02'],human_verdict=None))
for s in read(R2/'registration.json')['scenes']:
 name=s['name'];old=R2/name;out=ROOT/name;out.mkdir();frames=read(OLD/name/'frames.json');rows=[]
 sheet=Image.new('RGB',(960*3,536*4))
 for row,i in enumerate([0,7,15,29]):
  fr=frames[str(s['source_frames'][i])];c,k=camera(fr,s['camera'],HW);orig=rgb(old/'rgb'/f'{i:05}.png');targ=mask(OLD/name/'sam'/f'core_{i:05}.png');dyn=mask(old/'dynamic'/f'{i:05}.png');static=mask(old/'static'/f'{i:05}.png')
  y,x=np.where(targ);region=np.zeros(HW,bool);region[max(0,y.min()-30):min(H,y.max()+31),max(0,x.min()-70):min(W,x.max()+71)]=True
  ims=[Image.fromarray(orig),Image.fromarray(orig),Image.fromarray(orig)];d=ImageDraw.Draw(ims[0]);visible=[]
  for b in fr['all_boxes']:
   if not b['category'].startswith('vehicle.'):continue
   rect=project_bbox(b['pose'],b['size_lwh'],c,k,HW)
   if rect is None:continue
   h=hull_mask(b,c,k,HW);overlap=int((h&region).sum())
   if overlap<40:continue
   depth=float(transform(np.array(b['pose'])[None,:3,3],np.linalg.inv(c))[0,2]);color='yellow' if b['actor_id']==s['actor'] else 'magenta'
   d.rectangle(rect,outline=color,width=2);d.text((rect[0],max(0,rect[1]-12)),f'{b["actor_id"]}: {depth:.1f}m',fill=color)
   visible.append(dict(actor=b['actor_id'],depth=depth,box=rect,overlap_target_sam=int((h&targ).sum()),overlap_dynamic=int((h&dyn).sum())))
  for j,m in enumerate([dyn,static],1):
   overlay=orig.copy();overlay[m]=(.3*overlay[m]+.7*np.array([0,230,230] if j==1 else [255,110,40])).astype('uint8');ims[j]=Image.fromarray(overlay)
  for j,im in enumerate(ims):
   ImageDraw.Draw(im).text((8,8),f'{name} f{s["source_frames"][i]} | '+['GT vehicle IDs (not exact masks)','restored dynamic pixels','restored static pixels'][j],fill='yellow');sheet.paste(im,(j*960,row*536))
  if i==0:
   with np.load(OLD/name/'guard/source_masks_00000.npz') as z:
    candidates=[]
    for key in z.files:
     m=cv2.resize(z[key].astype('uint8'),(W,H),interpolation=cv2.INTER_NEAREST)>0
     if (m&region).sum()<20:continue
     o=orig.copy();o[m]=(.3*o[m]+.7*np.array([255,60,130])).astype('uint8');im=Image.fromarray(o)
     ImageDraw.Draw(im).text((8,8),f'{key} core overlap {int((m&targ).sum())}/{int(targ.sum())}',fill='yellow');candidates.append(im)
    contact=Image.new('RGB',(3*640,((len(candidates)+2)//3)*357))
    for q,im in enumerate(candidates):contact.paste(im.resize((640,357)),((q%3)*640,(q//3)*357))
    contact.save(out/'source_instances.jpg',quality=94)
  rows.append(dict(frame=s['source_frames'][i],target_sam_pixels=int(targ.sum()),target_in_dynamic=int((targ&dyn).sum()),target_in_static=int((targ&static).sum()),gt_boxes=visible))
 sheet.resize((1920,1429)).save(out/'identity_audit.jpg',quality=94);dump(out/'identity_audit.json',rows)
status=REPO/'docs/RESEARCH_STATUS.md';shutil.copy2(status,ROOT/'research_status_before.md')
status.write_text('''# 当前研究状态

更新：2026-09-27，v77。持续目标未完成。上一轮r3–r9为实质进展：新增分阶段控制并定位部分写回污染，未获得合格补景。

当前WS-V77-HYBRID-BG-20260927/r10：先查原RGB、GT身份投影与动态/静态保护mask，固定生成结果，区分目标残留、真实栏杆混色和邻车显露。源图身份审计只用原输入，不从生成画面反推事实。若证据支持，局部matting只处理已定位的前景颜色污染；不加新神经模型，不训练，不扫seed。GPU启动前已确认空闲，没有其他生成进程。

旧DriveEditor默认、GLB、Ω、所有失败完整保留；人工verdict null。审核入口仍为outputs/v77-hybrid-delete/stage-audit/index.html，本轮会补充实际结果。
''',encoding='utf-8')
print('SOURCE AUDIT COMPLETE')

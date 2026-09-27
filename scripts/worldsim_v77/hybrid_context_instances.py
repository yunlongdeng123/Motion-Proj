"""r6：逐车辆条件孔洞；保留车间结构，原r3输出范围不变。"""
import sys, datetime
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from repair_common import camera, hull_mask, largest
import hybrid_context_control as runner
R2=ROOT; R3=ROOT.parent/'r3'; R4=ROOT.parent/'r4'; ROOT=ROOT.parent/'r6'

def prepare():
 assert not (ROOT/'registration.json').exists(); ROOT.mkdir(exist_ok=True)
 cfg=read(R4/'registration.json')
 cfg.update(run_id='r6', registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  hypothesis='r4 hides structures between vehicles; separate GT-constrained instance rectangles preserve those cues while suppressing vehicle context',
  change='separate source-vehicle rectangles instead of single row rectangle; GT projected hull+16px clips SAM outliers',
  source_runs=['r3','r4','r5'], input_selection='development scenes; GT boxes and assistant scene-side rule; not fully automatic',
  fallback=None, human_verdict=None)
 dump(ROOT/'registration.json',cfg)
 for s in cfg['scenes']:
  old=R2/s['name']; r3=R3/s['name']; out=ROOT/s['name']; out.mkdir()
  for sub in ['condition','masked']:(out/sub).mkdir()
  for sub in ['input','write','observed']:(out/sub).symlink_to(r3/sub,target_is_directory=True)
  frames=read(OLD/s['name']/'frames.json'); stats=[]
  for i,f in enumerate(s['source_frames']):
   fr=frames[str(f)]; c,k=camera(fr,s['camera'],HW); gt=np.zeros(HW,bool)
   for b in fr['all_boxes']:
    if b['category'].startswith('vehicle.'):
     gt|=hull_mask(b,c,k,HW,pad=16)
   modelmask=mask(r3/'condition'/f'{i:05}.png'); boxes=[]
   with np.load(OLD/s['name']/'guard'/f'source_masks_{i:05}.npz') as z:
    for key in z.files:
     m=cv2.resize(z[key].astype('uint8'),(W,H),interpolation=cv2.INTER_NEAREST)>0
     m=largest(m&gt); y,x=np.where(m)
     if len(x)<32:continue
     if s['name']=='scene_0230' and x.mean()>W/2:continue
     b=[max(0,int(x.min())-8),max(0,int(y.min())-8),min(W,int(x.max())+9),min(H,int(y.max())+9)]
     modelmask[b[1]:b[3],b[0]:b[2]]=True; boxes.append(b)
   write_mask(out/'condition'/f'{i:05}.png',modelmask)
   orig=rgb(r3/'input'/f'{i:05}.png'); Image.fromarray(np.where(modelmask[...,None],0,orig).astype('uint8')).save(out/'masked'/f'{i:05}.png')
   prev=mask(R4/s['name']/'condition'/f'{i:05}.png')
   stats.append(dict(frame=f,boxes=boxes,model_mask_pixels=int(modelmask.sum()),r4_mask_pixels=int(prev.sum()),reexposed_pixels=int((prev&~modelmask).sum()),write_mask_pixels=int(mask(r3/'write'/f'{i:05}.png').sum())))
  dump(out/'context_stats.json',stats)
  sheet=Image.new('RGB',(W*2,H*4))
  for j,i in enumerate([0,7,15,29]):
   sheet.paste(Image.open(R4/s['name']/'masked'/f'{i:05}.png'),(0,H*j));sheet.paste(Image.open(out/'masked'/f'{i:05}.png'),(W,H*j))
  sheet.resize((960,1072)).save(out/'context_review.jpg',quality=94)
 print('INSTANCE CONTEXT PREPARED',flush=True)

def run():
 runner.ROOT=ROOT; runner.run()
 # runner的对照图标题是r4，因此另存准确标注的r6图，不把它当r4新结果。
 from PIL import ImageDraw
 for s in read(ROOT/'registration.json')['scenes']:
  old=R2/s['name']; out=ROOT/s['name']; sheet=Image.new('RGB',(420*4,236*4))
  for j,i in enumerate([0,7,15,29]):
   y,x=np.where(mask(old/'core'/f'{i:05}.png'));cw=480 if s['name']=='scene_0230' else 400;ch=round(cw*H/W)
   left=round(max(0,min(W-cw,(x.min()+x.max()-cw)/2)));top=round(max(0,min(H-ch,(y.min()+y.max()-ch)/2)))
   for n,(name,path) in enumerate([('original',old/'rgb'),('r4 row',R4/s['name']/'final'),('r6 prior',out/'prior_png'),('r6 instances',out/'final')]):
    im=Image.fromarray(rgb(path/f'{i:05}.png')).crop((left,top,left+cw,top+ch)).resize((420,236));ImageDraw.Draw(im).text((5,5),f'{name} f{s["source_frames"][i]}',fill='yellow');sheet.paste(im,(n*420,j*236))
  sheet.save(out/'quicklook_instances.jpg',quality=94)

if __name__=='__main__':
 if sys.argv[-1]=='prepare':prepare()
 elif sys.argv[-1]=='run':run()
 else:raise ValueError('prepare/run required')

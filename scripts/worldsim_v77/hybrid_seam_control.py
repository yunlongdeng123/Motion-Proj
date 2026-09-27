"""r9：只对0255已有r6输出做固定Poisson边界控制；不改模型和几何。"""
import sys,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from PIL import ImageDraw
R2=ROOT;R3=ROOT.parent/'r3';R6=ROOT.parent/'r6';R8=ROOT.parent/'r8';ROOT=ROOT.parent/'r9';NAME='scene_0255'
assert not (ROOT/'registration.json').exists();ROOT.mkdir(exist_ok=True)
dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r9',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene=NAME,source_runs=['r6','r8'],change='OpenCV NORMAL_CLONE over fixed r3 envelope, then restore r8 protect/observed/outside exactly',fixed=['r6 generation','r3 envelope','r8 fence mask','original RGB'],role='boundary blending control; cannot validate or repair geometry',training=False,model_forwards=0,failure_ledger_refs=['V77-F02'],human_verdict=None))
out=ROOT/NAME;out.mkdir();(out/'final').mkdir();rows=[]
for i in range(30):
 orig=rgb(R2/NAME/'rgb'/f'{i:05}.png');gen=rgb(R6/NAME/'native_png'/f'{i:05}.png');env=mask(R3/NAME/'condition'/f'{i:05}.png');protect=mask(R8/NAME/'protect'/f'{i:05}.png');m=env.astype('uint8')*255;x,y,w,h=cv2.boundingRect(m);center=(x+w//2,y+h//2)
 clone=cv2.seamlessClone(gen,orig,m,center,cv2.NORMAL_CLONE);write=env&~protect;result=orig.copy();result[write]=clone[write]
 obs=mask(R3/NAME/'observed'/f'{i:05}.png')&~protect;e=rgb(R3/NAME/'input'/f'{i:05}.png');result[obs]=e[obs]
 assert np.array_equal(result[protect],orig[protect]);assert np.array_equal(result[~env],orig[~env]);Image.fromarray(result).save(out/'final'/f'{i:05}.png')
 rows.append(dict(frame=65+i,protect_changed=0,outside_envelope_changed=0,write_pixels=int(write.sum())))
dump(out/'pixel_checks.json',rows);sheet=Image.new('RGB',(420*3,236*4))
for j,i in enumerate([0,7,15,29]):
 y,x=np.where(mask(R2/NAME/'core'/f'{i:05}.png'));cw=400;ch=round(cw*H/W);l=round(max(0,min(W-cw,(x.min()+x.max()-cw)/2)));t=round(max(0,min(H-ch,(y.min()+y.max()-ch)/2)))
 for k,(label,path) in enumerate([('r6 old write',R6/NAME/'final'),('r8 mask correction',R8/NAME/'final'),('r9 Poisson boundary',out/'final')]):
  im=Image.fromarray(rgb(path/f'{i:05}.png')).crop((l,t,l+cw,t+ch)).resize((420,236));ImageDraw.Draw(im).text((5,5),f'{label} f{65+i}',fill='yellow');sheet.paste(im,(420*k,236*j))
sheet.save(out/'quicklook.jpg',quality=94);dump(ROOT/'state.json',dict(state='complete',frames=30,gpu_forwards=0,human_verdict=None));print('SEAM CONTROL COMPLETE',flush=True)

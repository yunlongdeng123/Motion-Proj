"""只拆解B→C：输入→flow后像素传播→Transformer prior；不重跑扩散。"""
import sys,os,time,random,types
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_diffueraser as io
from hybrid_common import *
import torch
R2=ROOT;R3=ROOT.parent/'r3';ROOT=R3

def capture_stage(tensor,out,name,is_mask=False):
 d=Path(out).parent/name;d.mkdir(exist_ok=False);a=tensor.detach().float().cpu().numpy()[0]
 if is_mask:
  for i,m in enumerate(a):Image.fromarray(np.clip(m[0]*255,0,255).astype('uint8')).save(d/f'{i:05}.png')
 else:
  for i,im in enumerate(a):Image.fromarray(np.clip((im.transpose(1,2,0)+1)*127.5,0,255).astype('uint8')).save(d/f'{i:05}.png')

assert read(ROOT/'state.json')['state']=='complete'
source=io.SOURCE/'propainter/inference.py';code=source.read_text();needle='        comp_frames = [None] * video_length';assert code.count(needle)==1
code=code.replace(needle,'''        capture_stage(masked_frames, output_path, "B_masked")
        capture_stage(updated_frames, output_path, "C1_propagated")
        capture_stage(updated_masks, output_path, "C1_remaining", True)
        capture_stage(masks_dilated_ori, output_path, "B_mask", True)
'''+needle)
needle='        ##save composed video##';assert code.count(needle)==1;code=code.replace(needle,'        capture(comp_frames, output_path, "C2_transformer")\n'+needle)
mod=types.ModuleType('prior_instrument');mod.__file__=str(source);mod.capture=io.capture;mod.capture_stage=capture_stage;exec(compile(code,str(source),'exec'),mod.__dict__)
mod.read_frame_from_videos=lambda p,l:(io.images(p),10.,(W,H),'probe',30);mod.read_mask=io.exact_prior_mask
torch.set_num_threads(4);p=mod.Propainter(str(io.WEIGHTS/'propainter'),device=torch.device('cuda'));summaries=[]
for s in read(ROOT/'registration.json')['scenes']:
 out=ROOT/s['name'];dest=out/'prior_probe';dest.mkdir(exist_ok=False)
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 p.forward(str(out/'input'),str(out/'condition'),str(dest/'prior.mp4'),video_length=3,mask_dilation=0,save_fps=10,resize_ratio=.6,neighbor_length=10,subvideo_length=50)
 rows=[]
 for i,f in enumerate(s['source_frames']):
  original=mask(dest/'B_mask'/f'{i:05}.png');remaining=mask(dest/'C1_remaining'/f'{i:05}.png');known=original&~remaining;repro=np.array(Image.open(dest/'C2_transformer'/f'{i:05}.png')).astype(int)-np.array(Image.open(out/'prior_png'/f'{i:05}.png')).astype(int)
  rows.append(dict(frame=f,masked_area=int(original.sum()),filled_by_propagation=int(known.sum()),still_missing=int((original&remaining).sum()),reproduction_max_abs=int(np.abs(repro).max()),reproduction_mae=float(np.abs(repro).mean())))
 dump(dest/'stages.json',rows);summaries.append(dict(scene=s['name'],mask_area=sum(r['masked_area'] for r in rows),filled_by_propagation=sum(r['filled_by_propagation'] for r in rows),reproduction_max_abs=max(r['reproduction_max_abs'] for r in rows)))
 sheet=Image.new('RGB',(420*4,236*4))
 from PIL import ImageDraw
 for j,i in enumerate([0,7,15,29]):
  m=mask(R2/s['name']/'core'/f'{i:05}.png');y,x=np.where(m);cw=480 if s['name']=='scene_0230' else 400;ch=round(cw*H/W);left=round(max(0,min(W-cw,(x.min()+x.max()-cw)/2)));top=round(max(0,min(H-ch,(y.min()+y.max()-ch)/2)))
  for k,name in enumerate(['B_masked','C1_propagated','C1_remaining','C2_transformer']):
   im=Image.fromarray(rgb(dest/name/f'{i:05}.png')).crop((left,top,left+cw,top+ch)).resize((420,236));ImageDraw.Draw(im).text((4,5),f'{name} f{s["source_frames"][i]}',fill='yellow');sheet.paste(im,(k*420,j*236))
 sheet.save(dest/'stages.jpg',quality=94);print('PROBED',s['name'],summaries[-1],flush=True)
dump(ROOT/'prior_probe_summary.json',dict(scenes=summaries,scope='re-execution of prior only for intermediate capture, not new diffusion candidate',human_verdict=None))

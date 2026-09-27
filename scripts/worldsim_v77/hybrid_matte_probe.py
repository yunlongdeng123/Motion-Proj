"""r11：高分辨率栏杆matting，先验证首帧trimap；不运行生成模型。"""
import sys,datetime,time
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
sys.path.insert(0,'/root/autodl-tmp/vendor/v77_pymatting_1_1_16')
from hybrid_common import *
from PIL import ImageDraw
from pymatting import estimate_alpha_cf,estimate_foreground_ml
R2=ROOT;BASE=ROOT.parent;ROOT=BASE/'r11';NAME='scene_0255'
assert not (ROOT/'registration.json').exists();ROOT.mkdir(exist_ok=True)
cfg=read(BASE/'r8/registration.json');geometry=cfg['geometry']
dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r11',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene=NAME,source_runs=['r6','r8','r10'],hypothesis='binary copied fence pixels mix original vehicle color; use high-resolution trimap alpha and foreground color estimation',scope='first-frame matte preparation then bounded 30-frame transfer only after image review',manual=True,pymatting='1.1.16 isolated vendor, no new neural weights',fixed=['r6 native generation','original RGB','other protected objects'],failure_ledger_refs=['V77-F02'],human_verdict=None))
im=np.array(Image.open('/root/autodl-tmp/data/v76_vadgs/scene_0255/images/065_3.jpg').convert('RGB'));fh,fw=im.shape[:2]
scale=np.diag([fw/W,fh/H]);shift=np.array([fw/W,fh/H])*.5-.5
def full(points):return np.array(points,float)@scale+shift
gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(float)
def sample(a,xy):return cv2.remap(a.astype('float32'),xy[:,0].astype('float32')[:,None],xy[:,1].astype('float32')[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[:,0]
refined=[];audit=[]
for line in geometry['bars']:
 line=full(line);t=np.linspace(.12,.78,45);xy=line[0][None]*(1-t[:,None])+line[1][None]*t[:,None];n=np.array([1.,0.])
 scores=[]
 for off in range(-10,11):
  q=xy+off*n;score=np.median(sample(gray,q)-.5*(sample(gray,q+4*n)+sample(gray,q-4*n)));scores.append(float(score))
 off=int(np.argmax(scores))-10;refined.append(line+off*n);audit.append(dict(original=line.tolist(),offset_x=off,contrast_scores=scores))
refinedlines=[full(l) for l in geometry['lines']];panel=full(geometry['panel'])
known=np.zeros((fh,fw),np.uint8);support=np.zeros_like(known);base=np.zeros_like(known)
for l in refinedlines:
 points=np.rint(l).astype('int32');cv2.polylines(known,[points],False,1,1);cv2.polylines(support,[points],False,1,11);cv2.polylines(base,[points],False,1,3)
for l in refined:
 points=np.rint(l).astype('int32');cv2.polylines(known,[points],False,1,1);cv2.polylines(support,[points],False,1,9);cv2.polylines(base,[points],False,1,2)
pm=np.zeros_like(known);cv2.fillPoly(pm,[np.rint(panel).astype('int32')],1)
known|=cv2.erode(pm,np.ones((9,9),np.uint8));support|=cv2.dilate(pm,np.ones((11,11),np.uint8));base|=pm
trimap=np.zeros_like(gray);trimap[support>0]=.5;trimap[known>0]=1
crop=(570,495,975,665);x1,y1,x2,y2=crop;img=im[y1:y2,x1:x2]/255.;tri=trimap[y1:y2,x1:x2]
t=time.time();alpha=estimate_alpha_cf(img,tri,cg_kwargs={'maxiter':2000});fg=estimate_foreground_ml(img,alpha)
np.savez_compressed(ROOT/'matte_first.npz',alpha=alpha,foreground=fg,trimap=tri,crop=np.array(crop),bars=np.array(refined),lines=np.array(refinedlines,dtype=object))
Image.fromarray(np.rint(alpha*255).astype('uint8')).save(ROOT/'alpha_first.png');Image.fromarray(np.rint(tri*255).astype('uint8')).save(ROOT/'trimap_first.png')
gen=np.array(Image.open(BASE/'r6'/NAME/'native_png/00000.png').convert('RGB').resize((fw,fh),Image.Resampling.BILINEAR));b=gen[y1:y2,x1:x2]/255.
hard=np.where(base[y1:y2,x1:x2,None]>0,img,b);raw=alpha[...,None]*img+(1-alpha[...,None])*b;matte=alpha[...,None]*fg+(1-alpha[...,None])*b
views=[('original',img),('geometry binary',hard),('alpha only',raw),('alpha + foreground',matte),('alpha',np.repeat(alpha[...,None],3,axis=-1))]
sheet=Image.new('RGB',(810,340*len(views)))
for j,(name,a) in enumerate(views):
 v=Image.fromarray(np.rint(np.clip(a,0,1)*255).astype('uint8')).resize((810,340));ImageDraw.Draw(v).text((5,5),name,fill='yellow');sheet.paste(v,(0,340*j))
sheet.save(ROOT/'matte_probe.jpg',quality=96)
overlay=Image.fromarray(im);d=ImageDraw.Draw(overlay)
for l in refined:d.line([tuple(p) for p in l],fill='cyan',width=1)
overlay.crop(crop).resize((1215,510)).save(ROOT/'centerlines.jpg',quality=96)
dump(ROOT/'matte_first.json',dict(crop=crop,seconds=time.time()-t,bar_refinement=audit,trimap_known_fg=int((tri==1).sum()),trimap_unknown=int((tri==.5).sum()),alpha_pixels_gt_half=int((alpha>.5).sum()),source='only original f65 RGB, not generated reference',human_verdict=None))
print('FIRST FRAME MATTE COMPLETE',time.time()-t,flush=True)

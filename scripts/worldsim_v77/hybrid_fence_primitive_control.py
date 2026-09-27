"""r21：显式细杆alpha/颜色模型；冻结r18视频，只改变围栏写回。

与旧CF-matting不同，不把原混合RGB直接复制为细杆颜色。
"""
import sys,datetime,time
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import read,dump,write_mask
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw
cv2.setNumThreads(4)
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');ROOT=BASE/'r21';DATA=Path('/root/autodl-tmp/data/v76_vadgs/scene_0255');HW=(900,1600)

def sample(im,xy):
 return cv2.remap(im.astype('float32'),xy[...,0].astype('float32'),xy[...,1].astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def fit_stroke(im,p0,p1):
 vec=p1-p0;length=np.linalg.norm(vec);normal=np.array([-vec[1],vec[0]])/length
 t=np.linspace(.10,.90,max(12,int(length*.8/2)));u=np.arange(-8,9,dtype=float)
 centers=p0+(p1-p0)*t[:,None];xy=centers[:,None,:]+u[None,:,None]*normal;obs=sample(im,xy)
 lo=np.median(obs[:,:3],axis=1);hi=np.median(obs[:,-3:],axis=1);bg=lo[:,None]*(1-(u+7)[None,:,None]/14)+hi[:,None]*(u+7)[None,:,None]/14
 # 留出每第三段沿杆位置；不把拟合位置误差当独立验证。
 train=np.arange(len(t))%3!=0;test=~train
 def prediction(p):
  center=p[0]+p[1]*(t-.5);w=p[2];alpha=ndtr((u[None]-center[:,None]+w/2)/.6)-ndtr((u[None]-center[:,None]-w/2)/.6)
  return bg+alpha[...,None]*(p[3:]-bg),alpha
 def residual(p):
  pred,_=prediction(p);err=(pred-obs)[train];regular=np.array([p[0]*.25,p[1]*.6,(p[3]-p[4])*.2,(p[4]-p[5])*.2]);return np.r_[err.ravel(),regular]
 opt=least_squares(residual,np.array([0.,0.,1.2,120.,120.,115.]),bounds=([-6,-5,.25,20,20,20],[6,5,3.,230,230,230]),loss='soft_l1',f_scale=6,max_nfev=250)
 pred,alpha=prediction(opt.x);active=alpha>.08;held=active&test[:,None];error=np.abs(pred-obs).mean(-1);baseline=np.abs(bg-obs).mean(-1)
 q0=p0+normal*(opt.x[0]-.5*opt.x[1]);q1=p1+normal*(opt.x[0]+.5*opt.x[1])
 return dict(p0=q0.tolist(),p1=q1.tolist(),width=float(opt.x[2]),foreground=opt.x[3:].tolist(),offset=float(opt.x[0]),slope=float(opt.x[1]),success=bool(opt.success),evaluations=opt.nfev,heldout_pixels=int(held.sum()),heldout_mae=float(np.mean(error[held])),heldout_no_stroke_mae=float(np.mean(baseline[held])),source='f65 original RGB; local side interpolation estimates background',manual_geometry=True)

def draw_stroke(alpha,premul,stroke):
 p0=np.array(stroke['p0']);p1=np.array(stroke['p1']);v=p1-p0;length=np.linalg.norm(v);v/=length;n=np.array([-v[1],v[0]]);lo=np.floor(np.minimum(p0,p1)-6).astype(int);hi=np.ceil(np.maximum(p0,p1)+6).astype(int);lo=np.maximum(lo,0);hi=np.minimum(hi,[1600,900]);yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]];q=np.stack([xx,yy],-1)-p0;along=q@v;across=q@n
 w=stroke['width'];a=(ndtr((across+w/2)/.6)-ndtr((across-w/2)/.6))*ndtr((along+.5)/.6)*ndtr((length+.5-along)/.6);sl=np.s_[lo[1]:hi[1],lo[0]:hi[0]];premul[sl]=a[...,None]*np.array(stroke['foreground'])+(1-a[...,None])*premul[sl];alpha[sl]=a+(1-a)*alpha[sl]

def prepare():
 assert not ROOT.exists();ROOT.mkdir();start=time.time();geom=read(BASE/'r8/registration.json')['geometry'];scale=np.array([1600/960,900/536]);shift=scale*.5-.5;im=np.array(Image.open(DATA/'images/065_3.jpg').convert('RGB'))
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r21',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='scene_0255',fixed='r18 actual 10 native frames, original source RGB, outside-region protection',
  change='Replace contaminated fence RGB copying with fitted analytic thin-stroke alpha/foreground plus opaque panel interior texture. No original mixed RGB is copied along thin bars.',
  inputs='Assistant r8 traced geometry, original f65 RGB, previously fixed short-window LK similarity; no generated RGB fits material.',
  method='Local RGB background interpolation, constrained Gaussian-blurred opaque line width and foreground fit via SciPy least_squares soft_l1; every third along-line sample held out. Separate panel texture extends eroded interior to edge.',
  assumptions='Thin circular-bar appearance approximated by a 2D line model; local background approximately linear; existing geometry/LK may be inaccurate. This is a diagnostic foreground representation, not automatic photorealism.',
  resources=dict(cpu_threads=4,gpu_forwards=0),stop_rule='One fixed fit and ten-frame composite; keep current native unchanged. Reject structural/temporal artifacts even if pixel residual falls.',failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 strokes=[]
 for j,line in enumerate(geom['bars']):
  p=np.array(line)*scale+shift;s=fit_stroke(im,*p);s.update(kind='vertical',index=j);strokes.append(s)
 for j,line in enumerate(geom['lines']):
  pts=np.array(line)*scale+shift
  for k in range(len(pts)-1):
   if np.linalg.norm(pts[k+1]-pts[k])<14:
    # 弯角短段没有足够独立采样，材质取相邻长杆的已拟合中位数，明确属于几何近似。
    s=dict(p0=pts[k].tolist(),p1=pts[k+1].tolist(),width=2.,foreground=np.median([x['foreground'] for x in strokes],axis=0).tolist(),source='same fence fitted median; short connector geometry')
   else:s=fit_stroke(im,pts[k],pts[k+1])
   s.update(kind='frame',index=j,segment=k);strokes.append(s)
 alpha=np.zeros(HW,np.float32);premul=np.zeros((*HW,3),np.float32)
 for s in strokes:draw_stroke(alpha,premul,s)
 panel=np.zeros(HW,np.uint8);poly=np.array(geom['panel'])*scale+shift;cv2.fillPoly(panel,[np.rint(poly).astype('int32')],1);safe=cv2.erode(panel,np.ones((9,9),np.uint8))>0;_,indices=distance_transform_edt(~safe,return_indices=True);texture=im[indices[0],indices[1]]
 # 8x子像素多边形覆盖，原车周围的混合边缘不进入panel颜色。
 box=(np.floor(poly.min(0))-3).astype(int);top=(np.ceil(poly.max(0))+4).astype(int);ss=8;aa=np.zeros(((top[1]-box[1])*ss,(top[0]-box[0])*ss),np.uint8);cv2.fillPoly(aa,[np.rint((poly-box)*ss).astype('int32')],255);ap=np.zeros(HW,np.float32);ap[box[1]:top[1],box[0]:top[0]]=cv2.resize(aa,tuple(top-box),interpolation=cv2.INTER_AREA)/255
 premul=ap[...,None]*texture+(1-ap[...,None])*premul;alpha=ap+(1-ap)*alpha
 np.savez_compressed(ROOT/'fence_rgba.npz',alpha=alpha,premultiplied=premul);dump(ROOT/'material_fit.json',dict(strokes=strokes,seconds=time.time()-start,human_verdict=None));Image.fromarray(np.uint8(alpha*255)).save(ROOT/'alpha.png')
 views=[]
 for label,bg in [('diagnostic overlay on source (not reconstruction)',im.astype(float)),('flat gray background',np.full_like(im,100,dtype=float))]:
  render=premul+(1-alpha[...,None])*bg;views.append((label,np.clip(render,0,255).astype('uint8')))
 outline=Image.fromarray(im);d=ImageDraw.Draw(outline)
 for s in strokes:d.line([tuple(s['p0']),tuple(s['p1'])],fill='cyan',width=1)
 views.insert(0,('original',im));views.append(('fitted centerlines',np.array(outline)));sheet=Image.new('RGB',(1230,540*len(views)))
 for j,(label,a) in enumerate(views):
  v=Image.fromarray(a).crop((570,490,980,670)).resize((1230,540));ImageDraw.Draw(v).text((8,8),label,fill='yellow');sheet.paste(v,(0,540*j))
 sheet.save(ROOT/'material_review.jpg',quality=96);print('R21_MATERIAL_PREPARED',len(strokes),flush=True)

def composite():
 out=ROOT/'scope_fix';assert (ROOT/'fence_rgba.npz').exists() and not out.exists();out.mkdir();rgba=np.load(ROOT/'fence_rgba.npz');a0=rgba['alpha'];c0=rgba['premultiplied'];stats=read(BASE/'r2/scene_0255/mask_stats.json');T=np.array([[1600/960,0,(1600/960)*.5-.5],[0,900/536,(900/536)*.5-.5],[0,0,1.]])
 for sub in ['final','without_fence','alpha_frames','thin_rgb_control','change_region']:(out/sub).mkdir()
 rows=[]
 def loadmask(p):return np.array(Image.open(p).convert('L').resize((1024,576),Image.Resampling.NEAREST))>127
 for i in range(10):
  H=T@np.array(stats[i]['fence_homography'])@np.linalg.inv(T);alpha=cv2.warpPerspective(a0,H,(1600,900));premul=cv2.warpPerspective(c0,H,(1600,900));alpha=cv2.resize(alpha,(1024,576),interpolation=cv2.INTER_AREA);premul=cv2.resize(premul,(1024,576),interpolation=cv2.INTER_AREA)
  # 沿用原围栏替换范围；其他静态设施及动态保护保留。
  region=np.zeros((536,960),np.uint8);region[299:391,347:578]=1;region=cv2.warpPerspective(region,np.array(stats[i]['fence_homography']),(960,536),flags=cv2.INTER_NEAREST);region=cv2.resize(region,(1024,576),interpolation=cv2.INTER_NEAREST)>0
  static=loadmask(BASE/'r2/scene_0255/static'/f'{i:05}.png');dyn=loadmask(BASE/'r2/scene_0255/dynamic'/f'{i:05}.png');mask=loadmask(BASE/'r15/condition'/f'{i:05}.png');protected=(static&~region)|dyn
  original=np.array(Image.open(BASE/'r15/input'/f'{i:05}.png'));previous=np.array(Image.open(BASE/'r18/final'/f'{i:05}.png'));native=np.array(Image.open(BASE/'r18/native'/f'{i:05}.png'));background=previous.copy();write=mask&region&~protected;background[write]=native[write]
  use=mask&region;alpha[~use]=0;premul[~use]=0;final=np.clip(premul+(1-alpha[...,None])*background,0,255).astype('uint8');assert np.array_equal(final[~mask],original[~mask]);assert np.array_equal(final[~region],previous[~region])
  thin=background.copy();thin[use&(alpha>.15)]=original[use&(alpha>.15)]
  for name,im in [('final',final),('without_fence',background),('thin_rgb_control',thin)]:Image.fromarray(im).save(out/name/f'{i:05}.png')
  Image.fromarray(np.uint8(alpha*255)).save(out/'alpha_frames'/f'{i:05}.png');write_mask(out/'change_region'/f'{i:05}.png',use);rows.append(dict(frame=65+i,alpha_pixels=int((alpha>.01).sum()),outside_condition_changed=0,outside_fence_region_changed=0,dynamic_pixels_in_region=int((dyn&region).sum())))
 dump(out/'state.json',dict(state='complete',frames=10,checks=rows,engineering_correction='Start from frozen r18 final and replace only registered fence ROI; prior r21 whole-mask composition kept as rejected control.',human_verdict=None,background_input_dir=None));print('R21_SCOPE_FIX_COMPLETE',flush=True)

if __name__=='__main__':{'prepare':prepare,'composite':composite}[sys.argv[-1]]()

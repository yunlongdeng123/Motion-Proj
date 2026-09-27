"""r23：已知可见车的局部成像尺度控制，保留r16参考和模型。"""
import os,sys,copy,time,datetime,signal
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/work')
from hybrid_object_positive_control import Engine,BASE,DE,set_seed,load_model,image_tensor,same
from hybrid_common import dump,read,write_mask
import numpy as np,torch,cv2
from torch.nn import functional as F
from PIL import Image
ROOT=BASE/'r23'

def pick_roi(masks):
 union=np.any(masks,axis=0);y,x=np.where(union);lo=np.array([x.min(),y.min()]);hi=np.array([x.max()+1,y.max()+1]);center=(lo+hi)/2
 for w in [384,512,640,768,896,1024]:
  h=w*9//16;xy=np.floor(np.clip(center-np.array([w,h])/2,0,np.array([1024-w,576-h]))/8).astype(int)*8
  if np.all(xy<=lo) and np.all(xy+np.array([w,h])>=hi):return [int(xy[0]),int(xy[1]),w,h]
 raise RuntimeError('model mask union did not fit source image')

def resize_crop(t,roi,size,nearest=False,latent=False):
 x,y,w,h=roi
 if latent:x,y,w,h=x//8,y//8,w//8,h//8
 q=t[...,y:y+h,x:x+w]
 if nearest:return F.interpolate(q,size=size,mode='nearest')
 return F.interpolate(q,size=size,mode='bilinear',align_corners=False,antialias=True)

def crop_payload(p,roi):
 x,y,w,h=roi;sx=1024/w;sy=576/h;assert sx==sy;q=list(copy.deepcopy(p))
 q[0]=resize_crop(p[0],roi,(576,1024));q[2]=resize_crop(p[2],roi,(72,128),True,True);q[3]=resize_crop(p[3],roi,(72,128),True,True);q[9]=resize_crop(p[9],roi,(576,1024))
 for d in q[7]:
  old=d['yx'].clone();d['yx']=old.clone();d['yx'][...,0]=((old[...,0]*576-y+.5)*sy-.5)/576;d['yx'][...,1]=((old[...,1]*1024-x+.5)*sx-.5)/1024;d['box_height']=d['box_height']*sy
 for k in [1,4,5,6,8,10,11]:assert same(p[k],q[k]),k
 assert all(torch.isfinite(t).all() for t in q if isinstance(t,torch.Tensor))
 return tuple(q)

def build():
 e=Engine(False);p=e.payloads['anchor'];m=(p[2][:,0].numpy()>0).astype('uint8');full=np.stack([cv2.resize(x,(1024,576),interpolation=cv2.INTER_NEAREST)>0 for x in m]);roi=pick_roi(full);q=crop_payload(p,roi);e.payloads={'anchor':q};e.arm='anchor'
 return e,p,q,roi,full

def prepare():
 assert not ROOT.exists();ROOT.mkdir();e,p,q,roi,full=build();x,y,w,h=roi
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r23',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='scene_0255',actor='52',camera=3,frames=list(range(130,140)),
  question='Does enlarging the same visible actor in the video latent grid improve reconstruction detail before applying it to actual DELETE?',
  roles='Oracle known-visible reconstruction only. First-frame RGB/SAM plus GT box condition; later masked pixels are evaluation only. Reuse exact r16 conditions, no extra hidden truth input.',
  baseline='Saved r16 official zero-Repositioning with same-view first-frame paste; frozen trained DriveEditor checkpoint',
  intervention='Crop same image/condition/depth/mask/position tensors to a fixed ROI containing all ten original masks, then resize to original 1024x576 inference size. Reference, CLIP, SV3D, angles, obj ratio, seed, steps unchanged.',
  crop_rule='Smallest 16:9 crop in widths 384,512,640,768,896,1024 containing all ten masks, centered and clamped to image on 8px grid. No result-driven crop tuning.',roi_xywh=roi,linear_magnification=1024/w,
  confounds='Crop reduces context and changes apparent scale together; no pure latent-resolution attribution. RGB comes from original r16 1024x576 conditions, so this adds no factual detail beyond r16.',
  fixed=dict(seed=42,steps=25,size=[1024,576],fps=10,frames=10,previous_segment_condition=False,sequential_cfg=True,decode_chunk=1),
  stop_rule='One 10-frame crop control. Inspect original-scale outputs and real RGB; no DELETE extension if identity/detail is worse. Do not turn increased sharpness into fidelity claim.',resources=dict(gpu='RTX3090',cpu_threads=4,timeout_s=600),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 for name in ['input_crop','condition','mask','input_full','full_box','box_crop']:(ROOT/name).mkdir()
 checks=[]
 for i in range(10):
  expected=np.array(Image.open(BASE/'r16/anchor_condition'/f'{i:05}.png'));assert np.array_equal(image_tensor(p[0][i]),expected)
  raw=np.array(Image.open(BASE/'r16/input'/f'{i:05}.png'));Image.fromarray(raw).save(ROOT/'input_full'/f'{i:05}.png');Image.fromarray(raw).crop((x,y,x+w,y+h)).resize((1024,576),Image.Resampling.BILINEAR).save(ROOT/'input_crop'/f'{i:05}.png');Image.fromarray(image_tensor(q[0][i])).save(ROOT/'condition'/f'{i:05}.png')
  cm=cv2.resize((q[2][i,0].numpy()>0).astype('uint8'),(1024,576),interpolation=cv2.INTER_NEAREST)>0;write_mask(ROOT/'mask'/f'{i:05}.png',cm)
  b=np.array(Image.open(BASE/'r16/box_support'/f'{i:05}.png'))>0;write_mask(ROOT/'full_box'/f'{i:05}.png',b);bc=cv2.resize(b[y:y+h,x:x+w].astype('uint8'),(1024,576),interpolation=cv2.INTER_NEAREST)>0;write_mask(ROOT/'box_crop'/f'{i:05}.png',bc)
  # 用公式独立验证归一化位置变换，保持SV3D图内object height不变。
  old=p[7][i]['yx'].numpy()[0]*[576,1024];new=q[7][i]['yx'].numpy()[0]*[576,1024];expected_xy=(old-[y,x]+.5)*[576/h,1024/w]-.5
  assert np.max(np.abs(new-expected_xy))<1e-4
  assert torch.equal(p[7][i]['obj_height'],q[7][i]['obj_height'])
  assert np.array_equal(full[i][y:y+h,x:x+w],cv2.resize(cm.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST)>0)
  checks.append(dict(frame=130+i,original_condition_maxdiff=0,position_transform_error=float(np.max(np.abs(new-expected_xy))),source_box_pixels=int(b.sum()),crop_box_pixels=int(bc.sum()),full_to_crop_mask_roundtrip_exact=True))
 dump(ROOT/'condition_validation.json',dict(roi_xywh=roi,unchanged_payload_indices=[1,4,5,6,8,10,11],checks=checks,human_verdict=None));print('R23_PREPARED',roi,checks[0],flush=True)

def run():
 assert (ROOT/'condition_validation.json').exists() and not (ROOT/'state.json').exists();os.chdir(DE);state=dict(state='loading',pid=os.getpid(),human_verdict=None,background_input_dir=None);dump(ROOT/'state.json',state);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('r23 exceeded 600s')))
 try:
  e,p,q,roi,full=build();assert roi==read(ROOT/'condition_validation.json')['roi_xywh'];e.model=load_model('configs/sample.yaml','cuda',num_steps=25,num_frames=10,verbose=True);e.im_result=[];set_seed(42);torch.cuda.reset_peak_memory_stats();start=time.time();state.update(state='running');dump(ROOT/'state.json',state);signal.alarm(600)
  try:e.predict(1,False,'Deletion')
  finally:signal.alarm(0)
  assert len(e.im_result)==10;(ROOT/'native_crop').mkdir();(ROOT/'restored_full').mkdir();rows=[];x,y,w,h=roi
  for i,im in enumerate(e.im_result):
   Image.fromarray(im).save(ROOT/'native_crop'/f'{i:05}.png');raw=np.array(Image.open(ROOT/'input_full'/f'{i:05}.png'));restored=raw.copy();back=np.array(Image.fromarray(im).resize((w,h),Image.Resampling.BILINEAR));sel=full[i][y:y+h,x:x+w];patch=restored[y:y+h,x:x+w];patch[sel]=back[sel];assert np.array_equal(restored[~full[i]],raw[~full[i]]);Image.fromarray(restored).save(ROOT/'restored_full'/f'{i:05}.png')
   box=np.array(Image.open(ROOT/'full_box'/f'{i:05}.png'))>0;old=np.array(Image.open(BASE/'r16/anchor/native'/f'{i:05}.png'));err_new=np.abs(restored.astype('float32')-raw).mean(-1);err_old=np.abs(old.astype('float32')-raw).mean(-1)
   evalmask=box&full[i];rows.append(dict(frame=130+i,pixels=int(evalmask.sum()),old_mae=float(err_old[evalmask].mean()),crop_mae=float(err_new[evalmask].mean()),outside_model_mask_changed=0))
  state.update(state='complete',frames=10,seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,oracle_metrics=rows,metric_limit='Same original-scale projected box and model mask intersection; includes background/occluders, not exact surface GT. Exclude frame0 when summarizing because it was a condition.');dump(ROOT/'state.json',state);print('R23_COMPLETE',state['seconds'],rows,flush=True)
 except Exception as exc:
  signal.alarm(0);state.update(state='failed_engineering',error=repr(exc));dump(ROOT/'state.json',state);raise

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[-1]]()

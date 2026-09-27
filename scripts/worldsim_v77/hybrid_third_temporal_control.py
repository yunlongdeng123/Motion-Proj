"""r31：固定r28矩形生成与r30写回规则，采用原有同时间重叠条件延展三秒。"""
import os,sys,time,datetime,signal,shutil
from pathlib import Path
os.environ.setdefault('DRIVEEDITOR_SEQUENTIAL_CFG','1');os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128')
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_road_plane_control import BASE,get,HW
from repair_common import read,dump,hull_mask
import numpy as np,cv2
from PIL import Image,ImageDraw
ROOT=BASE/'r31';OLD=BASE.parent/'WS-V77-DELETE-REPAIR-20260927/r1/official_000'

def inputs(f):
 original=np.array(Image.open(OLD/'rgb'/f'{f:05}.png'));write=np.array(Image.open(OLD/'mask'/f'{f:05}.png'))>0;y,x=np.where(write);model=np.zeros(HW,bool);model[max(0,y.min()-8):min(576,y.max()+25),max(0,x.min()-8):min(1024,x.max()+9)]=True
 fr,c,k,actual,_=get(f);assert np.array_equal(original,actual);protect=np.zeros(HW,bool);actors=[]
 for b in fr['all_boxes']:
  if b['actor_id']!='12':
   m=hull_mask(b,c,k,HW);protect|=m
   if np.any(m&model):actors.append(b['actor_id'])
 protect&=model&~write;dist=cv2.distanceTransform(model.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);t=np.clip(dist/8,0,1);alpha=t*t*(3-2*t);alpha[write]=1;alpha[protect]=0
 return original,write,model,protect,alpha,actors

def compose(original,native,alpha):return np.rint(native.astype('float32')*alpha[...,None]+original.astype('float32')*(1-alpha[...,None])).clip(0,255).astype('uint8')

def prepare():
 assert not ROOT.exists();ROOT.mkdir();cv2.setNumThreads(4)
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r31',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',actor='12',camera=0,frames=list(range(30)),
  question='Does the third-scene r28 generation plus r30 feather/protection retain its improvement over30frames with overlapping-window conditioning?',
  prefix='Reuse exact r28 native frames0..9 and exact r30 rect_feather_keep composites, no repeated GPU first-window generation.',
  fixed=dict(model='same frozen DriveEditor',seed=42,steps=25,size=[1024,576],window=10,new_window_starts=[9,18,27],stride=9,last_window='repeat f29 to pad10, retain only first3 chronological frames',previous='At each boundary the same timestamp previous final composite conditions the first frame through the existing previous_segment_last_frame adapter',model_mask='precise bounding rectangle padded8px left/right/top,24px bottom',writeback='r30 smoothstep8px inside rectangle, precise target weight1, other GT envelopes outside target weight0',cpu_threads=4,sequential_cfg=True,decode_chunk=1),
  roles='Original RGB/SAM2 and GT non-target envelopes; previous generated frame is explicitly a temporal appearance condition, not factual background evidence. No new reference or model.',
  comparisons='Old saved precise-mask repair and new30f output. Beyond first10f, generation masks, writeback and previous history differ; this is a deployment continuity check, not a single-variable attribution.',
  stop_rule='Inspect every newly completed window before next. Stop if obvious target regeneration, neighbor damage or collapse; preserve completed prefix and all overlap predictions. Do not enlarge seeds or bypass failures with Omega.',resources=dict(gpu='RTX3090',max_new_windows=3,timeout_per_window_s=600),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 for name in ['input','write_mask','model_mask','protect','alpha','final','native','windows']:(ROOT/name).mkdir()
 checks=[]
 for f in range(30):
  im,wr,m,pr,a,actors=inputs(f)
  for name,arr in [('input',im),('write_mask',np.uint8(wr)*255),('model_mask',np.uint8(m)*255),('protect',np.uint8(pr)*255),('alpha',np.rint(a*255).astype('uint8'))]:Image.fromarray(arr).save(ROOT/name/f'{f:05}.png')
  if f<10:
   native=np.array(Image.open(BASE/'r28/native'/f'{f:05}.png'));expected=np.array(Image.open(BASE/'r30/rect_feather_keep'/f'{f:05}.png'));actual=compose(im,native,a);assert np.array_equal(expected,actual);assert np.array_equal(m,np.array(Image.open(BASE/'r28/model_mask'/f'{f:05}.png'))>0);assert np.array_equal(np.rint(a*255).astype('uint8'),np.array(Image.open(BASE/'r30/alpha'/f'{f:05}.png')));Image.fromarray(native).save(ROOT/'native'/f'{f:05}.png');Image.fromarray(actual).save(ROOT/'final'/f'{f:05}.png')
  checks.append(dict(frame=f,prefix_exact=f<10,non_target_envelopes_in_model=actors,model_pixels=int(m.sum()),write_pixels=int(wr.sum()),protect_pixels=int(pr.sum())))
 dump(ROOT/'condition_validation.json',dict(prefix_exact_frames=10,rows=checks,human_verdict=None));print('R31_PREPARED',flush=True)

def run_window(start):
 assert start in [9,18,27];out=ROOT/'windows'/f'{start:05}';assert not out.exists();out.mkdir();assert (ROOT/'final'/f'{start:05}.png').exists()
 from repair_drive import Engine,set_seed,torch
 torch.set_num_threads(4);cv2.setNumThreads(4);state=dict(state='loading',pid=os.getpid(),window_start=start,human_verdict=None,background_input_dir=None);dump(out/'state.json',state)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('r31 window exceeded600s')))
 try:
  e=Engine();ids=[min(start+j,29) for j in range(10)];valid=min(10,30-start);all_inputs=[inputs(f) for f in ids];e.im=[r[0] for r in all_inputs];e.masks=[r[2] for r in all_inputs];previous=np.array(Image.open(ROOT/'final'/f'{start:05}.png'));e.previous_segment_last_frame=previous;e.im_result=[];set_seed(42);torch.cuda.reset_peak_memory_stats();beg=time.time();state.update(state='running');dump(out/'state.json',state);signal.alarm(600)
  try:e.predict(1,False,'Deletion')
  finally:signal.alarm(0)
  assert len(e.im_result)==10 and e.used_previous_segment_condition;checks=[];sheet=Image.new('RGB',(1600,5*250),(12,20,30));sheets=[sheet,Image.new('RGB',(1600,5*250),(12,20,30))]
  for j,(f,native,data) in enumerate(zip(ids,e.im_result,all_inputs)):
   im,wr,m,pr,a,_=data;result=compose(im,native,a);assert np.array_equal(result[~m],im[~m]);assert np.array_equal(result[pr],im[pr]);assert np.array_equal(result[wr],native[wr]);Image.fromarray(native).save(out/f'{j:02}_native.png');Image.fromarray(result).save(out/f'{j:02}_final.png')
   if 0<j<valid:
    assert not (ROOT/'final'/f'{f:05}.png').exists();Image.fromarray(native).save(ROOT/'native'/f'{f:05}.png');Image.fromarray(result).save(ROOT/'final'/f'{f:05}.png')
   y,x=np.where(wr);left=int(np.clip((x.min()+x.max())/2-192,0,640));top=int(np.clip((y.min()+y.max())/2-108,0,360));box=(left,top,left+384,top+216)
   old=np.array(Image.open(OLD/'precise'/f'{f:05}.png'))
   for col,(name,arr) in enumerate([('original',im),('old precise',old),('new final',result),('new native',native)]):
    tile=Image.new('RGB',(400,250),(12,20,30));tile.paste(Image.fromarray(arr).crop(box).resize((400,225)),(0,25));ImageDraw.Draw(tile).text((5,5),f'f{f} {name}',fill='white');sheets[j//5].paste(tile,(col*400,(j%5)*250))
   checks.append(dict(frame=f,duplicate_padding=j>=valid,kept_from_previous=j==0,protect_changed=0,outside_model_changed=0))
  for i,s in enumerate(sheets):s.save(out/f'all10_part{i+1}.jpg',quality=96)
  mask=all_inputs[0][2];overlap=np.array(Image.open(out/'00_final.png'));delta=np.abs(overlap.astype('float32')-previous).mean(-1);state.update(state='complete_pending_visual_review',source_frames=ids,retained_new_frames=valid-1,seconds=time.time()-beg,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,overlap_same_timestamp_mae_in_model=float(delta[mask].mean()),previous_generated_condition=True,checks=checks);dump(out/'state.json',state);print('R31_WINDOW_COMPLETE',start,state['seconds'],state['overlap_same_timestamp_mae_in_model'],flush=True)
 except Exception as exc:
  signal.alarm(0);state.update(state='failed_engineering',error=repr(exc));dump(out/'state.json',state);raise
if __name__=='__main__':
 if sys.argv[-1]=='prepare':prepare()
 else:run_window(int(sys.argv[-1]))

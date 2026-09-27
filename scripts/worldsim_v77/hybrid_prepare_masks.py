"""固定两场景原RGB；SAM2多对象传播保护邻车和可见静态设施。"""
import datetime,time,torch
from PIL import ImageDraw
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from hybrid_common import *
torch.set_num_threads(4);cv2.setNumThreads(4)
sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
from sam2.build_sam import build_sam2,build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
CHECK='/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt'
CFG='configs/sam2.1/sam2.1_hiera_l.yaml'
def main():
 assert not (ROOT/'registration.json').exists()
 ROOT.mkdir(parents=True,exist_ok=True)
 scenes=read(OLD/'registration.json')['scenes'][:2]
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r2',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scenes=scenes,seed=42,frames_per_scene=30,hw=list(HW),fps=10,pcm='2-Step',max_img_size=960,generate_dilation_px=16,scope='已曝光两开发scene；GT提示与标定辅助；三mask/LK证据/原Ω证据/DiffuEraser一次候选；无训练',failure_ledger_refs=['V77-F02'],human_verdict=None))
 modelpath='/root/autodl-tmp/models/worldsim_v77/grounding-dino-tiny'
 proc=AutoProcessor.from_pretrained(modelpath,local_files_only=True)
 det=AutoModelForZeroShotObjectDetection.from_pretrained(modelpath,local_files_only=True).cuda().eval()
 sam=SAM2ImagePredictor(build_sam2(CFG,CHECK,device='cuda'))
 for s in scenes:
  out=ROOT/s['name'];out.mkdir(exist_ok=True)
  for sub in ['rgb','raw_video','core','delete','protect','generate','static','dynamic','seeds']:(out/sub).mkdir(exist_ok=True)
  for i,f in enumerate(s['source_frames']):
   im=Image.fromarray(rgb(pathlib.Path(s['data'])/'images'/f'{f:03}_{s["camera"]}.jpg'))
   im.save(out/'rgb'/f'{i:05}.png');im.save(out/'raw_video'/f'{i:05}.jpg',quality=98)
  im=rgb(out/'rgb/00000.png');core=mask(OLD/s['name']/'sam/core_00000.png')
  seeds=[dict(id=1,role='target',source='existing SAM2 target core',mask=core)]
  with np.load(OLD/s['name']/'guard/source_masks_00000.npz') as z:
   for key in z.files:
    m=cv2.resize(z[key].astype('uint8'),(W,H),interpolation=cv2.INTER_NEAREST)>0
    if (m&core).sum()/max(1,core.sum())<=.05 and m.sum()>80:
     seeds.append(dict(id=len(seeds)+1,role='neighbor',source='factual vehicle detector + SAM2',mask=m&~core))
  inputs=proc(images=Image.fromarray(im),text='metal fence. railing. street light pole. traffic sign. tree trunk.',return_tensors='pt').to('cuda')
  with torch.inference_mode():pred=det(**inputs)
  found=proc.post_process_grounded_object_detection(pred,inputs.input_ids,box_threshold=.25,text_threshold=.25,target_sizes=[HW])[0]
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   sam.set_image(im)
   for b,l,score in zip(found['boxes'],found['labels'],found['scores']):
    m,_,_=sam.predict(box=b.cpu().numpy(),multimask_output=False);m=m[0]>0
    if m.sum()>30 and m.sum()<H*W*.45:
     seeds.append(dict(id=len(seeds)+1,role='static',source=str(l),score=float(score),box=b.cpu().tolist(),mask=m))
  for a in seeds:write_mask(out/'seeds'/f'{a["id"]:02}_{a["role"]}.png',a['mask'])
  dump(out/'seed_objects.json',[{k:v for k,v in a.items() if k!='mask'} for a in seeds])
  print(s['name'],'seeds',[(a['id'],a['role'],a.get('source'),int(a['mask'].sum())) for a in seeds],flush=True)
 del det,sam;torch.cuda.empty_cache()
 predictor=build_sam2_video_predictor(CFG,CHECK,device='cuda')
 summary=[]
 for s in scenes:
  out=ROOT/s['name'];seeds=read(out/'seed_objects.json');rows=[];frames=read(OLD/s['name']/'frames.json')
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   state=predictor.init_state(video_path=str(out/'raw_video'),offload_video_to_cpu=True,offload_state_to_cpu=True)
   for a in seeds:predictor.add_new_mask(state,frame_idx=0,obj_id=a['id'],mask=mask(out/'seeds'/f'{a["id"]:02}_{a["role"]}.png'))
   for i,obj_ids,logits in predictor.propagate_in_video(state):
    f=s['source_frames'][i];v=logits[:,0].float().cpu().numpy();win=np.argmax(v,axis=0);positive=v.max(axis=0)>0
    static=np.zeros(HW,bool);dyn=np.zeros(HW,bool)
    for j,oid in enumerate(obj_ids):
     role=next(a['role'] for a in seeds if a['id']==oid);m=(win==j)&positive
     if role=='static':static|=m
     elif role=='neighbor':dyn|=m
    oldcore=mask(OLD/s['name']/'sam'/f'core_{i:05}.png')
    # 独立逐帧原图检测补充新进入画面的邻车；不把目标候选算进保护。
    with np.load(OLD/s['name']/'guard'/f'source_masks_{i:05}.npz') as z:
     for key in z.files:
      m=cv2.resize(z[key].astype('uint8'),(W,H),interpolation=cv2.INTER_NEAREST)>0
      if (m&oldcore).sum()/max(1,oldcore.sum())<=.05:dyn|=m&~oldcore
    protect=static|dyn
    # 显式可见前景优先，记录从旧目标剥离的像素，不掩饰该变化。
    core=oldcore&~protect
    delete=(cv2.dilate(core.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0)&~protect
    m=mask_contract(delete,protect,np.zeros(HW,bool),16)
    for k in ['delete','protect','generate']:write_mask(out/k/f'{i:05}.png',m[k])
    for k,a in [('core',core),('static',static),('dynamic',dyn)]:write_mask(out/k/f'{i:05}.png',a)
    rows.append(dict(frame=f,old_core=int(oldcore.sum()),core=int(core.sum()),core_removed_by_protect=int((oldcore&protect).sum()),delete=int(delete.sum()),protect=int(protect.sum()),generate=int(m['generate'].sum())))
   predictor.reset_state(state)
  sheet=Image.new('RGB',(W*2,H*3))
  for j,i in enumerate([0,15,29]):
   im=rgb(out/'rgb'/f'{i:05}.png');annot=im.copy();p=mask(out/'protect'/f'{i:05}.png');g=mask(out/'generate'/f'{i:05}.png');d=mask(out/'delete'/f'{i:05}.png')
   annot[p]=(.45*annot[p]+.55*np.array([0,220,230])).astype('uint8');annot[g]=(.45*annot[g]+.55*np.array([60,80,255])).astype('uint8');annot[d]=(.45*annot[d]+.55*np.array([255,210,0])).astype('uint8')
   sheet.paste(Image.fromarray(im),(0,j*H));sheet.paste(Image.fromarray(annot),(W,j*H))
  sheet.resize((1280,1072)).save(out/'mask_review.jpg',quality=94)
  dump(out/'mask_stats.json',rows);summary.append(dict(scene=s['name'],frames=30,core_removed_by_protect=sum(r['core_removed_by_protect'] for r in rows),old_core=sum(r['old_core'] for r in rows),seeds=seeds))
  print(s['name'],'MASKS_COMPLETE',flush=True)
 dump(ROOT/'mask_summary.json',summary)
if __name__=='__main__':main()

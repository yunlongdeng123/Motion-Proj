"""三组新控制共180帧语义检查，原图检测只运行一次。"""
import sys,time,os,argparse
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from repair_guard import classify
import torch
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
R2=ROOT;R3=ROOT.parent/'r3';R4=ROOT.parent/'r4'
ap=argparse.ArgumentParser();ap.add_argument('--followup',action='store_true');args=ap.parse_args()
REPORT=ROOT.parent/'r6' if args.followup else R4
DEST=REPORT/'guard';DEST.mkdir(exist_ok=False)
if args.followup:
 assert read(ROOT.parent/'r5/state.json')['state']=='complete';assert read(ROOT.parent/'r6/state.json')['state']=='complete'
else:
 assert read(R4/'state.json')['state']=='complete';assert read(R3/'noise_control/state.json')['state']=='complete'
torch.set_num_threads(4);cv2.setNumThreads(4)
p='/root/autodl-tmp/models/worldsim_v77/grounding-dino-tiny';processor=AutoProcessor.from_pretrained(p,local_files_only=True);model=AutoModelForZeroShotObjectDetection.from_pretrained(p,local_files_only=True).cuda().eval()
sam=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',device='cuda'))
def detect(im):
 inputs=processor(images=Image.fromarray(im),text='car. truck. bus. van.',return_tensors='pt').to('cuda')
 with torch.inference_mode():o=model(**inputs)
 result=processor.post_process_grounded_object_detection(o,inputs.input_ids,box_threshold=.25,text_threshold=.25,target_sizes=[HW])[0];boxes=result['boxes'].cpu().numpy()
 if not len(boxes):return []
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):sam.set_image(im);m,_,_=sam.predict(box=boxes,multimask_output=False)
 if m.ndim==4:m=m[:,0]
 return [dict(box=b.tolist(),score=float(sc),mask=mm>0) for b,sc,mm in zip(boxes,result['scores'],m)]
summaries=[]
for s in read(R4/'registration.json')['scenes']:
 dest=DEST/s['name'];dest.mkdir();old=R2/s['name'];rows=[];arms={'r3_rect':R3/s['name']/'final','r3_noise':R3/'noise_control'/s['name']/'final','r4_context':R4/s['name']/'final'}
 if args.followup:arms={run:ROOT.parent/run/s['name']/'final' for run in ['r5','r6']}
 for i,f in enumerate(s['source_frames']):
  original=rgb(old/'rgb'/f'{i:05}.png');core=mask(old/'core'/f'{i:05}.png');edit=mask(R3/s['name']/'write'/f'{i:05}.png');protect=mask(old/'protect'/f'{i:05}.png');source=detect(original)
  targets=[r for r in source if (r['mask']&core).sum()/max(1,core.sum())>.3];neighbors=[r['mask'] for r in source if (r['mask']&core).sum()/max(1,core.sum())<=.05];nu=np.logical_or.reduce(neighbors) if neighbors else np.zeros(HW,bool)
  row=dict(frame=f,source_positive=bool(targets),arms={})
  for arm,path in arms.items():
   im=rgb(path/f'{i:05}.png');det=detect(im);details=[dict(box=r['box'],score=r['score'],**classify(r['mask'],core,edit,neighbors)) for r in det];changed=np.any(im!=original,axis=-1)
   row['arms'][arm]=dict(blocked=any(d['suspect_new_vehicle'] for d in details),detections=details,protect_changed=int((changed&protect).sum()),neighbor_pixels=int(nu.sum()),neighbor_changed=int((changed&nu).sum()))
  rows.append(row);dump(dest/'frames.json',rows);print(s['name'],f,{a:r['blocked'] for a,r in row['arms'].items()},flush=True)
 summary=dict(scene=s['name'],source_positive=sum(r['source_positive'] for r in rows),arms={a:dict(blocked_frames=[r['frame'] for r in rows if r['arms'][a]['blocked']],protect_changed=sum(r['arms'][a]['protect_changed'] for r in rows),neighbor_changed=sum(r['arms'][a]['neighbor_changed'] for r in rows),neighbor_pixels=sum(r['arms'][a]['neighbor_pixels'] for r in rows)) for a in arms},human_verdict=None);summaries.append(summary);dump(dest/'summary.json',summary)
dump(REPORT/'guard_summary.json',dict(scenes=summaries,model='GroundingDINO-tiny + SAM2.1 large',scope='same r2 gate; zero detection is not visual pass',human_verdict=None))

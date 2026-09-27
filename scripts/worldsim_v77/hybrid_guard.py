"""DELETE再生车检查；原图正控制、旧0230幻觉正控制及空路面负控制。"""
import torch,time
from PIL import ImageDraw
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from hybrid_common import *
from repair_guard import classify
sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def main():
 assert read(ROOT/'generation_state.json')['state']=='complete'
 torch.set_num_threads(4);cv2.setNumThreads(4)
 modelpath='/root/autodl-tmp/models/worldsim_v77/grounding-dino-tiny'
 processor=AutoProcessor.from_pretrained(modelpath,local_files_only=True);model=AutoModelForZeroShotObjectDetection.from_pretrained(modelpath,local_files_only=True).cuda().eval()
 sam=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',device='cuda'))
 def detect(im):
  inputs=processor(images=Image.fromarray(im),text='car. truck. bus. van.',return_tensors='pt').to('cuda')
  with torch.inference_mode():outputs=model(**inputs)
  found=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,box_threshold=.25,text_threshold=.25,target_sizes=[HW])[0]
  boxes=found['boxes'].cpu().numpy();rows=[]
  if len(boxes):
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):sam.set_image(im);ms,_,_=sam.predict(box=boxes,multimask_output=False)
   if ms.ndim==4:ms=ms[:,0]
   for b,m,score,label in zip(boxes,ms,found['scores'],found['labels']):rows.append(dict(box=b.tolist(),mask=m>0,score=float(score),label=str(label)))
  return rows
 summaries=[]
 for s in read(ROOT/'registration.json')['scenes']:
  out=ROOT/s['name'];dest=out/'guard';dest.mkdir(exist_ok=False);rows=[]
  for i,f in enumerate(s['source_frames']):
   original=rgb(out/'rgb'/f'{i:05}.png');core=mask(out/'core'/f'{i:05}.png');edit=mask(out/'generate'/f'{i:05}.png');source=detect(original)
   targets=[r for r in source if (r['mask']&core).sum()/max(1,core.sum())>.3];neighbors=[r['mask'] for r in source if (r['mask']&core).sum()/max(1,core.sum())<=.05]
   road=np.zeros(HW,bool);road[440:465,750:840]=True;road_hits=sum(classify(r['mask'],road,road,[])['suspect_new_vehicle'] for r in source)
   record=dict(frame=f,source_target_detected=bool(targets),source_vehicle_count=len(source),negative_control_hits=int(road_hits),arms={})
   for arm in ['prior_png','final']:
    im=rgb(out/arm/f'{i:05}.png');det=detect(im);details=[];overlay=Image.fromarray(im);draw=ImageDraw.Draw(overlay)
    for r in det:
     d={**{k:v for k,v in r.items() if k!='mask'},**classify(r['mask'],core,edit,neighbors)};details.append(d)
     if d['suspect_new_vehicle']:draw.rectangle(r['box'],outline='red',width=3)
    protect=mask(out/'protect'/f'{i:05}.png');changed=np.any(im!=original,axis=-1);nu=np.logical_or.reduce(neighbors) if neighbors else np.zeros(HW,bool)
    record['arms'][arm]=dict(suspect=any(d['suspect_new_vehicle'] for d in details),detections=details,source_neighbor_pixels=int(nu.sum()),source_neighbor_changed_pixels=int((changed&nu).sum()),protect_pixels=int(protect.sum()),protect_changed_pixels=int((changed&protect).sum()))
    overlay.save(dest/f'{arm}_{i:05}.jpg',quality=94)
   rows.append(record);dump(dest/'frames.json',rows);print(s['name'],f,'source',bool(targets),'suspect',record['arms']['final']['suspect'],flush=True)
  summary=dict(scene=s['name'],frames=30,source_target_detected_frames=sum(r['source_target_detected'] for r in rows),negative_control_hit_frames=[r['frame'] for r in rows if r['negative_control_hits']],arms={a:dict(blocked_frames=[r['frame'] for r in rows if r['arms'][a]['suspect']],protect_changed_pixels=sum(r['arms'][a]['protect_changed_pixels'] for r in rows),source_neighbor_changed_pixels=sum(r['arms'][a]['source_neighbor_changed_pixels'] for r in rows),source_neighbor_pixels=sum(r['arms'][a]['source_neighbor_pixels'] for r in rows)) for a in ['prior_png','final']},human_verdict=None)
  if s['name']=='scene_0230':
   old=rgb(FULL/s['name']/'cam5/background/00025.png');core=mask(out/'core/00007.png');edit=mask(out/'generate/00007.png');summary['known_hallucination_control']=[{**{k:v for k,v in r.items() if k!='mask'},**classify(r['mask'],core,edit,[])} for r in detect(old)]
  dump(dest/'summary.json',summary);summaries.append(summary)
 dump(ROOT/'guard_summary.json',dict(scenes=summaries,model='GroundingDINO-tiny + SAM2.1 large',thresholds=dict(box=.25,text=.25,minimum_edit_pixels=32,minimum_core_fraction=.05,neighbor_iou=.5),scope='zero detection does not certify background; visual review required',human_verdict=None))
if __name__=='__main__':main()

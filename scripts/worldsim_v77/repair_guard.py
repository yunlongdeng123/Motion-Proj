"""生成后车辆检查；源图正控制、原有邻车匹配及旧幻觉控制一同保存。"""
import torch,time
from PIL import Image,ImageDraw
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from repair_common import *
sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
MODEL='/root/autodl-tmp/models/worldsim_v77/grounding-dino-tiny'

def iou(a,b):return int((a&b).sum())/max(1,int((a|b).sum()))
def classify(m,core,edit,neighbors):
 area=int(m.sum());hit=int((m&core).sum());edited=int((m&edit).sum());matched=max([iou(m,n) for n in neighbors]+[0.])
 suspect=edited>=32 and hit/max(1,int(core.sum()))>=.05 and matched<.5
 return {'pixels':area,'target_core_overlap':hit,'target_core_fraction':hit/max(1,int(core.sum())),'edit_overlap':edited,'original_neighbor_iou':matched,'suspect_new_vehicle':bool(suspect)}

def main():
 assert read(ROOT/'drive_state.json')['state'] in ['complete','stopped_regression']
 assert not (ROOT/'guard_summary.json').exists()
 torch.set_num_threads(4);cv2.setNumThreads(4)
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);model=AutoModelForZeroShotObjectDetection.from_pretrained(MODEL,local_files_only=True).eval().cuda()
 sam=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',device='cuda'))
 def detect(rgb):
  inputs=processor(images=Image.fromarray(rgb),text='car. truck. bus. van.',return_tensors='pt').to('cuda')
  with torch.inference_mode():outputs=model(**inputs)
  r=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,box_threshold=.25,text_threshold=.25,target_sizes=[rgb.shape[:2]])[0]
  boxes=r['boxes'].detach().cpu().numpy();scores=r['scores'].detach().cpu().numpy();labels=r['labels'];rows=[]
  if len(boxes):
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    sam.set_image(rgb);masks,_,_=sam.predict(box=boxes,multimask_output=False)
   if masks.ndim==4:masks=masks[:,0]
   for b,v,l,m in zip(boxes,scores,labels,masks):rows.append({'box':b.tolist(),'score':float(v),'label':str(l),'mask':m>0})
  return rows
 allrows=[];started=time.monotonic()
 for s in read(ROOT/'registration.json')['scenes']:
  out=ROOT/s['name'];dest=out/'guard';dest.mkdir(exist_ok=False);records=[]
  count=next(x['comparison_count'] for x in read(ROOT/'comparison_plan.json')['scenes'] if x['scene']==s['name']) if (ROOT/'comparison_plan.json').exists() else 30
  for i,f in enumerate(s['source_frames'][:count]):
   core=cv2.imread(str(out/'sam'/f'core_{i:05}.png'),0)>0;edit=cv2.imread(str(out/'mask'/f'{i:05}.png'),0)>0
   orig=np.array(Image.open(out/'rgb'/f'{i:05}.png'));source=detect(orig)
   targets=[r for r in source if (r['mask']&core).sum()/max(1,core.sum())>.3]
   neighbors=[r['mask'] for r in source if (r['mask']&core).sum()/max(1,core.sum())<=.05]
   np.savez_compressed(dest/f'source_masks_{i:05}.npz',**{f'mask_{j}':r['mask'] for j,r in enumerate(source)})
   road=np.zeros_like(core);road[470:500,800:900]=True
   road_hits=sum(classify(r['mask'],road,road,[])['suspect_new_vehicle'] for r in source)
   row={'frame':f,'source_target_detected':bool(targets),'source_vehicles':len(source),'source_neighbor_instances':len(neighbors),'empty_road_negative_control_hits':int(road_hits),'arms':{}}
   for arm in ['precise','evidence_first']:
    rgb=np.array(Image.open(out/arm/f'{i:05}.png'));found=detect(rgb);details=[];overlay=Image.fromarray(rgb);d=ImageDraw.Draw(overlay)
    np.savez_compressed(dest/f'{arm}_masks_{i:05}.npz',**{f'mask_{j}':r['mask'] for j,r in enumerate(found)})
    for r in found:
     q=classify(r['mask'],core,edit,neighbors);details.append({**{k:v for k,v in r.items() if k!='mask'},**q})
     if q['suspect_new_vehicle']:d.rectangle(r['box'],outline='red',width=3);d.text((r['box'][0],max(0,r['box'][1]-14)),f"vehicle {r['score']:.2f}",fill='red')
    blocked=any(d['suspect_new_vehicle'] for d in details)
    status='blocked_vehicle' if blocked else ('detector_clear_needs_visual_review' if targets else 'unknown_detector_source_miss')
    neighbor_union=np.logical_or.reduce(neighbors) if neighbors else np.zeros_like(core)
    changed=np.any(rgb!=orig,axis=-1)
    row['arms'][arm]={'status':status,'detections':details,'source_neighbor_pixels':int(neighbor_union.sum()),'source_neighbor_changed_pixels':int((changed&neighbor_union).sum()),'source_neighbor_mask_overlap':int((edit&neighbor_union).sum())};overlay.save(dest/f'{arm}_{i:05}.jpg',quality=92)
   records.append(row);dump(dest/'frames.json',records);print(s['name'],f,{k:v['status'] for k,v in row['arms'].items()},flush=True)
  # 已知旧失败：固定0230 f25原生补景，guard是否真能识别，不只测试干净结果。
  legacy=None
  if s['name']=='scene_0230':
   f=25;i=s['source_frames'].index(f);rgb=np.array(Image.open(FULL/s['name']/'cam5/background'/f'{f:05}.png'));core=cv2.imread(str(out/'sam'/f'core_{i:05}.png'),0)>0;edit=cv2.imread(str(out/'mask'/f'{i:05}.png'),0)>0
   legacy=[{**{k:v for k,v in r.items() if k!='mask'},**classify(r['mask'],core,edit,[])} for r in detect(rgb)]
  summary={'scene':s['name'],'source_target_detected_frames':sum(r['source_target_detected'] for r in records),'frames':count,'empty_road_control_rect_xyxy':[800,470,900,500],'empty_road_control_hit_frames':[r['frame'] for r in records if r['empty_road_negative_control_hits']>0],'arms':{a:{'blocked_frames':[r['frame'] for r in records if r['arms'][a]['status']=='blocked_vehicle'],'source_miss_frames':[r['frame'] for r in records if r['arms'][a]['status']=='unknown_detector_source_miss'],'source_neighbor_pixels':sum(r['arms'][a]['source_neighbor_pixels'] for r in records),'source_neighbor_changed_pixels':sum(r['arms'][a]['source_neighbor_changed_pixels'] for r in records)} for a in ['precise','evidence_first']},'legacy_f25_control':legacy,'human_verdict':None}
  dump(dest/'summary.json',summary);allrows.append(summary)
 dump(ROOT/'guard_summary.json',{'scenes':allrows,'elapsed_s':time.monotonic()-started,'model':'GroundingDINO-tiny + SAM2.1 large','thresholds':{'box':.25,'text':.25,'edit_overlap_pixels':32,'target_core_fraction':.05,'non_target_source_instance_iou':.5},'scope':'车辆再生拦截；未覆盖阴影/结构/时序质量；检测空白不能作为真实背景证明。','human_verdict':None})
if __name__=='__main__':main()

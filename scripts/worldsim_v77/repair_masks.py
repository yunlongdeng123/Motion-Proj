"""保留原SAM；第三场景视频跟踪；凸包仅约束实例，不能替代实例。"""
from repair_common import *
import torch,time
from PIL import Image,ImageDraw
torch.set_num_threads(4);cv2.setNumThreads(4);torch.manual_seed(42)
reg=read(ROOT/'registration.json');s=reg['scenes'][2];out=ROOT/s['name'];frames=read(out/'frames.json')
sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
from sam2.build_sam import build_sam2_video_predictor
p=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml','/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',device='cuda')
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 state=p.init_state(video_path=str(out/'raw_video'),offload_video_to_cpu=True,offload_state_to_cpu=True)
 fr=frames['0'];b=next(b for b in fr['all_boxes'] if b['actor_id']==s['actor']);c,k=camera(fr,s['camera']);rect=project_bbox(b['pose'],b['size_lwh'],c,k,(576,1024))
 p.add_new_points_or_box(state,frame_idx=0,obj_id=1,box=np.array(rect,np.float32))
 for i,_,logit in p.propagate_in_video(state):Image.fromarray(((logit[0,0]>0).cpu().numpy()*255).astype('uint8')).save(out/'sam'/f'{i:05}.png')
del p;torch.cuda.empty_cache()
allrows=[]
for s in reg['scenes']:
 out=ROOT/s['name'];frames=read(out/'frames.json');rows=[]
 for i,f in enumerate(s['source_frames']):
  fr=frames[str(f)];c,k=camera(fr,s['camera']);b=next(b for b in fr['all_boxes'] if b['actor_id']==s['actor'])
  sam=cv2.imread(str(out/'sam'/f'{i:05}.png'),0)>0;gate=hull_mask(b,c,k,pad=3)
  core=largest(sam&gate);mask=cv2.dilate(core.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0;mask &= gate
  assert mask.any(),(s['name'],f)
  cv2.imwrite(str(out/'mask'/f'{i:05}.png'),mask.astype('uint8')*255)
  cv2.imwrite(str(out/'sam'/f'core_{i:05}.png'),core.astype('uint8')*255)
  old=cv2.imread(str(FULL/s['name']/f"cam{s['camera']}/mask/{f:05}.png"),0)>0 if s['name']!='official_000' else None
  rows.append({'frame':f,'raw_sam':int(sam.sum()),'core':int(core.sum()),'edit':int(mask.sum()),'old':int(old.sum()) if old is not None else None,'sam_removed_by_consistency':int((sam&~core).sum())})
 sheet=Image.new('RGB',(1536,288*3))
 for j,i in enumerate([0,15,29]):
  rgb=np.array(Image.open(out/'rgb'/f'{i:05}.png'));mask=cv2.imread(str(out/'mask'/f'{i:05}.png'),0)>0;sam=cv2.imread(str(out/'sam'/f'{i:05}.png'),0)>0
  images=[rgb.copy(),rgb.copy(),rgb.copy()]
  if s['name']!='official_000':old=cv2.imread(str(FULL/s['name']/f"cam{s['camera']}/mask/{s['source_frames'][i]:05}.png"),0)>0
  else:old=sam
  images[1][old]=(.5*images[1][old]+.5*np.array([40,90,255])).astype('uint8');images[2][mask]=(.5*images[2][mask]+.5*np.array([0,235,130])).astype('uint8')
  for a,img in enumerate(images):sheet.paste(Image.fromarray(img).resize((512,288)),(a*512,j*288))
 sheet.save(out/'mask_comparison.jpg',quality=94);dump(out/'mask_stats.json',rows);allrows.append({'scene':s['name'],'frames':len(rows),'old_pixels':sum(r['old'] or 0 for r in rows),'new_pixels':sum(r['edit'] for r in rows),'core_pixels':sum(r['core'] for r in rows)})
dump(ROOT/'mask_summary.json',allrows);print(allrows)

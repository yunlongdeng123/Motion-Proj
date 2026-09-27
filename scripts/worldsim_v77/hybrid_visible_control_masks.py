"""r14已知可见actor52正控制：先准备原RGB上的SAM2实例mask。"""
import sys,datetime,time
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from repair_common import camera,hull_mask
from video_review import scene_frame
from PIL import ImageDraw
import torch
sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

BASE=ROOT.parent;R2=ROOT;ROOT=BASE/'r14';assert not ROOT.exists();ROOT.mkdir()
cfg=next(s for s in read(R2/'registration.json')['scenes'] if s['name']=='scene_0255');data=Path(cfg['data']);inst=read(data/'instances/instances_info.json')
records=[r for r in read(BASE/'r13/scene_0255/calibration.json')['records'] if r['admitted']]
dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r14',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='scene_0255',actor='52',target_actor='25',
 role='known-visible original RGB reconstruction control; query RGB only for evaluation, never as a donor to itself',
 hypothesis='per-pixel LiDAR proximity may reject usable calibrated Omega surfaces; validate on visible original pixels before considering actual DELETE',
 sources=[dict(frame=r['frame'],camera=r['camera'],scale=r['scale']) for r in records],query_same_time_donors_excluded=True,
 source_mask='SAM2.1-large official pretrained, original RGB GT actor52 box prompt, highest predicted score, retain alternatives',
 arms=['r13 strict LiDAR-distance-supported points','dense calibrated depth + source SAM + two-view agreement; diagnostic, not admitted automatically'],
 fixed=dict(depth_gradient_m=.3,box_size_padding_m=.1,depth_agreement_m=.25,rgb_max_error=25,source_time_gap_frames=5,splat=3,erosion=3),
 resources=dict(cpu_threads=4,expected_sam_image_forwards=len(records),new_generative_forward=0,new_omega_forward=0),failure_ledger_refs=['V77-F02'],human_verdict=None))
torch.set_num_threads(4);cv2.setNumThreads(4)
sam=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',device='cuda'))
rows=[];sheet=Image.new('RGB',(960*2,536*len(records)))
for i,r in enumerate(records):
 f,cam=r['frame'],r['camera'];out=ROOT/f'{f:03}_{cam}';out.mkdir();fr=scene_frame(cfg['spec'],f,inst);b=next(b for b in fr['all_boxes'] if b['actor_id']=='52');c,k=camera(fr,cam,HW);rect=project_bbox(b['pose'],b['size_lwh'],c,k,HW);im=rgb(data/'images'/f'{f:03}_{cam}.jpg')
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  sam.set_image(im);m,scores,_=sam.predict(box=np.array(rect),multimask_output=True)
 choice=int(np.argmax(scores));selected=m[choice]>0;np.savez_compressed(out/'sam_alternatives.npz',masks=m,scores=scores);write_mask(out/'sam.png',selected)
 Image.fromarray(im).save(out/'rgb.png');overlay=im.copy();overlay[selected]=(.4*im[selected]+.6*np.array([25,220,230])).astype('uint8');overlay=Image.fromarray(overlay);ImageDraw.Draw(overlay).rectangle(rect,outline='yellow',width=2)
 for j,a in enumerate([Image.fromarray(im),overlay]):
  ImageDraw.Draw(a).text((8,8),f'actor52 ORIGINAL f{f} CAM{cam} '+('raw' if j==0 else f'SAM choice{choice} score{scores[choice]:.3f}'),fill='yellow');sheet.paste(a,(W*j,H*i))
 rows.append(dict(frame=f,camera=cam,scale=r['scale'],box=rect,scores=scores.tolist(),choice=choice,mask_pixels=int(selected.sum())))
dump(ROOT/'mask_stats.json',rows);sheet.resize((1280,round(536*len(records)*2/3))).save(ROOT/'source_mask_review.jpg',quality=96)
dump(ROOT/'mask_state.json',dict(state='complete',sam_forwards=len(rows),human_verdict=None));print('VISIBLE CONTROL MASKS READY',len(rows),flush=True)

"""r41：仅把Hunyuan形状输入改为r18已保存的SUV侧前方先验，控制车身类型。"""
import sys,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/work');sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_retained_actor_asset as asset
from repair_common import dump,read
from PIL import Image
import numpy as np
ROOT=asset.BASE/'r41'
def prepare():
 assert not ROOT.exists();ROOT.mkdir();rgb=np.array(Image.open(asset.BASE/'r18/source_sv3d.png').convert('RGB'));m=np.array(Image.open(asset.BASE/'r18/source_silhouette.png').convert('L'))>127;y,x=np.where(m);box=(max(0,int(x.min())-12),max(0,int(y.min())-12),min(rgb.shape[1],int(x.max())+13),min(rgb.shape[0],int(y.max())+13));Image.fromarray(np.dstack([rgb,np.uint8(m)*255])).crop(box).save(ROOT/'source_rgba.png');Image.fromarray(rgb).crop(box).save(ROOT/'source_context.png');Image.fromarray(np.uint8(m)*255).crop(box).save(ROOT/'source_mask.png')
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r41',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='scene_0255',delete_actor='25',preserve_actor='52',parent_failure='r40 original-frontal-only shape creates open pickup bed instead of known closed SUV. Viewed8angles in Blender5.2.2 before any texture/placement; r40 paint not run.',question='Does complete body-view information already present in the r18 SV3D prior prevent the wrong pickup body in fixed Hunyuan?',only_change='Replace shape input original frontal f130 crop with saved r18 SV3D view12 at315deg. Same Hunyuan checkpoint,seed7740,50steps,guidance7.5,octree256 and12px mask-bbox padding.',source={'rgb':str(asset.BASE/'r18/source_sv3d.png'),'mask':str(asset.BASE/'r18/source_silhouette.png'),'crop_xyxy':box},roles='Generated SV3D prior, not real observed side evidence. No new SV3D call/model/seed. Real front f130 remains source identity reference; hidden surfaces still hallucinated. This is an explicit retained-actor POC, not a new accepted background.',stop_rule='One shape-input control, no seed sweep. Inspect closed body and then observed-view geometry before any hidden edit integration; preserve user r18 and r21 unchanged.',resources={'cpu_threads':4,'gpu':'existing RTX3090'},failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None));print('R41_PREPARED',flush=True)
if __name__=='__main__':
 if sys.argv[-1]=='prepare':prepare()
 else:asset.ROOT=ROOT;asset.shape()

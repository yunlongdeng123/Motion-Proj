"""r20：手工可核对的同一面板四角对齐，只作前景来源强控制。"""
import sys,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import read,dump
import cv2,numpy as np
from PIL import Image,ImageDraw
cv2.setNumThreads(4)
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');ROOT=BASE/'r20';DATA=Path('/root/autodl-tmp/data/v76_vadgs/scene_0255')
def main():
 assert not ROOT.exists();ROOT.mkdir()
 # 坐标来自助手查看原始1600×900图像，仅是可复核POC提示，不宣称自动或精确GT。
 panel65=np.array([[715,530],[790,536],[783,616],[712,609]],np.float32)
 panel90=np.array([[669,548],[720,553],[716,634],[664,628]],np.float32)
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r20',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='scene_0255',scope='Manual planar foreground alignment positive control before composition',
  inputs='Only original f65 and f90 CAM3 RGB, assistant visible panel corner prompts; original r8 geometry as an independent rail overlay.',
  hypothesis='If a checked same-panel homography aligns the actual thin rails, material/alpha can be recovered from original views without vehicle RGB fragments.',
  adaptation='OpenCV getPerspectiveTransform from 4 manual points; no camera/3D truth claim, no neural forward. Four fitted points are not validation; rails away from panel must be visually checked.',
  corners=dict(f65=panel65.tolist(),f90=panel90.tolist()),stop_rule='No composition unless visible non-panel rails align; preserve failed automatic r19.',resources=dict(cpu_threads=4,gpu_forwards=0),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 H=cv2.getPerspectiveTransform(panel90,panel65);im65=np.array(Image.open(DATA/'images/065_3.jpg').convert('RGB'));im90=np.array(Image.open(DATA/'images/090_3.jpg').convert('RGB'));aligned=cv2.warpPerspective(im90,H,(1600,900),flags=cv2.INTER_LINEAR)
 Image.fromarray(aligned).save(ROOT/'f90_aligned_to_f65.png');dump(ROOT/'homography.json',dict(H_90_to_65=H.tolist(),condition=float(np.linalg.cond(H))))
 geom=read(BASE/'r8/registration.json')['geometry'];scale=np.array([1600/960,900/536]);shift=scale*.5-.5
 crop=(570,490,980,670);sheet=Image.new('RGB',(1230,540*3),(15,20,30))
 for j,(title,im) in enumerate([('f65 original',im65),('f90 aligned to f65 (background parallax is expected)',aligned),('f90 aligned + fixed r8 rails',aligned)]):
  a=Image.fromarray(im)
  if j==2:
   d=ImageDraw.Draw(a)
   for l in geom['lines']+geom['bars']:d.line([tuple(p) for p in np.array(l)*scale+shift],fill='cyan',width=1)
  a=a.crop(crop).resize((1230,540));ImageDraw.Draw(a).text((8,8),title,fill='yellow');sheet.paste(a,(0,j*540))
 sheet.save(ROOT/'alignment_review.jpg',quality=96);print('R20_ALIGNMENT_READY',flush=True)
if __name__=='__main__':main()

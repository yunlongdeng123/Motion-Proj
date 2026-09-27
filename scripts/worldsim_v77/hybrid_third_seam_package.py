from pathlib import Path
import json,shutil,subprocess
import cv2,imageio_ffmpeg
from PIL import Image
BASE=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927');ROOT=BASE/'r29';OUT=BASE/'road_review';dest=OUT/'r29';dest.mkdir()
for name in ['all10_part1.jpg','all10_part2.jpg','registration.json','state.json']:shutil.copy2(ROOT/name,dest/name)
for name in ['mask_boundary_audit.json','mask_boundary_audit.jpg']:shutil.copy2(BASE/'r28'/name,OUT/'r28'/name)
shutil.copy2(BASE/'r27/plane_extent_audit.json',OUT/'r27/plane_extent_audit.json')
file=OUT/'seam.mp4';subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-framerate','10','-i',str(ROOT/'final/%05d.png'),'-c:v','libx264','-crf','17','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(file)],check=True)
cap=cv2.VideoCapture(str(file));count=0
while True:
 ok,im=cap.read()
 if not ok:break
 assert im.shape==(576,1024,3);count+=1
cap.release();assert count==10;Image.open(ROOT/'final/00000.png').save(OUT/'seam_poster.jpg',quality=96)
p=OUT/'validation.json';shutil.copy2(p,OUT/'validation.before_r29.json');result=json.loads(p.read_text());result['videos'].append(dict(file='seam.mp4',frames=10,fps=10));result['decoded_frames']+=10;result['r29_outside_write_checks']=json.loads((ROOT/'state.json').read_text())['checks'];p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('R29_PACKAGED',count)

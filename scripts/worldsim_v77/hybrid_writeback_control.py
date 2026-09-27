"""r8只改0255栏杆保护mask；冻结r6生成，隔离D→E写回污染。"""
import sys,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from hybrid_common import *
from PIL import ImageDraw
R2=ROOT;R3=ROOT.parent/'r3';R6=ROOT.parent/'r6';ROOT=ROOT.parent/'r8';NAME='scene_0255'
assert not (ROOT/'registration.json').exists();ROOT.mkdir(exist_ok=True)
lines=[[(371,309),(363,311),(358,316),(355,372),(358,379)],[(371,309),(536,321),(545,326),(548,371),(550,376)],[(357,358),(553,375),(566,377),(570,375)]]
bars=[[(374,310),(370,359)],[(385,311),(382,360)],[(397,312),(394,362)],[(409,313),(406,363)],[(420,314),(417,364)],[(485,318),(481,369)],[(496,319),(493,370)],[(508,320),(505,371)],[(519,320),(517,372)],[(531,321),(528,373)]]
panel=[(429,314),(474,318),(470,367),(427,363)]
dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r8',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene=NAME,source_run='r6',role='D to E composition control, zero model forward',change='replace mis-segmented foreground fence with assistant image-traced rails and panel, propagated by previously frozen LK similarity',manual=True,geometry=dict(lines=lines,bars=bars,panel=panel,rail_width=3,bar_width=2),fixed=['r6 generated PNG','r3 write envelope','source RGB','neighbor masks','other static masks','factual observations'],failure_ledger_refs=['V77-F02'],human_verdict=None))
old=R2/NAME;out=ROOT/NAME;out.mkdir()
for sub in ['fence','protect','final','no_static','mask_overlay']:(out/sub).mkdir()
fence0=np.zeros(HW,np.uint8)
for line in lines:cv2.polylines(fence0,[np.array(line,np.int32)],False,1,3)
for line in bars:cv2.polylines(fence0,[np.array(line,np.int32)],False,1,2)
cv2.fillPoly(fence0,[np.array(panel,np.int32)],1)
stats=read(old/'mask_stats.json');rows=[]
for i,r in enumerate(stats):
 h=np.array(r['fence_homography']);fence=cv2.warpPerspective(fence0,h,(W,H),flags=cv2.INTER_NEAREST)>0
 region=np.zeros(HW,np.uint8);region[299:391,347:578]=1;region=cv2.warpPerspective(region,h,(W,H),flags=cv2.INTER_NEAREST)>0
 static=mask(old/'static'/f'{i:05}.png');dyn=mask(old/'dynamic'/f'{i:05}.png');newstatic=(static&~region)|fence;protect=dyn|newstatic
 original=rgb(old/'rgb'/f'{i:05}.png');native=rgb(R6/NAME/'native_png'/f'{i:05}.png');envelope=mask(R3/NAME/'condition'/f'{i:05}.png');write=envelope&~protect;obs=mask(R3/NAME/'observed'/f'{i:05}.png')&~protect
 final=original.copy();final[write]=native[write];e=rgb(R3/NAME/'input'/f'{i:05}.png');final[obs]=e[obs]
 assert np.array_equal(final[protect],original[protect]);assert np.array_equal(final[~envelope],original[~envelope])
 nostatic=original.copy();w=envelope&~dyn;nostatic[w]=native[w]
 overlay=original.copy();overlay[region&static]=(.25*overlay[region&static]+.75*np.array([255,75,60])).astype('uint8');overlay[fence]=(.25*overlay[fence]+.75*np.array([0,240,240])).astype('uint8')
 for key,im in [('final',final),('no_static',nostatic),('mask_overlay',overlay)]:Image.fromarray(im).save(out/key/f'{i:05}.png')
 write_mask(out/'fence'/f'{i:05}.png',fence);write_mask(out/'protect'/f'{i:05}.png',protect)
 rows.append(dict(frame=65+i,old_fence_region_pixels=int((static&region).sum()),new_fence_pixels=int(fence.sum()),removed_protection=int((static&region&~fence&~dyn).sum()),added_protection=int((fence&~static&~dyn).sum()),protect_changed=0,outside_envelope_changed=0,neighbor_mask_changed=int(np.any(final!=original,axis=-1)[dyn].sum())))
sheet=Image.new('RGB',(420*5,236*4))
for j,i in enumerate([0,7,15,29]):
 y,x=np.where(mask(old/'core'/f'{i:05}.png'));cw=400;ch=round(cw*H/W);l=round(max(0,min(W-cw,(x.min()+x.max()-cw)/2)));t=round(max(0,min(H-ch,(y.min()+y.max()-ch)/2)))
 for k,(label,path) in enumerate([('D raw',R6/NAME/'native_png'),('E old r6',R6/NAME/'final'),('E no static (control)',out/'no_static'),('E r8 thin fence',out/'final'),('old red / new cyan',out/'mask_overlay')]):
  im=Image.fromarray(rgb(path/f'{i:05}.png')).crop((l,t,l+cw,t+ch)).resize((420,236));ImageDraw.Draw(im).text((5,5),f'{label} f{65+i}',fill='yellow');sheet.paste(im,(k*420,j*236))
sheet.save(out/'quicklook.jpg',quality=94);dump(out/'pixel_checks.json',rows);dump(ROOT/'state.json',dict(state='complete',frames=30,gpu_forwards=0,human_verdict=None));print('WRITEBACK CONTROL COMPLETE',flush=True)

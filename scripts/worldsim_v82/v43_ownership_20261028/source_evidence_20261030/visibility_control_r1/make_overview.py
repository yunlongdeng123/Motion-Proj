from pathlib import Path
from PIL import Image,ImageDraw
import json,numpy as np
p=Path('work/v43_sim_mechanism_20261030/visibility_control_r1');s=json.loads((p/'visibility_summary.json').read_text());t=np.load(p/'target_world_points.npz');v=np.load(p/'original_support.npz');canvas=Image.new('RGB',(1440,1180),'white');d=ImageDraw.Draw(canvas)
for i,f in enumerate([0,8,15]):
 for j,name in enumerate(['original','hybrid8_z3.5']):canvas.paste(Image.open(p/f'{name}_anchor{f:06d}_coverage.png'),(j*480,i*345+110))
 a=np.asarray(Image.open(p/f'original_anchor{f:06d}_coverage.png')).copy();u=t['anchor_frame']==f;yx=t['yx'][u];ld=v['lidar_nearest_distance_m'][u];colors=np.tile([240,40,30],(len(ld),1));colors[ld<=.25]=[255,205,30];colors[ld<=.1]=[0,210,80];a[yx[:,0]+25,yx[:,1]]=colors;a[:25]=255;im=Image.fromarray(a);ImageDraw.Draw(im).text((4,5),f'Old LiDAR proximity anchor{f}: green<=.1m yellow<=.25m',fill='black');canvas.paste(im,(960,i*345+110))
for i,label in enumerate(['Independent deleted road XYZ','16 calibrated original-world rays','View counts + baseline + LiDAR','16-view rich control precheck']):
 x=i*360;d.rectangle([x+5,5,x+340,65],outline='black',width=2);d.text((x+12,28),label,fill='black')
 if i<3:d.text((x+342,28),'->',fill='black')
d.text((10,80),'Geometric visibility only. Counts are old-anchor pixel samples, not unique world area; shadow contrast is only proxy.',fill='black')
d.text((10,1155),'Original 1+/2+ visible: 50.54%/47.58%. Hybrid8 preserving anchors: 100%/98.95%. Same 16 times, one camera/time.',fill='black');canvas.save(p/'visibility_control_overview.png')

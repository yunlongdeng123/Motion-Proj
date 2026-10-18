from pathlib import Path
import json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image,ImageDraw,ImageFont
E=Path(__file__).resolve().parent;P=E/'all_vehicle_pipeline';S=P/'inputs/scene0008_089_137';OUT=E/'intersection_input_review'
def main():
 assert not OUT.exists();receipt=json.loads((E/'lidar_camera_intersection_input_receipt.json').read_text());assert receipt['actual_CPU_cameras']==49
 OUT.mkdir();(OUT/'frames').mkdir();(OUT/'sheets').mkdir();font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);actors=receipt['selected'];records=[]
 for raw in range(89,138):
  fid=f'0008{raw:03d}';rgb=np.array(Image.open(S/'image/image_0'/(fid+'.png')).convert('RGB'));labels=np.array(Image.open(S/'intersection_labels'/(fid+'.png')));overlay=rgb.copy();texts=[]
  for i,a in enumerate(actors):
   m=labels==i+1;col=np.array([((i+1)*k)%220+30 for k in [71,43,103]]);overlay[m]=np.rint(.65*rgb[m]+.35*col).astype(np.uint8);ys,xs=np.where(m)
   if len(xs):texts.append((int(xs.mean()),max(0,int(ys.min())-22),'a'+str(a['R15_actor'])))
  im=Image.fromarray(overlay);d=ImageDraw.Draw(im)
  for x,y,t in texts:d.text((x,y),t,font=font,fill='yellow',stroke_width=2,stroke_fill='black')
  d.text((20,20),fid+' LiDAR + camera intersection only; pure-camera context excluded',font=font,fill='white',stroke_width=2,stroke_fill='black')
  panel=Image.new('RGB',(1920,2560));panel.paste(Image.fromarray(rgb),(0,0));panel.paste(im,(0,1280));panel.save(OUT/'frames'/(fid+'.jpg'),quality=88)
  records.append(dict(frame_id=fid,union_pixels=int((labels>0).sum()),active_actors=[a['R15_actor'] for i,a in enumerate(actors) if (labels==i+1).any()]))
 for batch in range(7):
  sheet=Image.new('RGB',(1920,7*660),(12,22,33))
  for row,r in enumerate(records[batch*7:batch*7+7]):
   # Two complete-frame views per row, resized without spatial cropping.
   panel=Image.open(OUT/'frames'/(r['frame_id']+'.jpg'))
   a=panel.crop((0,0,1920,1280)).resize((960,640),Image.Resampling.LANCZOS);b=panel.crop((0,1280,1920,2560)).resize((960,640),Image.Resampling.LANCZOS)
   sheet.paste(a,(0,row*660+20));sheet.paste(b,(960,row*660+20));ImageDraw.Draw(sheet).text((5,row*660),r['frame_id'],font=small,fill='white')
  sheet.save(OUT/'sheets'/f'{batch}.jpg',quality=88)
 (OUT/'receipt.json').write_text(json.dumps(dict(status='COMPLETE49_FULL_FRAME_INPUT_REVIEW_MATERIALS',frames=49,scope='LIDAR_AND_CAMERA_INTERSECTION_ONLY',records=records,ROI=False,training_updates=0,human_verdict=None),indent=2));print('complete49 full-frame panels and7 sheets')
if __name__=='__main__':main()

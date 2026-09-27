"""r26：冻结r25的实测路面与来源规则，检查actual DELETE洞的证据覆盖。"""
import sys,time,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hybrid_road_plane_control as p
from repair_common import read,dump
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=p.BASE/'r26';OLD=p.OLD/'official_000'

def main():
 assert not ROOT.exists();ROOT.mkdir();beg=time.time()
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r26',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='official_000',actor='12',camera=0,frames=list(range(30)),
  question='How much of the real target deletion region is covered by the exact measured road surface validated in r25?',
  inputs='Frozen r25 plane and training xy hull, original RGB/GT cameras/boxes, existing precise SAM2 deletion mask. No target RGB is admitted as donor at its own timestamp.',
  fixed=dict(source_offsets=[5,10,20],same_camera=0,depth_range_m=[3,50],rgb_consensus_max_channel_error=20,size=[1024,576],cpu_threads=4),
  controls='All settings from visible-road positive control unchanged; query domain becomes existing deletion mask. Do not extrapolate plane outside measured ground extent. Nearest available vs two-time RGB consensus.',
  limitations='Only measures supported road hypothesis. Does not certify visibility against unlabeled static occluders or all non-ground surfaces; no hidden RGB GT.',
  stop_rule='If road support is absent, do not broaden geometry thresholds or claim unseen background. Inspect the exact geometry-support mismatch before any compositing/generation.',resources=dict(gpu_forwards=0),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 fit=read(p.ROOT/'plane_fit.json');n=np.array(fit['normal']);d=fit['offset'];hull=np.array(fit['training_convex_hull_xy'],np.float32)
 yy,xx=np.mgrid[:576,:1024];pix=np.stack([xx,yy,np.ones_like(xx)],-1).reshape(-1,3);rows=[]
 for f in range(30):
  out=ROOT/f'f{f:03}';out.mkdir();fr,c,k,im,ex=p.get(f);mask=np.array(Image.open(OLD/'mask'/f'{f:05}.png').convert('L').resize((1024,576),Image.Resampling.NEAREST))>0
  rays=pix@np.linalg.inv(k).T@c[:3,:3].T;den=rays@n;z=-(c[:3,3]@n+d)/np.where(np.abs(den)>1e-8,den,np.nan);world=c[:3,3]+rays*z[:,None]
  domain=mask.ravel()&np.isfinite(z)&(z>3)&(z<50);ids=np.flatnonzero(domain);inside=np.zeros(len(world),bool);inside[ids]=[cv2.pointPolygonTest(hull,(float(world[j,0]),float(world[j,1])),False)>=0 for j in ids];support=inside.reshape(p.HW)
  warped=[];valid=[];source=[]
  for sf in [f+5,f+10,f+20]:
   _,sc,sk,sim,sex=p.get(sf);uv,sz=p.project(world,sc,sk);mx=uv[:,0].reshape(p.HW).astype('float32');my=uv[:,1].reshape(p.HW).astype('float32');v=support&(sz.reshape(p.HW)>.2)&np.isfinite(mx)&np.isfinite(my)&(mx>=0)&(mx<1023)&(my>=0)&(my<575)
   v&=~(cv2.remap(sex.astype('uint8'),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=1)>0);valid.append(v);warped.append(cv2.remap(sim,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT));source.append(dict(frame=sf,pixels=int(v.sum())))
  nearest=np.zeros_like(im);hit=np.zeros(p.HW,bool);consensus=np.zeros_like(im);chit=np.zeros(p.HW,bool)
  for j in range(3):take=valid[j]&~hit;nearest[take]=warped[j][take];hit|=valid[j]
  for a,b in [(0,1),(0,2),(1,2)]:
   agree=np.max(np.abs(warped[a].astype('float32')-warped[b]),-1)<=20;take=valid[a]&valid[b]&agree&~chit;consensus[take]=np.rint((warped[a][take].astype('float32')+warped[b][take])/2).astype('uint8');chit|=take
  for name,arr in [('original',im),('nearest_rgb',nearest),('consensus_rgb',consensus),('mask',mask),('road_support',support),('nearest_mask',hit),('consensus_mask',chit)]:Image.fromarray(np.uint8(arr)*255 if arr.dtype==bool else arr).save(out/f'{name}.png')
  diag=im.copy();diag[mask]=[110,20,140];diag[support]=[40,110,240];diag[hit]=nearest[hit];Image.fromarray(diag).save(out/'diagnostic.png')
  marked=im.copy();contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(marked,contours,-1,(255,220,30),2);Image.fromarray(marked).save(out/'target.png')
  row=dict(frame=f,mask_pixels=int(mask.sum()),plane_depth_valid_pixels=int(domain.sum()),measured_road_support_pixels=int(support.sum()),nearest_pixels=int(hit.sum()),consensus_pixels=int(chit.sum()),sources=source);dump(out/'metrics.json',row);rows.append(row)
 summary=dict(state='road_hole_coverage_audit_complete',seconds=time.time()-beg,frames=30,total_mask_pixels=sum(r['mask_pixels'] for r in rows),measured_support=sum(r['measured_road_support_pixels'] for r in rows),nearest=sum(r['nearest_pixels'] for r in rows),consensus=sum(r['consensus_pixels'] for r in rows),rows=rows,human_verdict=None,background_input_dir=None)
 dump(ROOT/'state.json',summary);print({k:v for k,v in summary.items() if k!='rows'},flush=True)
 sheet=Image.new('RGB',(1024,600*3),(15,20,30))
 for j,f in enumerate([0,15,29]):sheet.paste(Image.open(ROOT/f'f{f:03}/diagnostic.png'),(0,j*600+24));ImageDraw.Draw(sheet).text((10,j*600+5),f'f{f}: purple=hole without measured road support; blue=supported road, no RGB; original RGB donors elsewhere',fill='white')
 sheet.save(ROOT/'contact.jpg',quality=95)
if __name__=='__main__':main()

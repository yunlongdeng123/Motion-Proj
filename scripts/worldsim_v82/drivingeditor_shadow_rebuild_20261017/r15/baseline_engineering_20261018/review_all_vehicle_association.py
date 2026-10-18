"""Read-only, full-image proposal/annotation review with explicit unresolved IDs.

Boxes only identify measurements for review. They never create body pixels or
select the initialization point subset. No GPU or training updates.
"""
from pathlib import Path
import os,json,collections,itertools
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from PIL import Image,ImageDraw,ImageFont
E=Path(__file__).resolve().parent;ROOT=E.parent;SRC=ROOT/'inputs/scene0008_089_137'
R=ROOT/'shadow_rebuild_20261017_v3';OUT=E/'all_vehicle_association_review'
OLD=R/'ownership_controls_r12_sam3/shared_sam3_masks_r1/native_predictions'
NEW=E/'all_vehicle_native49/native_predictions'

def main():
 assert not OUT.exists(),'preserve the existing review'
 OUT.mkdir();(OUT/'frames').mkdir()
 coverage=json.loads((E/'all_vehicle_fullres_coverage/coverage.json').read_text())
 tids=sorted(v['track_id'] for v in coverage['vehicles'])
 oldids=json.loads((R/'backend_restore_r15/data/manifest.json').read_text())['actor_ids']
 names={t:('a'+str(oldids.index(t)) if t in oldids else 'extra'+str(tids.index(t))) for t in tids}
 stats={t:dict(name=names[t],raw_track_id=t,projected_box_frames=[],behind_or_outside=[],frames=[]) for t in tids}
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
 roles={r['frame_id']:r['split'] for r in json.loads((R/'backend_restore_r15/data/manifest.json').read_text())['frames']}
 covs={'new':coverage,'old':json.loads((E/'all_vehicle_coverage/coverage.json').read_text())}
 maps={}
 for source,cov in covs.items():
  maps[source]={k:dict(track_id=v[0][0],name=names[v[0][0]],votes=v[0][1],purity=v[0][1]/sum(n for _,n in v)) if v else None for k,v in cov['proposal_track_votes'].items()}
 for raw in range(89,138):
  fid=f'0008{raw:03d}';rgb=Image.open(SRC/'image/image_0'/(fid+'.png')).convert('RGB')
  cal={s.split(':',1)[0]:np.fromstring(s.split(':',1)[1],sep=' ') for s in (SRC/'calib'/(fid+'.txt')).read_text().splitlines() if ':' in s}
  tr=cal['Tr_velo_to_cam_0'].reshape(3,4);p=cal['P0'].reshape(3,4)
  annotated=rgb.copy();d=ImageDraw.Draw(annotated)
  ann=json.loads((SRC/'annotations'/(fid+'.json')).read_text())
  for a in ann['laser_labels']:
   if a['type']!='TYPE_VEHICLE':continue
   b=a['box'];center=np.array([b['center_x'],b['center_y'],b['center_z']]);dims=np.array([b['length'],b['width'],b['height']]);h=b['heading'];c,s=np.cos(h),np.sin(h)
   rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
   xyz=np.array(list(itertools.product([-.5,.5],repeat=3)))*dims@rot.T+center
   cam=np.c_[xyz,np.ones(8)]@tr.T;z=cam[:,2]
   # Only boxes entirely in front can yield a finite projected rectangle.
   if (z>.01).all():
    q=np.c_[cam,np.ones(8)]@p.T;uv=q[:,:2]/q[:,2:3]
    box=[float(uv[:,0].min()),float(uv[:,1].min()),float(uv[:,0].max()),float(uv[:,1].max())]
    visible=box[2]>=0 and box[0]<1920 and box[3]>=0 and box[1]<1280
   else:box=None;visible=False
   row=dict(frame_id=fid,projected_box=box,min_depth_m=float(z.min()),max_depth_m=float(z.max()),entire_box_in_front=bool((z>.01).all()),box_intersects_image=bool(visible),annotation_lidar_points=a['num_lidar_points_in_box'])
   stats[a['id']]['frames'].append(row)
   stats[a['id']]['projected_box_frames' if visible else 'behind_or_outside'].append(fid)
   if visible:
    bb=[max(0,box[0]),max(0,box[1]),min(1919,box[2]),min(1279,box[3])]
    d.rectangle(bb,outline='yellow',width=2);d.text((bb[0],max(0,bb[1]-22)),names[a['id']],font=font,fill='yellow',stroke_width=2,stroke_fill='black')
  panels=[annotated]
  for source,path in [('new',NEW),('old',OLD)]:
   z=np.load(path/(fid+'.npz'));ov=np.array(rgb);base=ov.copy();dr=[]
   for m,tid in zip(z['masks'],z['tracker_ids']):
    if m.shape!=(1280,1920):m=np.asarray(Image.fromarray(m).resize((1920,1280),Image.Resampling.NEAREST))
    key='train:'+str(int(tid)) if source=='new' or roles[fid]=='train' else fid+':'+str(int(tid))
    association=maps[source].get(key);color=np.array([(int(tid)*71+30)%220+30,(int(tid)*43+70)%220+30,(int(tid)*103+20)%220+30])
    ov[m]=np.rint(.65*base[m]+.35*color).astype(np.uint8)
    ys,xs=np.where(m)
    if len(xs):dr.append((int(xs.mean()),int(ys.min()),str(tid)+':'+(association['name'] if association else '?')))
   im=Image.fromarray(ov);dd=ImageDraw.Draw(im)
   for x,y,label in dr:dd.text((x,max(0,y-22)),label,font=font,fill='yellow',stroke_width=2,stroke_fill='black')
   dd.text((20,20),fid+' '+source+' native proposals; labels = candidate only',font=font,fill='white',stroke_width=2,stroke_fill='black')
   panels.append(im)
  panel=Image.new('RGB',(1920,1280*3))
  for i,im in enumerate(panels):panel.paste(im,(0,1280*i))
  panel.save(OUT/'frames'/(fid+'.jpg'),quality=88)
 (OUT/'review.json').write_text(json.dumps(dict(status='CANDIDATES_REQUIRING_FULL49_REVIEW',vehicles=list(stats.values()),proposal_candidates=maps,boxes_create_body_pixels=False,human_verdict=None),indent=2))
 print(json.dumps(dict(projected=[(x['name'],len(x['projected_box_frames'])) for x in stats.values()],training_updates=0)))
if __name__=='__main__':main()

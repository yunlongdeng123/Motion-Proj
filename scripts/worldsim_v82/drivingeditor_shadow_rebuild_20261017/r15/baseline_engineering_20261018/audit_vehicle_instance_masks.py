"""Read-only native instance-mask association and real seed audit.

Initialization candidates use complete masks and the unchanged 2x2px/.5m
front-depth + .5m dominant-component filter. Annotation boxes only diagnose
identity and reject an entire unreliable observation. No per-point box crop.
No training gate is written by this inventory.
"""
from pathlib import Path
import os,json,argparse,collections
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
E=Path(__file__).resolve().parent;SRC=E.parent/'inputs/scene0008_089_137';NATIVE=E/'all_vehicle_instance_prompt49/native_predictions'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
 OUT=E/('all_vehicle_instance_audit_partial' if args.partial else 'all_vehicle_instance_audit')
 assert not OUT.exists(),'preserve previous audit'
 paths=sorted(NATIVE.glob('*.json'));assert paths and (args.partial or len(paths)==49)
 OUT.mkdir();(OUT/'frames').mkdir();(OUT/'labels').mkdir();(OUT/'source_lidar_indices').mkdir()
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
 review=json.loads((E/'all_vehicle_association_review/review.json').read_text());vehicles={v['raw_track_id']:dict(actor=v['name'],raw_track_id=v['raw_track_id'],projected_frames=len(v['projected_box_frames']),mask_frames=0,accepted_real_seed_points=0,frames=[]) for v in review['vehicles'] if v['projected_box_frames']}
 order=sorted(vehicles);frames=[]
 for path in paths:
  fid=path.stem;native=json.loads(path.read_text());z=np.load(path.with_suffix('.npz'));masks=z['masks'];sc=z['scores']
  anns={a['id']:a for a in json.loads((SRC/'annotations'/(fid+'.json')).read_text())['laser_labels'] if a['type']=='TYPE_VEHICLE'}
  xyz=np.fromfile(SRC/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6)[:,:3].astype(float)
  cal={s.split(':',1)[0]:np.fromstring(s.split(':',1)[1],sep=' ') for s in (SRC/'calib'/(fid+'.txt')).read_text().splitlines() if ':' in s}
  cam=np.c_[xyz,np.ones(len(xyz))]@cal['Tr_velo_to_cam_0'].reshape(3,4).T
  positive=np.flatnonzero(cam[:,2]>.01);q=np.c_[cam[positive],np.ones(len(positive))]@cal['P0'].reshape(3,4).T;uv=np.floor(q[:,:2]/q[:,2:3]+.5).astype(int)
  keep=(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1280);idx=positive[keep];uv=uv[keep];depth=cam[idx,2];bins=(uv[:,1]//2)*960+uv[:,0]//2
  nearest=np.full(960*640,np.inf);np.minimum.at(nearest,bins,depth);vis=depth<=nearest[bins]+.5
  idx=idx[vis];uv=uv[vis];boxes={};box_hits={}
  for t,a in anns.items():
   b=a['box'];h=b['heading'];c,s=np.cos(h),np.sin(h);rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
   boxes[t]=b;box_hits[t]=((((xyz[idx]-[b['center_x'],b['center_y'],b['center_z']])@rot).__abs__())<=np.array([b['length'],b['width'],b['height']])/2).all(1)
  labels=np.zeros((1280,1920),np.uint16);rgb=np.array(Image.open(SRC/'image/image_0'/(fid+'.png')).convert('RGB'));overlay=rgb.copy();selected=[];sources={}
  for row in native['queries']:
   t=row['track_id'];b=boxes[t];bound=np.linalg.norm([b['length'],b['width'],b['height']])/2+2;center=np.array([b['center_x'],b['center_y'],b['center_z']]);candidates=[]
   for j in range(row['proposal_offset'],row['proposal_offset']+row['proposal_count']):
    m=masks[j];hit=m[uv[:,1],uv[:,0]];retained=idx[hit];initial=len(retained);reject=None;dominance=None
    votes={other:int((hit&bh).sum()) for other,bh in box_hits.items()};votes={other:n for other,n in votes.items() if n};target=votes.get(t,0);others=sum(n for other,n in votes.items() if other!=t)
    if initial:
     edges=cKDTree(xyz[retained]).query_pairs(.5,output_type='ndarray');graph=coo_matrix((np.ones(len(edges),np.uint8),(edges[:,0],edges[:,1])),shape=(initial,initial));_,cc=connected_components(graph,directed=False);count=np.bincount(cc);dominant=cc==count.argmax();dominance=float(dominant.mean())
     if dominance<.5:retained=retained[:0];reject='entire_observation_no_dominant_component'
     else:retained=retained[dominant]
    maxdist=float(np.linalg.norm(xyz[retained]-center,axis=1).max()) if len(retained) else None
    if maxdist is not None and maxdist>bound:retained=retained[:0];reject='entire_observation_physical_geometry_QA'
    purity=target/(target+others) if target+others else None
    ys,xs=np.where(m);bb=row['prompt_box_xyxy'];fraction=float(((xs>=bb[0])&(xs<=bb[2])&(ys>=bb[1])&(ys<=bb[3])).mean()) if len(xs) else 0
    confidence='supported' if target>0 and target>=others and fraction>=.5 else 'unresolved'
    score=(1 if confidence=='supported' else 0, target/(1+others),float(sc[j]),len(retained))
    candidates.append(dict(native_index=j,pixels=int(m.sum()),predicted_quality=float(sc[j]),target_front_LiDAR_hits=target,other_annotation_hits=others,purity=purity,fraction_in_prompt_rectangle=fraction,initial_real_points=initial,dominant_fraction=dominance,accepted_real_seed_points=len(retained),max_distance_to_annotated_center_m=maxdist,diagnostic_bound_m=float(bound),whole_observation_rejected=reject,association=confidence,raw_votes=sorted(votes.items(),key=lambda x:-x[1]),_score=score,_retained=retained))
   best=max(candidates,key=lambda r:r['_score']);m=masks[best['native_index']];v=vehicles[t];overlap=int(((labels>0)&m).sum())
   # Keep source masks intact for review. Labels are a candidate composite only;
   # overlap is reported, never silently made into a training-ready annotation.
   label=order.index(t)+1;labels[m]=label;col=np.array([(label*71)%220+30,(label*43)%220+30,(label*103)%220+30]);overlay[m]=np.rint(.7*rgb[m]+.3*col).astype(np.uint8)
   record=dict(frame_id=fid,actor=row['actor'],native_index=best['native_index'],association=best['association'],pixels=best['pixels'],accepted_real_seed_points=best['accepted_real_seed_points'],target_front_LiDAR_hits=best['target_front_LiDAR_hits'],overlap_with_prior_selected_pixels=overlap,seedless_or_unresolved=not best['accepted_real_seed_points'] or best['association']!='supported',candidates=[{k:val for k,val in c.items() if not k.startswith('_')} for c in candidates])
   v['frames'].append(record);v['mask_frames']+=best['pixels']>0;v['accepted_real_seed_points']+=best['accepted_real_seed_points'];sources[t]=best['_retained'].astype(np.int32);selected.append(record)
  Image.fromarray(labels).save(OUT/'labels'/(fid+'.png'));np.savez_compressed(OUT/'source_lidar_indices'/(fid+'.npz'),**sources)
  im=Image.fromarray(overlay);d=ImageDraw.Draw(im)
  for row,record in zip(native['queries'],selected):
   m=masks[record['native_index']];ys,xs=np.where(m)
   if len(xs):d.text((int(xs.mean()),max(0,int(ys.min())-24)),row['actor']+(' ?' if record['seedless_or_unresolved'] else ''),font=font,fill='yellow',stroke_width=2,stroke_fill='black')
  panel=Image.new('RGB',(1920,2560));panel.paste(Image.fromarray(rgb),(0,0));panel.paste(im,(0,1280));panel.save(OUT/'frames'/(fid+'.jpg'),quality=88)
  frames.append(dict(frame_id=fid,queries=selected));print(fid,flush=True)
 result=dict(status='INSTANCE_MASK_AND_REAL_SEED_INVENTORY_REQUIRES_VISUAL_REVIEW',frames=len(frames),vehicles=list(vehicles.values()),label_actor_order=order,geometry_rule='unchanged2x2px/.5m_front_depth,.5m_dominant_component,whole_observation_geometry_QA',GT_boxes_filter_individual_initialization_points=False,all_RGB_retained=True,readiness=False,human_verdict=None,training_updates=0)
 (OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps([(v['actor'],v['mask_frames'],v['accepted_real_seed_points']) for v in vehicles.values()]))
if __name__=='__main__':main()
